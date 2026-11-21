#!/usr/bin/env python3
"""Turns an STL or OBJ mesh into a solid set of voxels and writes them, with the joints between neighboring voxels, as JSON for Unity and the AVBD package.

    python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1
    python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --resolution 64 --zup-to-yup
    python3 Tools/Voxelize/voxelize_stl.py model.obj -o model.voxels.json --voxel-size 0.1

The input is a binary or ASCII STL, or a Wavefront OBJ; the file extension (.obj, otherwise STL) tells which.

The grid is given either by the edge length of a voxel (--voxel-size, in the units of the output, that is of the STL times --scale) or by the number of
voxels along the longest side of the mesh (--resolution). The grid starts at the minimum corner of the mesh.

The result is solid: the voxels fill the whole inside of the mesh, also if the STL is only a surface, and also the closed cavities inside it. A voxel
is solid when it is inside the outer surface:
  * --surface center (default): the voxel's center is inside the mesh. The size of the voxel object matches the mesh on average; a part thinner than a voxel
    can vanish.
  * --surface touch: the voxel touches the mesh or is inside it. The object is up to one voxel larger; nothing is lost.
The fill needs a closed mesh: the outside is found by flooding the grid from its corner, so a hole in the surface bigger than a voxel lets the flood into the
inside and leaves the model hollow (the script warns when the mesh has open edges). The center test counts the crossings of a ray with the mesh, so it
does not depend on the direction of the triangles.

Each voxel is a box of voxel-size edges, and every voxel is linked to its immediate neighbors (--connectivity 6: the voxels that share a face; 18 adds
the ones that share an edge; 26 the ones that share a corner) by one ball-socket joint, once per pair. The joint's anchor is the middle between the two
voxel centers, which is where the faces meet for the 6-neighborhood.

Output (JSON, see README.md for the format): the grid (origin, voxel size, dimensions), the voxel indices as one flat list of (i, j, k) triples, and the
joints as one flat list of pairs of voxel numbers (positions in the voxel list). The center of voxel (i, j, k) is origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize.

Colors (OBJ only): a surface voxel, that is a voxel with a face that no other voxel shares, gets the color of the mesh at the point of the nearby triangles that is
closest to the center of the voxel, from a texture (map_Kd of the MTL file, with the texture coordinates of the faces; needs the Pillow package, without it the material's Kd is used), else
from the vertex colors (`v x y z r g b`), else from the diffuse color Kd of the material. Voxels below the surface, and surface voxels without any color in the file,
get none. They are written as `colorVoxels` (voxel numbers) and `colors` (red, green, blue as 0..255 integers, sRGB as in the file); --no-colors leaves them out.

Needs Python 3 and numpy. Reads binary and ASCII STL and Wavefront OBJ (vertices and faces only: polygons are split into triangles as a fan, which is right for
the usual triangles and convex quads; normals, texture coordinates, materials, groups and objects are ignored, so all objects of a file become one mesh).
"""
import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np

FORMAT_NAME = "mckrill-avbd-voxels"
FORMAT_VERSION = 1

PAD = 2          # empty cells around the grid for the flood fill
ALIGNMENT_SHIFT = np.array([1.0e-3, 7.07e-4, 5.77e-4])  # in voxels, see voxelize()
TOUCH_EPS = 1e-6  # in voxels: a triangle that only grazes a voxel counts as touching it
COLOR_TOUCH = 1e-2  # in voxels: a surface voxel for the colors is one that a triangle touches to within this distance (a face of the mesh that lies on the
                    # boundary of the grid, shifted out of the last voxel by ALIGNMENT_SHIFT, belongs to that voxel)
MAX_ELEMS = 300_000  # cells of a batch (triangles x cells of their boxes) in the vectorized loops


# ---- reading -------------------------------------------------------------------------------------------------------

def read_stl(path):
    """Returns the triangles of an STL file as a float64 array of shape (T, 3, 3): triangle, vertex, xyz."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * count:  # binary: an 80 byte header, a count, 50 bytes per triangle
            record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            triangles = np.frombuffer(data, dtype=record, count=count, offset=84)["vertices"].astype(np.float64)
            return np.ascontiguousarray(triangles)
    text = data.decode("utf-8", errors="replace")
    numbers = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not numbers:
        raise ValueError(path + ": not an STL file (neither a binary STL of the right size nor an ASCII STL with 'vertex' lines)")
    if len(numbers) % 3 != 0:
        raise ValueError(path + ": the number of vertices is not a multiple of 3")
    return np.array(numbers, dtype=np.float64).reshape(-1, 3, 3)


class ColorSource:
    """The colors of a mesh, aligned with its triangles: per triangle either a texture (with the texture coordinates of its corners), the colors of its
    three vertices, or one color of its material. `sample` gives the color at a point of a triangle."""

    def __init__(self, uv, vertex_colors, tri_material, materials, warnings):
        self.uv = uv                          # (T, 3, 2) float64 texture coordinates of the corners, NaN where the face has none
        self.vertex_colors = vertex_colors    # (T, 3, 3) float64 colors 0..1 of the corners, NaN where the vertices have none
        self.tri_material = tri_material      # (T,) int, the index in materials, or -1
        self.materials = materials            # list of (kd or None, texture or None): kd is rgb 0..1, texture an array (H, W, 3) of uint8
        self.warnings = warnings              # what was not found or could not be read

    def has_colors(self):
        """True if some triangle has a color."""
        has_vertex = ~np.isnan(self.vertex_colors[:, 0, 0])
        for m, (kd, texture) in enumerate(self.materials):
            if kd is not None or texture is not None:
                if np.any(self.tri_material == m):
                    return True
        return bool(has_vertex.any())

    def sample(self, tri, bary):
        """Colors (n, 3), 0..1 (sRGB as written in the file), at the points with barycentric coordinates bary (n, 3) in the triangles tri (n,). NaN where
        the triangle has no color. A texture is used when the triangle has texture coordinates, then vertex colors, then the diffuse color of the material."""
        n = len(tri)
        out = np.full((n, 3), np.nan)
        mat = self.tri_material[tri]
        # the lowest priority first, so that a higher one overwrites it
        for m, (kd, texture) in enumerate(self.materials):
            if kd is not None:
                out[mat == m] = kd
        vc = self.vertex_colors[tri]                                   # (n, 3, 3)
        has_vc = ~np.isnan(vc[:, 0, 0])
        if has_vc.any():
            out[has_vc] = np.einsum("nc,ncd->nd", bary[has_vc], vc[has_vc])
        uv = np.einsum("nc,ncd->nd", bary, self.uv[tri])               # (n, 2)
        for m, (kd, texture) in enumerate(self.materials):
            if texture is None:
                continue
            use = (mat == m) & ~np.isnan(uv[:, 0])
            if use.any():
                out[use] = _sample_texture(texture, uv[use])
        return out


def _sample_texture(texture, uv):
    """Bilinear sample of an (H, W, 3) uint8 image at the texture coordinates uv (n, 2), repeating outside 0..1; v = 0 is the bottom row."""
    h, w = texture.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = ((1.0 - uv[:, 1]) % 1.0) * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    t = texture.astype(np.float64) / 255.0
    return (t[y0, x0] * (1 - fx) * (1 - fy) + t[y0, x1] * fx * (1 - fy) + t[y1, x0] * (1 - fx) * fy + t[y1, x1] * fx * fy)


def _read_mtl(path, warnings, cache):
    """The materials of an MTL file: {name: [kd or None, texture file or None]}, and loads the textures into cache {file: array or None}."""
    materials = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError as error:
        warnings.append("cannot read the material file %s: %s" % (path, error))
        return materials
    current = None
    for line in text.replace("\\\r\n", " ").replace("\\\n", " ").splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "newmtl":
            current = line.split("#", 1)[0].strip()[len("newmtl"):].strip()
            materials[current] = [None, None]
        elif current is not None and key == "kd" and len(parts) >= 4:
            try:
                materials[current][0] = np.clip([float(parts[1]), float(parts[2]), float(parts[3])], 0.0, 1.0)
            except ValueError:
                warnings.append("%s: the diffuse color of '%s' is not made of numbers" % (os.path.basename(path), current))
        elif current is not None and key == "map_kd" and len(parts) >= 2:
            # the file name is the last word; a name with spaces (and no options) is the rest of the line
            rest = line.split("#", 1)[0].strip()[len("map_Kd"):].strip()
            materials[current][1] = parts[-1] if os.path.exists(os.path.join(os.path.dirname(path), parts[-1])) or not os.path.exists(os.path.join(os.path.dirname(path), rest)) else rest
    return materials


def _load_texture(directory, name, warnings, cache):
    path = os.path.join(directory, name.replace("\\", "/"))
    if path in cache:
        return cache[path]
    texture = None
    try:
        from PIL import Image  # optional: only textures need it
    except ImportError:
        warnings.append("the texture %s is not used: reading images needs the Pillow package (pip install pillow); the diffuse color of the material is used" % name)
    else:
        try:
            with Image.open(path) as image:
                texture = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except (OSError, ValueError) as error:
            warnings.append("cannot read the texture %s (%s); the diffuse color of the material is used" % (name, error))
    cache[path] = texture
    return texture


def read_obj_colored(path):
    """Returns (triangles, colors) of a Wavefront OBJ file: the triangles as a float64 array of shape (T, 3, 3): triangle, vertex, xyz, and a ColorSource for them
    (its has_colors() is False if the file has no colors: no vertex colors, no material with a diffuse color or texture).

    Only the geometry and the colors are read. Vertices are `v x y z` (and `v x y z r g b`, the vertex color extension, 0..1 or 0..255), faces `f` with corners
    `v`, `v/vt`, `v//vn` or `v/vt/vn` (an index is 1-based, or negative to count back from the last one defined so far); a polygon with more than three corners is
    split into a fan. Colors come from the textures (`vt`, `mtllib`, `usemtl`, and `map_Kd` in the MTL file, which needs Pillow), the vertex colors, and the diffuse
    colors (`Kd`) of the materials, in that order of preference for a triangle. Lines, points, normals and the rest are skipped.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")  # a line that ends with a backslash goes on in the next one
    directory = os.path.dirname(os.path.abspath(path))

    vertices = []
    vertex_colors = []   # one per vertex: a list of three numbers or None
    texcoords = []
    triangles = []
    tri_uv = []          # per triangle: three texture coordinate indices or None
    tri_mat = []
    mtl_files = []
    material_names = []  # the names that usemtl used, in order of first use
    current_material = -1
    warnings = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].split()
        if not line:
            continue
        if line[0] == "v":
            if len(line) < 4:
                raise ValueError("%s, line %d: a vertex needs three numbers" % (path, number))
            try:
                vertices.append([float(line[1]), float(line[2]), float(line[3])])
                vertex_colors.append([float(line[4]), float(line[5]), float(line[6])] if len(line) >= 7 else None)
            except ValueError:
                raise ValueError("%s, line %d: a vertex has a coordinate that is not a number" % (path, number))
        elif line[0] == "vt":
            try:
                texcoords.append([float(line[1]), float(line[2]) if len(line) > 2 else 0.0])
            except (ValueError, IndexError):
                raise ValueError("%s, line %d: a texture coordinate needs a number" % (path, number))
        elif line[0] == "mtllib" and len(line) >= 2:
            mtl_files.extend(line[1:])
        elif line[0] == "usemtl" and len(line) >= 2:
            name = " ".join(line[1:])
            if name not in material_names:
                material_names.append(name)
            current_material = material_names.index(name)
        elif line[0] == "f":
            if len(line) < 4:
                raise ValueError("%s, line %d: a face needs at least three vertices" % (path, number))
            corners = []
            uvs = []
            for token in line[1:]:
                fields = token.split("/")
                try:
                    index = int(fields[0])
                except ValueError:
                    raise ValueError("%s, line %d: '%s' is not a vertex index" % (path, number, token))
                index = index - 1 if index > 0 else len(vertices) + index
                if index < 0 or index >= len(vertices):
                    raise ValueError("%s, line %d: the face refers to vertex %s, which is not defined (yet)" % (path, number, fields[0]))
                corners.append(index)
                uv = None
                if len(fields) > 1 and fields[1] != "":
                    try:
                        uv = int(fields[1])
                    except ValueError:
                        raise ValueError("%s, line %d: '%s' is not a texture coordinate index" % (path, number, token))
                    uv = uv - 1 if uv > 0 else len(texcoords) + uv
                    if uv < 0 or uv >= len(texcoords):
                        raise ValueError("%s, line %d: the face refers to texture coordinate %s, which is not defined (yet)" % (path, number, fields[1]))
                uvs.append(uv)
            for k in range(1, len(corners) - 1):
                triangles.append((corners[0], corners[k], corners[k + 1]))
                picked = (uvs[0], uvs[k], uvs[k + 1])
                tri_uv.append(picked if None not in picked else None)
                tri_mat.append(current_material)
    if not triangles:
        raise ValueError(path + ": no faces found (not an OBJ file, or one without faces)")

    corner_index = np.array(triangles, dtype=np.int64)
    positions = np.array(vertices, dtype=np.float64)[corner_index]

    # the vertex colors: 0..1, or 0..255 if any is above 1
    per_vertex = np.full((len(vertices), 3), np.nan)
    for v, color in enumerate(vertex_colors):
        if color is not None:
            per_vertex[v] = color
    if np.nanmax(per_vertex, initial=0.0) > 1.0 + 1e-9:
        per_vertex /= 255.0
    per_vertex = np.clip(per_vertex, 0.0, 1.0)
    tri_vertex_colors = per_vertex[corner_index]                     # NaN for a vertex without a color
    tri_vertex_colors[np.isnan(tri_vertex_colors).any(axis=(1, 2))] = np.nan  # all three corners need one

    # the materials
    defined = {}
    cache = {}
    for name in mtl_files:
        mtl_path = os.path.join(directory, name.replace("\\", "/"))
        for key, value in _read_mtl(mtl_path, warnings, cache).items():
            defined[key] = value
    materials = []
    for name in material_names:
        if name not in defined:
            if mtl_files:
                warnings.append("the material '%s' is not in the material files" % name)
            else:
                warnings.append("the material '%s' is used but the file has no mtllib" % name)
            materials.append((None, None))
            continue
        kd, texture_file = defined[name]
        texture = _load_texture(directory, texture_file, warnings, cache) if texture_file else None
        materials.append((kd, texture))

    uv = np.full((len(triangles), 3, 2), np.nan)
    if texcoords:
        table = np.array(texcoords, dtype=np.float64)
        for t, picked in enumerate(tri_uv):
            if picked is not None:
                uv[t] = table[list(picked)]
    return positions, ColorSource(uv, tri_vertex_colors, np.array(tri_mat, dtype=np.int64), materials, warnings)


def read_obj(path):
    """Returns the triangles of a Wavefront OBJ file as a float64 array of shape (T, 3, 3): triangle, vertex, xyz (see read_obj_colored)."""
    return read_obj_colored(path)[0]


def read_mesh(path):
    """Reads the triangles of a mesh file: a Wavefront OBJ if the name ends with .obj (any case), otherwise an STL (binary or ASCII)."""
    if os.path.splitext(path)[1].lower() == ".obj":
        return read_obj(path)
    return read_stl(path)


def read_mesh_colored(path):
    """Like read_mesh, but returns (triangles, colors): the ColorSource of an OBJ, and None for an STL (an STL has no colors)."""
    if os.path.splitext(path)[1].lower() == ".obj":
        return read_obj_colored(path)
    return read_stl(path), None


def transform(triangles, scale, zup_to_yup):
    """Scales the mesh and, for a mesh from a z-up tool, turns it into Unity's y-up frame: (x, y, z) -> (x, z, -y), a rotation (it keeps the winding)."""
    t = triangles * scale
    if zup_to_yup:
        t = np.stack([t[..., 0], t[..., 2], -t[..., 1]], axis=-1)
    return t


# ---- checks --------------------------------------------------------------------------------------------------------

def open_edge_count(triangles):
    """The number of edges that do not belong to exactly two triangles (after welding the vertices that coincide): 0 for a closed mesh."""
    flat = triangles.reshape(-1, 3)
    extent = max(float(np.ptp(flat, axis=0).max()), 1e-12)
    quantized = np.round(flat / (extent * 1e-7)).astype(np.int64)
    _, ids = np.unique(quantized, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    edges = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]  # an edge of a collapsed triangle
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts != 2))


def mesh_volume(triangles):
    """The volume of a closed mesh (the sum of the signed tetrahedra to the origin; the absolute value, so the winding does not matter)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


# ---- batches -------------------------------------------------------------------------------------------------------

def _groups(lo, hi):
    """Groups triangles whose cell boxes (lo..hi, inclusive) have the same size, so that a group can be done with one array operation.
    Yields (indices, (nx, ny, nz)) with the group split so that a batch is at most MAX_ELEMS cells in all."""
    dims = hi - lo + 1
    unique, inverse = np.unique(dims, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
    for g in range(len(unique)):
        indices = order[boundaries[g]:boundaries[g + 1]]
        cells = int(np.prod(unique[g]))
        step = max(1, MAX_ELEMS // cells)
        for s in range(0, len(indices), step):
            yield indices[s:s + step], tuple(int(x) for x in unique[g])


# ---- the surface: the voxels a triangle touches ------------------------------------------------------------------

def _overlap(v, centers, h):
    """Triangle against box (separating axis theorem, Akenine-Moller): v has shape (G, 3, 3), centers (G, C, 3), the half size of the boxes is h.
    Returns a boolean (G, C): the triangle and the box have a point in common."""
    d = [v[:, i, None, :] - centers for i in range(3)]          # the vertices relative to the box centers, (G, C, 3)
    separated = np.zeros(centers.shape[:2], dtype=bool)
    for axis in range(3):                                      # the faces of the box
        lo = np.minimum(np.minimum(d[0][..., axis], d[1][..., axis]), d[2][..., axis])
        hi = np.maximum(np.maximum(d[0][..., axis], d[1][..., axis]), d[2][..., axis])
        separated |= (lo > h) | (hi < -h)
    e = [v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2]]   # the edges, (G, 3)
    n = np.cross(e[0], e[1])                                   # the plane of the triangle
    dist = np.einsum("gk,gck->gc", n, d[0])
    separated |= np.abs(dist) > h * np.abs(n).sum(axis=1)[:, None]
    for edge in e:                                             # the edges against the three axes of the box
        ex, ey, ez = edge[:, 0], edge[:, 1], edge[:, 2]
        zero = np.zeros_like(ex)
        for axis_vector in ((zero, ez, -ey), (-ez, zero, ex), (ey, -ex, zero)):
            a = np.stack(axis_vector, axis=1)                  # (G, 3)
            p = [np.einsum("gk,gck->gc", a, d[i]) for i in range(3)]
            r = h * np.abs(a).sum(axis=1)[:, None]
            separated |= (np.minimum(np.minimum(p[0], p[1]), p[2]) > r) | (np.maximum(np.maximum(p[0], p[1]), p[2]) < -r)
    return ~separated


def _touching(tris_u, dims, pad=PAD, tolerance=TOUCH_EPS):
    """Yields (triangles, cells) for batches of the triangle/cell pairs that touch: the indices of the triangles (n,) and the cells (n, 3) at index c + pad, as in
    surface_voxels. Cells farther than pad outside the grid are left out."""
    lo = np.floor(tris_u.min(axis=1) - tolerance).astype(np.int64)
    hi = np.floor(tris_u.max(axis=1) + tolerance).astype(np.int64)
    lo = np.maximum(lo, -pad)
    hi = np.minimum(hi, np.array(dims) + pad - 1)
    keep = np.nonzero(np.all(hi >= lo, axis=1))[0]
    tris_k, lo, hi = tris_u[keep], lo[keep], hi[keep]
    for indices, (nx, ny, nz) in _groups(lo, hi):
        offsets = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1).reshape(-1, 3)
        cells = lo[indices][:, None, :] + offsets[None, :, :]                  # (G, C, 3)
        hits = _overlap(tris_k[indices], cells + 0.5, 0.5 + tolerance)
        which = np.nonzero(hits)[0]                                            # the row (triangle) of every hit
        yield keep[indices][which], cells[hits] + pad


def surface_voxels(tris_u, dims, pad=PAD):
    """The cells that the triangles touch. tris_u are the triangles in cell units (a cell c is [c, c + 1]); the result is a boolean array of the
    shape dims + 2 * pad, in which cell c is at index c + pad. Cells farther than pad outside the grid are not kept."""
    shape = tuple(int(d) + 2 * pad for d in dims)
    surface = np.zeros(shape, dtype=bool)
    for _, touched in _touching(tris_u, dims, pad):
        surface[touched[:, 0], touched[:, 1], touched[:, 2]] = True
    return surface


# ---- the colors of the surface voxels ------------------------------------------------------------------------------

def closest_barycentric(p, a, b, c):
    """The point of the triangle (a, b, c) that is closest to p, for arrays of shape (n, 3): returns its barycentric coordinates (n, 3) for a, b, c and its
    squared distance from p (n,). (Ericson, Real-Time Collision Detection, 5.1.5.)"""
    ab, ac, ap = b - a, c - a, p - a
    bp, cp = p - b, p - c
    d1, d2 = np.einsum("nk,nk->n", ab, ap), np.einsum("nk,nk->n", ac, ap)
    d3, d4 = np.einsum("nk,nk->n", ab, bp), np.einsum("nk,nk->n", ac, bp)
    d5, d6 = np.einsum("nk,nk->n", ab, cp), np.einsum("nk,nk->n", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    total = va + vb + vc
    safe = np.where(np.abs(total) > 1e-300, total, 1.0)
    v = np.where(np.abs(total) > 1e-300, vb / safe, 0.0)
    w = np.where(np.abs(total) > 1e-300, vc / safe, 0.0)
    u = 1.0 - v - w
    # The regions of the plane outside the triangle. Ericson tests them in the order A, B, AB, C, AC, BC and takes the first that matches: here the last
    # assignment wins, so they come in the reverse order.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        t_ac = d2 / (d2 - d6)
        t_ab = d1 / (d1 - d3)
    regions = (
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), (0.0, 1.0 - t_bc, t_bc)),    # edge BC
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1.0 - t_ac, 0.0, t_ac)),                  # edge AC
        ((d6 >= 0) & (d5 <= d6), (0.0, 0.0, 1.0)),                                     # corner C
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1.0 - t_ab, t_ab, 0.0)),                  # edge AB
        ((d3 >= 0) & (d4 <= d3), (0.0, 1.0, 0.0)),                                     # corner B
        ((d1 <= 0) & (d2 <= 0), (1.0, 0.0, 0.0)),                                      # corner A
    )
    for hit, (ru, rv, rw) in regions:
        u, v, w = np.where(hit, ru, u), np.where(hit, rv, v), np.where(hit, rw, w)
    bary = np.stack([u, v, w], axis=1)
    nearest = bary[:, 0, None] * a + bary[:, 1, None] * b + bary[:, 2, None] * c
    return bary, np.einsum("nk,nk->n", p - nearest, p - nearest)


def exposed_voxels(solid):
    """A boolean array like solid: the voxels that have a face that no other voxel shares (the ones that are seen from outside, or from an open side of the
    grid), which are the voxels of the surface."""
    padded = np.pad(solid, 1)
    enclosed = np.ones_like(solid)
    for axis in range(3):
        for shift in (1, -1):
            enclosed &= np.roll(padded, shift, axis=axis)[1:-1, 1:-1, 1:-1]
    return solid & ~enclosed


def surface_colors(tris_u, dims, solid, colors, chunk=50_000):
    """The colors of the surface voxels. A surface voxel is a voxel with a face that no other voxel shares (exposed_voxels); the color is the one at the point of
    the mesh that is closest to the center of the voxel, among the triangles that touch the voxel or one of its six face neighbors (to within COLOR_TOUCH): the mesh
    passes between the center of a surface voxel and the center of the empty neighbor, so the closest triangle is among them. One sample, no averaging.
    tris_u are the triangles in cell units, solid the (nx, ny, nz) boolean array of the voxels, colors a ColorSource for the triangles.

    Returns (cells, rgb): the (N, 3) integer indices of the cells that got a color, in the order i, j, k, and their colors (N, 3) as uint8 (0..255). Voxels that
    are below the surface, and surface voxels whose closest triangle has no color (a part of the mesh without a material, for example), are not in the result."""
    nx, ny, nz = (int(d) for d in dims)
    shape = np.array([nx, ny, nz])
    surface = np.argwhere(exposed_voxels(solid)).astype(np.int64)
    if len(surface) == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.uint8)

    # every triangle/cell pair of the grid, sorted by cell
    tri_all, flat_all = [], []
    for tri, cells in _touching(tris_u, dims, PAD, COLOR_TOUCH):
        cells = cells - PAD
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        tri_all.append(tri[inside])
        flat_all.append(((cells[inside, 0] * ny + cells[inside, 1]) * nz + cells[inside, 2]))
    if not tri_all:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.uint8)
    pair_tri, pair_flat = np.concatenate(tri_all), np.concatenate(flat_all)
    order = np.argsort(pair_flat, kind="stable")
    pair_tri, pair_flat = pair_tri[order], pair_flat[order]

    offsets = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64)
    best_cells, best_tri = [], []
    for start in range(0, len(surface), chunk):
        part = surface[start:start + chunk]
        owner, candidate = [], []
        for offset in offsets:
            cell = part + offset
            valid = np.all((cell >= 0) & (cell < shape), axis=1)
            who = np.nonzero(valid)[0]
            flat = (cell[who, 0] * ny + cell[who, 1]) * nz + cell[who, 2]
            lo = np.searchsorted(pair_flat, flat, side="left")
            count = np.searchsorted(pair_flat, flat, side="right") - lo
            total = int(count.sum())
            if total == 0:
                continue
            owner.append(np.repeat(who, count))
            within = np.arange(total) - np.repeat(np.cumsum(count) - count, count)   # 0, 1, ... inside each run of pairs
            candidate.append(pair_tri[np.repeat(lo, count) + within])
        if not owner:
            continue
        owner, candidate = np.concatenate(owner), np.concatenate(candidate)
        v = tris_u[candidate]
        _, dist = closest_barycentric(part[owner] + 0.5, v[:, 0], v[:, 1], v[:, 2])
        order = np.lexsort((dist, owner))             # by voxel, then by distance: the first of a voxel is its closest triangle
        first = np.unique(owner[order], return_index=True)[1]
        pick = order[first]
        best_cells.append(part[owner[pick]])
        best_tri.append(candidate[pick])
    if not best_cells:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.uint8)
    cells, tri = np.concatenate(best_cells), np.concatenate(best_tri)
    v = tris_u[tri]
    bary, _ = closest_barycentric(cells + 0.5, v[:, 0], v[:, 1], v[:, 2])
    rgb = colors.sample(tri, bary)
    has = ~np.isnan(rgb).any(axis=1)
    cells, rgb = cells[has], rgb[has]
    sort = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))   # the order of the voxels: i, then j, then k
    return cells[sort], np.clip(np.rint(rgb[sort] * 255.0), 0, 255).astype(np.uint8)


# ---- the outside ---------------------------------------------------------------------------------------------------

def exterior(surface):
    """The cells that can be reached from the corner of the array without passing a surface cell (6-connected flood fill, by frontiers)."""
    free = ~surface
    if not free[0, 0, 0]:
        raise ValueError("the corner of the padded grid is a surface cell; the mesh reaches outside its own bounding box")
    visited = np.zeros_like(surface)
    visited[0, 0, 0] = True
    frontier = np.zeros((1, 3), dtype=np.int64)
    shape = np.array(surface.shape)
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)
    while len(frontier):
        candidates = (frontier[:, None, :] + directions[None, :, :]).reshape(-1, 3)
        inside = np.all((candidates >= 0) & (candidates < shape), axis=1)
        candidates = candidates[inside]
        fresh = free[candidates[:, 0], candidates[:, 1], candidates[:, 2]] & ~visited[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
        candidates = candidates[fresh]
        visited[candidates[:, 0], candidates[:, 1], candidates[:, 2]] = True
        # a cell that was reached from two cells in this round is twice in the list: it only costs time, the next round sees it as visited
        frontier = np.unique(candidates, axis=0) if len(candidates) else candidates
    return visited


# ---- the center test: a ray against the mesh -------------------------------------------------------------------------

JITTER_X = 1.2345e-4  # in cells: the rays go through the cell centers shifted by this, so that they do not run along an edge of a mesh that is aligned with the grid
JITTER_Y = 7.6543e-5


def column_crossings(tris_u, nx, ny):
    """The crossings of the rays along +z through the (shifted) centers of the columns (i, j) with the triangles: (column numbers i * ny + j, z in cell units)."""
    a, b, c = tris_u[:, 0], tris_u[:, 1], tris_u[:, 2]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    lo = np.ceil(tris_u[..., :2].min(axis=1) - 0.5 - np.array([JITTER_X, JITTER_Y])).astype(np.int64)
    hi = np.floor(tris_u[..., :2].max(axis=1) - 0.5 - np.array([JITTER_X, JITTER_Y])).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array([nx, ny]) - 1)
    keep = (np.abs(area2) > 1e-12) & np.all(hi >= lo, axis=1)  # a triangle seen edge-on has no crossings
    tris_u, lo, hi, area2 = tris_u[keep], lo[keep], hi[keep], area2[keep]
    columns, heights = [], []
    for indices, (gx, gy) in _groups(lo, hi):
        offsets = np.stack(np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij"), axis=-1).reshape(-1, 2)
        cells = lo[indices][:, None, :] + offsets[None, :, :]                       # (G, C, 2)
        px = cells[..., 0] + 0.5 + JITTER_X
        py = cells[..., 1] + 0.5 + JITTER_Y
        t = tris_u[indices]                                                         # (G, 3, 3)
        ax, ay, bx, by, cx, cy = (t[:, 0, 0, None], t[:, 0, 1, None], t[:, 1, 0, None], t[:, 1, 1, None], t[:, 2, 0, None], t[:, 2, 1, None])
        s = area2[indices][:, None]
        la = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / s                    # barycentric coordinates of the ray in the projected triangle
        lb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / s
        lc = 1.0 - la - lb
        inside = (la >= 0) & (lb >= 0) & (lc >= 0)
        z = la * t[:, 0, 2, None] + lb * t[:, 1, 2, None] + lc * t[:, 2, 2, None]
        columns.append((cells[..., 0] * ny + cells[..., 1])[inside])
        heights.append(z[inside])
    if not columns:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(columns), np.concatenate(heights)


class RayParity:
    """Tells for points whether they are inside a closed mesh: the number of crossings of the ray up (+z) from the point is odd."""

    def __init__(self, tris_u, nx, ny):
        self.ny = ny
        columns, heights = column_crossings(tris_u, nx, ny)
        self.z0 = float(heights.min()) if len(heights) else 0.0
        self.span = (float(heights.max()) - self.z0 + 1.0) if len(heights) else 1.0
        keys = columns + (heights - self.z0) / self.span * 0.999999  # the column number plus the height as a fraction: sorts by column, then by height
        self.keys = np.sort(keys)

    def inside(self, i, j, k):
        """For the voxels (i, j, k) (integer arrays): is the center inside the mesh?"""
        column = i * self.ny + j
        frac = np.clip((k + 0.5 - self.z0) / self.span, 0.0, 0.999999)
        start = np.searchsorted(self.keys, column.astype(np.float64), side="left")
        end = np.searchsorted(self.keys, column + 1.0 - 1e-9, side="left")
        below = np.searchsorted(self.keys, column + frac, side="left") - start
        above = (end - start) - below
        return (above % 2) == 1


# ---- the voxels ----------------------------------------------------------------------------------------------------

def grid_for(triangles, voxel_size=None, resolution=None):
    """The origin, the voxel size and the dimensions of the grid of a mesh."""
    lo = triangles.reshape(-1, 3).min(axis=0)
    hi = triangles.reshape(-1, 3).max(axis=0)
    extent = hi - lo
    if (voxel_size is None) == (resolution is None):
        raise ValueError("give exactly one of voxel_size and resolution")
    if resolution is not None:
        if resolution < 1:
            raise ValueError("the resolution must be at least 1")
        longest = float(extent.max())
        if longest <= 0.0:
            raise ValueError("the mesh has no extent")
        voxel_size = longest / resolution
    if not voxel_size > 0.0:
        raise ValueError("the voxel size must be positive")
    dims = np.maximum(1, np.ceil(extent / voxel_size - 1e-9)).astype(np.int64)
    return lo, float(voxel_size), dims


def to_cell_units(triangles, origin, size):
    """The triangles in cell units, shifted by a thousandth of a voxel (a different amount on each axis): a mesh that is aligned with the grid (a box of whole
    voxels, say) would otherwise graze the cells next to its faces and count as touching them, and two faces a voxel apart would close the gap between them."""
    return (triangles - origin) / size + ALIGNMENT_SHIFT


def voxelize(triangles, voxel_size=None, resolution=None, surface="center", max_cells=200_000_000):
    """Returns (solid, origin, voxel_size): a boolean array of the shape dims (True where there is a voxel), the corner of the voxel (0, 0, 0) and the voxel size."""
    if surface not in ("center", "touch"):
        raise ValueError("surface must be 'center' or 'touch'")
    if len(triangles) == 0:
        raise ValueError("the mesh has no triangles")
    origin, size, dims = grid_for(triangles, voxel_size, resolution)
    cells = int(np.prod(dims + 2 * PAD))
    if cells > max_cells:
        raise ValueError("the grid has %d cells (%d x %d x %d with the margin), above the limit of %d: use a larger voxel size or a lower resolution, "
                         "or raise --max-cells" % (cells, *(dims + 2 * PAD), max_cells))
    tris_u = to_cell_units(triangles, origin, size)

    touched = surface_voxels(tris_u, dims)
    outside = exterior(touched)
    filled = ~outside                       # the surface and everything it encloses, closed cavities included
    inner = tuple(slice(PAD, PAD + int(d)) for d in dims)

    if surface == "center":
        # Only the cells that a triangle touches can have their center outside the mesh (a cell that no triangle touches is wholly in or wholly out).
        # Starting from the cells next to the outside, test the center of each (a ray through the mesh counts the crossings: odd is inside) and take
        # out the ones that are outside; they become outside themselves, so the cells behind them (a part that is thinner than a few voxels) are
        # tested in the next round. The cells next to a cavity are never tested, so cavities stay filled.
        parity = None
        checked = np.zeros_like(filled)
        while True:
            next_to_outside = np.zeros_like(filled)
            for axis in range(3):
                for shift in (1, -1):
                    next_to_outside |= np.roll(outside, shift, axis=axis)
            candidates = filled & touched & next_to_outside & ~checked
            candidates_inner = np.zeros_like(candidates)
            candidates_inner[inner] = candidates[inner]
            checked |= candidates
            ci, cj, ck = (idx - PAD for idx in np.nonzero(candidates_inner))
            if len(ci) == 0:
                break
            if parity is None:
                parity = RayParity(tris_u, int(dims[0]), int(dims[1]))
            removed = ~parity.inside(ci, cj, ck)
            if not removed.any():
                break
            gone = (ci[removed] + PAD, cj[removed] + PAD, ck[removed] + PAD)
            filled[gone] = False
            outside[gone] = True

    return filled[inner].copy(), origin, size


# ---- the joints ----------------------------------------------------------------------------------------------------

def neighbor_offsets(connectivity):
    """One of each pair of opposite neighbor offsets (the 'positive' half): 3 for connectivity 6, 9 for 18, 13 for 26."""
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    limit = {6: 1, 18: 2, 26: 3}[connectivity]
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nonzero = abs(dx) + abs(dy) + abs(dz)
                if nonzero == 0 or nonzero > limit:
                    continue
                if (dx, dy, dz) > (0, 0, 0):  # the lexicographically positive one of the two opposite offsets
                    offsets.append((dx, dy, dz))
    return offsets


def build_joints(solid, connectivity=6):
    """Returns (voxels, joints): the (N, 3) integer indices of the solid cells in the order i, j, k, and the (M, 2) numbers of the pairs of neighbors (each pair once)."""
    voxels = np.argwhere(solid).astype(np.int64)
    number = np.full(solid.shape, -1, dtype=np.int64)
    number[solid] = np.arange(len(voxels))  # the order of argwhere is the order of a C-ordered boolean mask
    pairs = []
    nx, ny, nz = solid.shape
    for dx, dy, dz in neighbor_offsets(connectivity):
        # cells (i, j, k) and (i + dx, j + dy, k + dz) that are both in the grid
        a = number[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy), max(0, -dz):nz - max(0, dz)]
        b = number[max(0, dx):nx - max(0, -dx), max(0, dy):ny - max(0, -dy), max(0, dz):nz - max(0, -dz)]
        both = (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[both], b[both]], axis=1))
    joints = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    return voxels, joints


# ---- writing -------------------------------------------------------------------------------------------------------

def _flat(array):
    return "[" + ",".join(map(str, array.reshape(-1).tolist())) + "]"


def color_voxel_numbers(voxels, dims, cells):
    """The voxel numbers (positions in the voxel list) of the cells that have a color; every one of the cells must be a voxel."""
    ny, nz = int(dims[1]), int(dims[2])
    flat = (voxels[:, 0] * ny + voxels[:, 1]) * nz + voxels[:, 2]    # ascending: the voxels are ordered i, j, k
    wanted = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
    numbers = np.searchsorted(flat, wanted)
    if len(wanted) and (numbers.max() >= len(flat) or not np.array_equal(flat[numbers], wanted)):
        raise ValueError("a color belongs to a cell that is not a voxel")
    return numbers


def write_json(path, source, origin, voxel_size, dims, voxels, joints, connectivity, surface, colors=None):
    """colors is None, or (numbers, rgb): the voxel numbers of the surface voxels that have a color (ascending) and their colors (N, 3) as 0..255."""
    header = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source": os.path.basename(source),
        "voxelSize": voxel_size,
        "origin": [float(x) for x in origin],
        "gridDimensions": [int(x) for x in dims],
        "connectivity": connectivity,
        "surface": surface,
        "voxelCount": int(len(voxels)),
        "jointCount": int(len(joints)),
    }
    if colors is not None and len(colors[0]) > 0:
        header["colorCount"] = int(len(colors[0]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key, value in header.items():
            f.write("  %s: %s,\n" % (json.dumps(key), json.dumps(value)))
        f.write('  "voxels": ' + _flat(voxels) + ",\n")
        if colors is not None and len(colors[0]) > 0:
            f.write('  "joints": ' + _flat(joints) + ",\n")
            f.write('  "colorVoxels": ' + _flat(colors[0]) + ",\n")
            f.write('  "colors": ' + _flat(colors[1]) + "\n")
        else:
            f.write('  "joints": ' + _flat(joints) + "\n")
        f.write("}\n")


# ---- command line --------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 1)[1])
    p.add_argument("mesh", help="the mesh file: an STL (binary or ASCII) or a Wavefront OBJ (.obj)")
    p.add_argument("-o", "--output", help="the JSON file to write (default: the mesh file's name with .voxels.json)")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--voxel-size", type=float, help="edge length of a voxel, in the output units")
    size.add_argument("--resolution", type=int, help="number of voxels along the longest side of the mesh")
    p.add_argument("--surface", choices=("center", "touch"), default="center", help="solid voxels: center inside the mesh (default), or touching or inside it")
    p.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=6, help="neighbors that are joined: 6 (faces, default), 18 (+ edges), 26 (+ corners)")
    p.add_argument("--scale", type=float, default=1.0, help="factor applied to the mesh first (for example 0.001 for a mesh in millimeters, to get meters)")
    p.add_argument("--zup-to-yup", action="store_true", help="rotate the mesh from a z-up tool's frame into Unity's y-up frame: (x, y, z) -> (x, z, -y)")
    p.add_argument("--no-colors", action="store_true", help="do not write the colors of the surface voxels, even if the mesh file has colors")
    p.add_argument("--max-cells", type=int, default=200_000_000, help="refuse a grid with more cells than this (with a margin), default 200 million")
    args = p.parse_args(argv)

    output = args.output or os.path.splitext(args.mesh)[0] + ".voxels.json"
    try:
        mesh, color_source = read_mesh_colored(args.mesh)
        triangles = transform(mesh, args.scale, args.zup_to_yup)
        if not np.all(np.isfinite(triangles)):
            raise ValueError("the mesh has vertices that are not finite numbers")
        open_edges = open_edge_count(triangles)
        solid, origin, voxel_size = voxelize(triangles, args.voxel_size, args.resolution, args.surface, args.max_cells)
        voxels, joints = build_joints(solid, args.connectivity)
        colors = None
        if color_source is not None and not args.no_colors and color_source.has_colors():
            cells, rgb = surface_colors(to_cell_units(triangles, origin, voxel_size), solid.shape, solid, color_source)
            colors = (color_voxel_numbers(voxels, solid.shape, cells), rgb)
    except (ValueError, OSError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1

    if open_edges:
        print("warning: the mesh is not closed (%d edges do not belong to exactly two triangles): the outside can leak into the inside through a hole "
              "and leave the model hollow" % open_edges, file=sys.stderr)
    if len(voxels) == 0:
        print("warning: no voxel is solid: the voxel size is too large for this mesh (try a smaller one, or --surface touch)", file=sys.stderr)

    if color_source is not None:
        for warning in color_source.warnings:
            print("warning: %s" % warning, file=sys.stderr)
    write_json(output, args.mesh, origin, voxel_size, solid.shape, voxels, joints, args.connectivity, args.surface, colors)

    mesh_vol = mesh_volume(triangles)
    voxel_vol = len(voxels) * voxel_size ** 3
    print("%s: %d triangles, grid %d x %d x %d, voxel size %.6g" % (os.path.basename(args.mesh), len(triangles), *solid.shape, voxel_size))
    print("%d voxels, %d joints (%d-neighborhood); voxel volume %.6g, mesh volume %.6g%s" % (
        len(voxels), len(joints), args.connectivity, voxel_vol, mesh_vol, " (%.1f%%)" % (100.0 * voxel_vol / mesh_vol) if mesh_vol > 0 else ""))
    if colors is not None:
        print("colors: %d surface voxels have a color (the other voxels have none)" % len(colors[0]))
    elif color_source is not None and not args.no_colors and not color_source.has_colors():
        print("colors: the mesh file has none")
    print("written to %s" % output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
