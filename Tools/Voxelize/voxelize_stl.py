#!/usr/bin/env python3
"""Turns an STL mesh into a solid set of voxels and writes them, with the joints between neighboring voxels, as JSON for Unity and the AVBD package.

    python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1
    python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --resolution 64 --zup-to-yup

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

With --smooth the script also bakes the smooth surface of the mesh into the blocks (see smooth_chunks.py): for every voxel on the surface, the piece of the mesh
that belongs to it, closed into a solid of its own with flat faces where it touches its neighbors. The demo draws these chunks instead of the boxes, so an object looks
like the mesh while it is whole, and a block that breaks off is still a block (with the curved outside of the mesh). It makes the file much larger.

Output (JSON, see README.md for the format): the grid (origin, voxel size, dimensions), the voxel indices as one flat list of (i, j, k) triples, and the
joints as one flat list of pairs of voxel numbers (positions in the voxel list). The center of voxel (i, j, k) is origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize.

Needs Python 3 and numpy. Reads binary and ASCII STL.
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
FORMAT_VERSION = 1   # the chunk arrays of --smooth are optional additions: a reader that does not know them reads the file as before

PAD = 2          # empty cells around the grid for the flood fill
ALIGNMENT_SHIFT = np.array([1.0e-3, 7.07e-4, 5.77e-4])  # in voxels, see voxelize()
TOUCH_EPS = 1e-6  # in voxels: a triangle that only grazes a voxel counts as touching it
POSITION_DECIMALS = 5  # decimals of the positions of a chunk's vertices, in the units of the output
NORMAL_DECIMALS = 3
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


def surface_voxels(tris_u, dims, pad=PAD):
    """The cells that the triangles touch. tris_u are the triangles in cell units (a cell c is [c, c + 1]); the result is a boolean array of the
    shape dims + 2 * pad, in which cell c is at index c + pad. Cells farther than pad outside the grid are not kept."""
    shape = tuple(int(d) + 2 * pad for d in dims)
    surface = np.zeros(shape, dtype=bool)
    lo = np.floor(tris_u.min(axis=1) - TOUCH_EPS).astype(np.int64)
    hi = np.floor(tris_u.max(axis=1) + TOUCH_EPS).astype(np.int64)
    lo = np.maximum(lo, -pad)
    hi = np.minimum(hi, np.array(dims) + pad - 1)
    keep = np.all(hi >= lo, axis=1)
    tris_u, lo, hi = tris_u[keep], lo[keep], hi[keep]
    for indices, (nx, ny, nz) in _groups(lo, hi):
        offsets = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1).reshape(-1, 3)
        cells = lo[indices][:, None, :] + offsets[None, :, :]                  # (G, C, 3)
        hits = _overlap(tris_u[indices], cells + 0.5, 0.5 + TOUCH_EPS)
        touched = cells[hits] + pad
        surface[touched[:, 0], touched[:, 1], touched[:, 2]] = True
    return surface


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
    # In cell units, shifted by a thousandth of a voxel (a different amount on each axis): a mesh that is aligned with the grid (a box of whole voxels, say)
    # would otherwise graze the cells next to its faces and count as touching them, and two faces a voxel apart would close the gap between them.
    tris_u = (triangles - origin) / size + ALIGNMENT_SHIFT

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


def _flat_floats(array, decimals):
    """A flat JSON list of floats with at most `decimals` decimals (trailing zeros left out)."""
    text = []
    for x in np.round(array.reshape(-1), decimals).tolist():
        t = ("%.*f" % (decimals, x)).rstrip("0").rstrip(".")
        text.append("0" if t in ("", "-0") else t)
    return "[" + ",".join(text) + "]"


def write_json(path, source, origin, voxel_size, dims, voxels, joints, connectivity, surface, chunks=None):
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        for key, value in header.items():
            f.write("  %s: %s,\n" % (json.dumps(key), json.dumps(value)))
        f.write('  "voxels": ' + _flat(voxels) + ",\n")
        f.write('  "joints": ' + _flat(joints))
        if chunks is not None and chunks.count > 0:
            # Optional: the baked chunks. Chunk i belongs to voxel chunkVoxels[i]; its vertices are chunkPositions / chunkNormals [3 * chunkVertexStart[i] ...
            # 3 * chunkVertexStart[i + 1]) and its triangles chunkTriangles [3 * chunkTriangleStart[i] ... 3 * chunkTriangleStart[i + 1]), indices counted
            # from the first vertex of the chunk. Positions are relative to the center of the voxel, in the units of the output.
            f.write(",\n")
            f.write('  "chunkCount": %d,\n' % chunks.count)
            f.write('  "chunkVoxels": ' + _flat(np.array(chunks.voxel_numbers)) + ",\n")
            f.write('  "chunkVertexStart": ' + _flat(np.array(chunks.vertex_start)) + ",\n")
            f.write('  "chunkTriangleStart": ' + _flat(np.array(chunks.triangle_start)) + ",\n")
            f.write('  "chunkPositions": ' + _flat_floats(chunks.positions, POSITION_DECIMALS) + ",\n")
            f.write('  "chunkNormals": ' + _flat_floats(chunks.normals, NORMAL_DECIMALS) + ",\n")
            f.write('  "chunkTriangles": ' + _flat(chunks.triangles))
        f.write("\n}\n")


# ---- command line --------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n", 1)[1])
    p.add_argument("stl", help="the STL file (binary or ASCII)")
    p.add_argument("-o", "--output", help="the JSON file to write (default: the STL's name with .voxels.json)")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--voxel-size", type=float, help="edge length of a voxel, in the output units")
    size.add_argument("--resolution", type=int, help="number of voxels along the longest side of the mesh")
    p.add_argument("--surface", choices=("center", "touch"), default="center", help="solid voxels: center inside the mesh (default), or touching or inside it")
    p.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=6, help="neighbors that are joined: 6 (faces, default), 18 (+ edges), 26 (+ corners)")
    p.add_argument("--scale", type=float, default=1.0, help="factor applied to the mesh first (for example 0.001 for an STL in millimeters, to get meters)")
    p.add_argument("--zup-to-yup", action="store_true", help="rotate the mesh from a z-up tool's frame into Unity's y-up frame: (x, y, z) -> (x, z, -y)")
    p.add_argument("--smooth", action="store_true", help="also bake the smooth surface of the mesh into the blocks on the surface (a chunk each, see smooth_chunks.py); larger files")
    p.add_argument("--crease", type=float, default=50.0, help="with --smooth: the angle in degrees up to which the normals of neighboring triangles are smoothed (default 50)")
    p.add_argument("--max-cells", type=int, default=200_000_000, help="refuse a grid with more cells than this (with a margin), default 200 million")
    args = p.parse_args(argv)

    output = args.output or os.path.splitext(args.stl)[0] + ".voxels.json"
    try:
        triangles = transform(read_stl(args.stl), args.scale, args.zup_to_yup)
        if not np.all(np.isfinite(triangles)):
            raise ValueError("the mesh has vertices that are not finite numbers")
        open_edges = open_edge_count(triangles)
        solid, origin, voxel_size = voxelize(triangles, args.voxel_size, args.resolution, args.surface, args.max_cells)
        voxels, joints = build_joints(solid, args.connectivity)
        chunks = None
        if args.smooth and len(voxels):
            import smooth_chunks
            chunks = smooth_chunks.bake(triangles, origin, voxel_size, solid, voxels, args.crease)
    except (ValueError, OSError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1

    if open_edges:
        print("warning: the mesh is not closed (%d edges do not belong to exactly two triangles): the outside can leak into the inside through a hole "
              "and leave the model hollow" % open_edges, file=sys.stderr)
    if len(voxels) == 0:
        print("warning: no voxel is solid: the voxel size is too large for this mesh (try a smaller one, or --surface touch)", file=sys.stderr)

    write_json(output, args.stl, origin, voxel_size, solid.shape, voxels, joints, args.connectivity, args.surface, chunks)

    mesh_vol = mesh_volume(triangles)
    voxel_vol = len(voxels) * voxel_size ** 3
    print("%s: %d triangles, grid %d x %d x %d, voxel size %.6g" % (os.path.basename(args.stl), len(triangles), *solid.shape, voxel_size))
    print("%d voxels, %d joints (%d-neighborhood); voxel volume %.6g, mesh volume %.6g%s" % (
        len(voxels), len(joints), args.connectivity, voxel_vol, mesh_vol, " (%.1f%%)" % (100.0 * voxel_vol / mesh_vol) if mesh_vol > 0 else ""))
    if chunks is not None:
        st = chunks.stats
        print("smooth surface: %d chunks (the other %d voxels are plain boxes), %d vertices, %d triangles (%d of them caps, %.1f per chunk)" % (
            st["chunks"], len(voxels) - st["chunks"], st["vertices"], st["triangles"], st["cap_triangles"], st["triangles"] / max(st["chunks"], 1)))
        if st["outside_cells_given_to_a_voxel"]:
            print("  %d cells that are not voxels but have mesh in them were given to the nearest voxel" % st["outside_cells_given_to_a_voxel"])
        if st["pieces_lost"]:
            print("warning: the mesh has %d pieces in cells with no voxel around them (a part thinner than a voxel); they are not drawn" % st["pieces_lost"], file=sys.stderr)
        if st["cavity_triangles_left_out"]:
            print("  %d triangles of closed cavities were left out (the object is solid)" % st["cavity_triangles_left_out"])
    print("written to %s" % output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
