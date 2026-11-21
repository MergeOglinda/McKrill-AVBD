"""Bakes a smooth surface into the blocks of a voxelized mesh: for every voxel on the surface, the piece of the original mesh that belongs to it (a "chunk").

voxelize_stl.py --smooth uses this. The union of the chunks of an intact object is the mesh itself (smooth, not stair-stepped), and every chunk is a closed
solid of its own, so a block that breaks off the object is still a block: it has the curved outside of the mesh and flat faces where it was cut from its
neighbors. A voxel that has no mesh surface in its cell (the inside of the object) has no chunk: it is a plain box.

How a chunk is made (all in cell units, a cell c is [c, c + 1] along each axis, so the cuts are exact):

 1. The triangles are turned so that they face outward (the ray test of the voxelizer, on a point just in front of each triangle), the triangles of closed
    cavities are left out (the object is solid, as the voxels are), and smooth vertex normals are computed (angles of up to --crease degrees are smoothed).
 2. Every triangle is clipped against the boxes of the cells it overlaps (Sutherland-Hodgman, normals interpolated). The pieces in a cell that is a voxel
    belong to that voxel. The pieces in a cell that is not a voxel (the mesh passes through a cell whose center is outside) belong to the nearest voxel of the 26
    cells around it, so that no part of the surface is lost; such a chunk reaches out of its own cell by up to one cell.
 3. Every face of a cell of a chunk that borders something solid that is not part of the same chunk gets a cap: the cross-section of the mesh with the face, inside
    the face's square (a slice just inside the cell, cut into trapezoids and triangulated; the inside of the cross-section is found with a ray test, not
    assumed). The caps are what is seen when the neighbor is gone. A face that borders a cell without any mesh solid gets no cap (the mesh itself is there).
 4. The vertices are written relative to the center of the voxel, in the units of the output, so that the chunk can be drawn with the pose of the body.

The mesh is nudged by at most a thousandth of a cell (GUARD) so that no triangle lies exactly in a cut plane (a box of whole voxels has its faces on the
planes); that is far below anything visible. Needs numpy.
"""
import math
from collections import defaultdict

import numpy as np

GUARD = 3.0e-4                                       # cells: the mesh is shrunk about the center of the grid by this much on each side
GUARD_OFFSET = np.array([1.1e-5, 7.3e-6, 5.1e-6])    # cells: and moved by this much (a different amount on each axis), so that no face is exactly on a plane
CAP_SLICE = 1.0e-4                                   # cells: the cross-section for a cap is taken this far inside the cell
MIN_AREA = 1.0e-12                                   # cells^2: smaller polygons are dropped
ORIENT_STEP = 1.0e-3                                 # cells: how far in front of a triangle the ray test looks to find out whether it faces outward
QUERY_JITTER = np.array([3.1e-7, 1.7e-7])            # cells: the rays of the inside test are shifted by this much, so that they do not run along an edge
WELD_POSITION = 1.0e5                                # vertices of a chunk closer than 1e-5 (in the units of the output) and with the same normal are one
WELD_NORMAL = 1.0e3

NEIGHBORS_26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
FACES = [(axis, side) for axis in range(3) for side in (-1, 1)]  # the six faces of a cell: the axis and the direction of the outward normal


# ---- the inside test ------------------------------------------------------------------------------------------------

class PointParity:
    """Tells for points whether they are inside a closed mesh: the number of crossings of the ray up (+z) from the point is odd. Like RayParity of
    voxelize_stl.py, but for any point; the triangles are binned by their x, y box (one bin per cell column)."""

    def __init__(self, tris):
        self.tris = tris
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        self.area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        lo = np.floor(tris[..., :2].min(axis=1)).astype(np.int64)
        hi = np.floor(tris[..., :2].max(axis=1)).astype(np.int64)
        bins = defaultdict(list)
        for t in np.nonzero(np.abs(self.area2) > 1e-14)[0]:  # a triangle seen edge-on has no crossings
            for bx in range(lo[t, 0], hi[t, 0] + 1):
                for by in range(lo[t, 1], hi[t, 1] + 1):
                    bins[(bx, by)].append(t)
        self.bins = {key: np.array(value, dtype=np.int64) for key, value in bins.items()}

    def inside(self, point):
        x = point[0] + QUERY_JITTER[0]
        y = point[1] + QUERY_JITTER[1]
        candidates = self.bins.get((math.floor(x), math.floor(y)))
        if candidates is None:
            return False
        t = self.tris[candidates]
        s = self.area2[candidates]
        ax, ay, az = t[:, 0, 0], t[:, 0, 1], t[:, 0, 2]
        bx, by, bz = t[:, 1, 0], t[:, 1, 1], t[:, 1, 2]
        cx, cy, cz = t[:, 2, 0], t[:, 2, 1], t[:, 2, 2]
        la = ((bx - x) * (cy - y) - (by - y) * (cx - x)) / s   # barycentric coordinates of the ray in the projected triangle
        lb = ((cx - x) * (ay - y) - (cy - y) * (ax - x)) / s
        lc = 1.0 - la - lb
        hit = (la >= 0) & (lb >= 0) & (lc >= 0)
        z = la * az + lb * bz + lc * cz
        return int(np.count_nonzero(hit & (z > point[2]))) % 2 == 1


# ---- the mesh in cell units ----------------------------------------------------------------------------------------------

def to_cell_units(triangles, origin, size, dims):
    """The triangles in cell units (a cell c is [c, c + 1]), shrunk and nudged by GUARD so that no face lies exactly in a plane between cells."""
    u = (np.asarray(triangles, dtype=np.float64) - origin) / size
    n = np.asarray(dims, dtype=np.float64)
    center = n / 2.0
    return center + (u - center) * (1.0 - 2.0 * GUARD / n) + GUARD_OFFSET


def orient_outward(tris):
    """Returns the triangles with the winding turned where it faces into the mesh (a point just in front of the triangle is inside), and drops the
    triangles that have no area. The winding that faces outward is the one with the counterclockwise order seen from outside (the right hand rule)."""
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normal, axis=1)
    keep = length > 1e-12
    tris, normal, length = tris[keep], normal[keep], length[keep]
    unit = normal / length[:, None]
    parity = PointParity(tris)
    probes = tris.mean(axis=1) + unit * ORIENT_STEP
    flip = np.array([parity.inside(p) for p in probes], dtype=bool)
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def corner_normals(tris, crease_degrees):
    """Smooth normals for the three corners of every triangle (shape (T, 3, 3)): the angle-weighted mean of the normals of the triangles that share the
    corner (vertices that coincide) and whose normals are within crease_degrees of this triangle's, so that a sharp edge stays sharp."""
    count = len(tris)
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face /= np.linalg.norm(face, axis=1)[:, None]
    angles = np.zeros((count, 3))
    for i in range(3):
        u = tris[:, (i + 1) % 3] - tris[:, i]
        v = tris[:, (i + 2) % 3] - tris[:, i]
        cosine = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        angles[:, i] = np.arccos(np.clip(cosine, -1.0, 1.0))
    keys = np.round(tris.reshape(-1, 3) / 1e-6).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    order = np.argsort(ids, kind="stable")
    boundaries = np.concatenate([[0], np.nonzero(np.diff(ids[order]))[0] + 1, [len(ids)]])
    result = np.zeros((count * 3, 3))
    cosine_limit = math.cos(math.radians(crease_degrees))
    flat_angles = angles.reshape(-1)
    for g in range(len(boundaries) - 1):
        corners = order[boundaries[g]:boundaries[g + 1]]
        normals = face[corners // 3]
        weights = flat_angles[corners]
        close = (normals @ normals.T) >= cosine_limit
        summed = (close * weights[None, :]) @ normals
        length = np.linalg.norm(summed, axis=1)
        length[length < 1e-12] = 1.0
        result[corners] = summed / length[:, None]
    return result.reshape(count, 3, 3)


# ---- clipping ------------------------------------------------------------------------------------------------------

def _clip_axis(polygon, axis, value, keep_above):
    """Sutherland-Hodgman against one plane. A vertex is (x, y, z, nx, ny, nz); the vertices on the plane are kept."""
    out = []
    count = len(polygon)
    for i in range(count):
        a = polygon[i]
        b = polygon[(i + 1) % count]
        da = (a[axis] - value) if keep_above else (value - a[axis])
        db = (b[axis] - value) if keep_above else (value - b[axis])
        if da >= 0.0:
            out.append(a)
        if (da >= 0.0) != (db >= 0.0):
            t = da / (da - db)
            out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(6)))
    return out


def clip_to_cell(polygon, cell):
    """The part of a convex polygon inside the box of a cell, or None if there is none (or it has no area)."""
    for axis in range(3):
        polygon = _clip_axis(polygon, axis, float(cell[axis]), True)
        if len(polygon) < 3:
            return None
        polygon = _clip_axis(polygon, axis, float(cell[axis] + 1), False)
        if len(polygon) < 3:
            return None
    return polygon if polygon_area(polygon) > MIN_AREA else None


def polygon_area(polygon):
    p0 = np.array(polygon[0][:3])
    total = np.zeros(3)
    for i in range(1, len(polygon) - 1):
        total += np.cross(np.array(polygon[i][:3]) - p0, np.array(polygon[i + 1][:3]) - p0)
    return 0.5 * float(np.linalg.norm(total))


# ---- the caps ------------------------------------------------------------------------------------------------------

def slice_segments(polygons, axis, plane):
    """The segments in which the convex polygons cross the plane (coordinate `axis` equal to `plane`), as pairs of points in the other two axes
    (u, v) = ((axis + 1) % 3, (axis + 2) % 3)."""
    u, v = (axis + 1) % 3, (axis + 2) % 3
    segments = []
    for polygon in polygons:
        points = []
        count = len(polygon)
        for i in range(count):
            a = polygon[i]
            b = polygon[(i + 1) % count]
            da = a[axis] - plane
            db = b[axis] - plane
            if (da >= 0.0) != (db >= 0.0):
                t = da / (da - db)
                points.append((a[u] + (b[u] - a[u]) * t, a[v] + (b[v] - a[v]) * t))
        if len(points) == 2 and (abs(points[0][0] - points[1][0]) > 1e-12 or abs(points[0][1] - points[1][1]) > 1e-12):
            segments.append((points[0], points[1]))
    return segments


def _chain(segments):
    """The points of the polyline that the segments form, if they form exactly one open polyline (every point but the two ends is in two segments); else None."""
    nodes = []

    def node(p):
        for i, q in enumerate(nodes):
            if abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9:
                return i
        nodes.append(p)
        return len(nodes) - 1

    adjacent = defaultdict(list)
    for a, b in segments:
        ia, ib = node(a), node(b)
        if ia != ib:
            adjacent[ia].append(ib)
            adjacent[ib].append(ia)
    ends = [i for i in adjacent if len(adjacent[i]) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in adjacent.values()):
        return None
    chain, previous, current = [ends[0]], None, ends[0]
    while True:
        following = [n for n in adjacent[current] if n != previous]
        if not following or following[0] in chain:
            break
        previous, current = current, following[0]
        chain.append(current)
    return [nodes[i] for i in chain] if len(chain) == len(adjacent) else None


def _perimeter(p, u0, u1, v0, v1):
    """The place of a point on the border of the square as a number in [0, 4) going counterclockwise from the corner (u0, v0) (corners are the whole numbers),
    or None if the point is not on the border."""
    eps = 1e-7
    du, dv = u1 - u0, v1 - v0
    if abs(p[1] - v0) < eps:
        return (p[0] - u0) / du % 4.0
    if abs(p[0] - u1) < eps:
        return 1.0 + (p[1] - v0) / dv
    if abs(p[1] - v1) < eps:
        return 2.0 + (u1 - p[0]) / du
    if abs(p[0] - u0) < eps:
        return (3.0 + (v1 - p[1]) / dv) % 4.0
    return None


def triangulate_polygon(points):
    """Ear clipping of a simple counterclockwise polygon in (u, v): a list of triples of point numbers, or None if it fails (a polygon that is not simple)."""
    count = len(points)
    if count < 3:
        return None
    index = list(range(count))
    triangles = []

    def cross(a, b, c):
        return (points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) - (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])

    def strictly_inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        return d1 > 1e-14 and d2 > 1e-14 and d3 > 1e-14

    while len(index) > 3:
        for i in range(len(index)):
            a, b, c = index[i - 1], index[i], index[(i + 1) % len(index)]
            if cross(a, b, c) <= 1e-14:
                continue   # a reflex corner, or a straight or doubled one
            if any(strictly_inside(j, a, b, c) for j in index if j not in (a, b, c)):
                continue
            triangles.append((a, b, c))
            del index[i]
            break
        else:
            # no ear: straight or doubled points keep the polygon from shrinking: take them out and go on, or give up
            straight = [i for i in range(len(index)) if abs(cross(index[i - 1], index[i], index[(i + 1) % len(index)])) <= 1e-14]
            if not straight:
                return None
            del index[straight[0]]
    if cross(*index) > 1e-14:
        triangles.append(tuple(index))
    return triangles


def _cap_from_one_chain(chain, u0, u1, v0, v1, is_inside):
    """The cap when the cross-section within the square is one polyline from a point of the border to another: the polyline cuts the square into two parts, of
    which one is inside. Returns triangles in (u, v), or None if the case is not this one."""
    ta = _perimeter(chain[0], u0, u1, v0, v1)
    tb = _perimeter(chain[-1], u0, u1, v0, v1)
    if ta is None or tb is None or abs(ta - tb) < 1e-9 or abs(abs(ta - tb) - 4.0) < 1e-9:
        return None
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    parts = []
    for first, ta_, tb_ in ((chain, ta, tb), (chain[::-1], tb, ta)):
        # this part is bounded by the polyline (from the point at ta_ to the point at tb_) and the border from tb_ back to ta_, counterclockwise
        polygon = list(first)
        ordered = sorted([k for k in range(4) if 1e-9 < (k - tb_) % 4.0 < (ta_ - tb_) % 4.0 - 1e-9], key=lambda k: (k - tb_) % 4.0)
        polygon += [corners[k] for k in ordered]
        twice_area = sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))
        if twice_area < 0.0:
            return None   # the polyline folds back on itself: not a simple cut
        triangles = triangulate_polygon(polygon)
        if triangles is None:
            return None
        parts.append((polygon, triangles))
    # the two parts together are the square (a check of the cut)
    total = 0.0
    for polygon, triangles in parts:
        for a, b, c in triangles:
            total += 0.5 * ((polygon[b][0] - polygon[a][0]) * (polygon[c][1] - polygon[a][1]) - (polygon[b][1] - polygon[a][1]) * (polygon[c][0] - polygon[a][0]))
    if abs(total - (u1 - u0) * (v1 - v0)) > 1e-6 * (u1 - u0) * (v1 - v0):
        return None
    # which part is inside: look at the middle of the largest triangle of the first one, which is far from the cut
    polygon, triangles = parts[0]

    def triangle_area(t):
        a, b, c = (polygon[i] for i in t)
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    best = max(triangles, key=triangle_area)
    middle = tuple(sum(polygon[i][k] for i in best) / 3.0 for k in range(2))
    polygon, triangles = parts[0] if is_inside(middle[0], middle[1]) else parts[1]
    return [tuple(polygon[i] for i in t) for t in triangles]


def cap_polygons(segments, u0, u1, v0, v1, is_inside):
    """The inside of a cross-section within the square [u0, u1] x [v0, v1], as triangles in (u, v), counterclockwise. The cross-section is given by
    its boundary segments (the ones inside the square); is_inside(u, v) tells whether a point of the square is inside it.

    The usual case, one polyline from one point of the border of the square to another, is cut into two polygons that are triangulated (ear clipping). Any
    other case: the square is cut into slabs at the u of every segment end; in a slab the segments that span it do not cross, so the space between two of
    them alternates inside, outside."""
    if segments:
        chain = _chain(segments)
        if chain is not None:
            done = _cap_from_one_chain(chain, u0, u1, v0, v1, is_inside)
            if done is not None:
                return done
    cuts = sorted({u0, u1} | {p[0] for s in segments for p in s if u0 < p[0] < u1})
    lines = []  # (u of the start, v there, u of the end, v there), the start has the smaller u
    for (ua, va), (ub, vb) in segments:
        if ua > ub:
            ua, va, ub, vb = ub, vb, ua, va
        if ub - ua > 1e-12:
            lines.append((ua, va, ub, vb))
    triangles = []
    for k in range(len(cuts) - 1):
        left, right = cuts[k], cuts[k + 1]
        if right - left < 1e-9:
            continue
        middle = 0.5 * (left + right)
        spanning = []
        for ua, va, ub, vb in lines:
            if ua <= left + 1e-9 and ub >= right - 1e-9:
                slope = (vb - va) / (ub - ua)
                at_left = va + slope * (left - ua)
                at_right = va + slope * (right - ua)
                spanning.append((0.5 * (at_left + at_right), min(max(at_left, v0), v1), min(max(at_right, v0), v1)))
        spanning.sort()
        borders = [(v0, v0)] + [(a, b) for _, a, b in spanning] + [(v1, v1)]   # the border lines, the edges of the square first and last
        middles = [0.5 * (a + b) for a, b in borders]
        # The space between two borders alternates between inside and outside. Look at the thickest space to know which it is: the point is far from any surface.
        thickest = max(range(len(borders) - 1), key=lambda i: middles[i + 1] - middles[i])
        if middles[thickest + 1] - middles[thickest] < 1e-9:
            continue
        thickest_is_inside = is_inside(middle, 0.5 * (middles[thickest] + middles[thickest + 1]))
        for i in range(len(borders) - 1):
            inside = thickest_is_inside if (i - thickest) % 2 == 0 else not thickest_is_inside
            if not inside:
                continue
            (la, lb), (ha, hb) = borders[i], borders[i + 1]
            quad = [(left, la), (right, lb), (right, hb), (left, ha)]
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                twice_area = (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) - (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0])
                if twice_area > 1e-12:
                    triangles.append(tri)
    return triangles


# ---- the baking ------------------------------------------------------------------------------------------------------

class ChunkSet:
    """The chunks of a voxel object: for chunk i, voxel_numbers[i] is the number of its voxel (its place in the list of voxels), its vertices are
    positions[vertex_start[i]:vertex_start[i + 1]] (relative to the center of the voxel, in the units of the output) with the same range of normals, and
    its triangles are triangles[triangle_start[i]:triangle_start[i + 1]], indices into the vertices of the chunk (counted from its first one)."""

    def __init__(self):
        self.voxel_numbers = []
        self.vertex_start = [0]
        self.triangle_start = [0]
        self.positions = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.triangles = np.zeros((0, 3), dtype=np.int64)
        self.stats = {}

    @property
    def count(self):
        return len(self.voxel_numbers)


def bake(triangles, origin, size, solid, voxels, crease_degrees=50.0):
    """Bakes the chunks of the voxels that are on the surface of the mesh.

    triangles: the mesh (T, 3, 3) in the units of the output; origin, size: the grid; solid: the boolean array of the voxels; voxels: (N, 3) the indices of
    the voxels in the order of the output (voxel n is voxels[n]). Returns a ChunkSet."""
    dims = solid.shape
    solid_cells = {tuple(int(x) for x in v): n for n, v in enumerate(voxels)}
    tris = orient_outward(to_cell_units(triangles, origin, size, dims))
    normals = corner_normals(tris, crease_degrees)

    # which cells are worth a look: the voxels and the cells around them (the surface can pass through a cell whose center is outside)
    relevant = set(solid_cells)
    for cell in solid_cells:
        for d in NEIGHBORS_26:
            relevant.add((cell[0] + d[0], cell[1] + d[1], cell[2] + d[2]))

    # 1. clip every triangle against the cells it overlaps
    pieces = defaultdict(list)   # cell -> [(triangle number, polygon)]
    lows = np.floor(tris.min(axis=1)).astype(np.int64)
    highs = np.floor(tris.max(axis=1)).astype(np.int64)
    for t in range(len(tris)):
        base = [tuple(tris[t, i]) + tuple(normals[t, i]) for i in range(3)]
        for cx in range(lows[t, 0], highs[t, 0] + 1):
            for cy in range(lows[t, 1], highs[t, 1] + 1):
                for cz in range(lows[t, 2], highs[t, 2] + 1):
                    cell = (cx, cy, cz)
                    if cell not in relevant:
                        continue
                    polygon = clip_to_cell(base, cell)
                    if polygon is not None:
                        pieces[cell].append((t, polygon))

    # 2. leave out the triangles of closed cavities: a triangle is part of the outside if one of its pieces is in a cell that is not a voxel or that has
    #    an empty cell next to it (also across an edge or a corner). The wall of a cavity only touches cells that are surrounded by voxels.
    def on_the_outside(cell):
        if cell not in solid_cells:
            return True
        return any((cell[0] + d[0], cell[1] + d[1], cell[2] + d[2]) not in solid_cells for d in NEIGHBORS_26)

    outer = set()
    for cell, cell_pieces in pieces.items():
        if on_the_outside(cell):
            outer.update(t for t, _ in cell_pieces)
    pieces = {cell: [p for p in cell_pieces if p[0] in outer] for cell, cell_pieces in pieces.items()}
    pieces = {cell: cell_pieces for cell, cell_pieces in pieces.items() if cell_pieces}
    outer_tris = tris[sorted(outer)] if outer else np.zeros((0, 3, 3))

    # 3. who owns what: a cell that is a voxel is its own chunk's; a cell with pieces that is not a voxel goes to the nearest voxel around it
    owner = {}                   # cell -> the voxel cell that owns its pieces
    dropped = 0
    reassigned = 0
    for cell in pieces:
        if cell in solid_cells:
            owner[cell] = cell
            continue
        centroid = np.zeros(3)
        weight = 0.0
        for _, polygon in pieces[cell]:
            area = polygon_area(polygon)
            centroid += area * np.mean([v[:3] for v in polygon], axis=0)
            weight += area
        centroid /= weight
        best = None
        for d in NEIGHBORS_26:
            neighbor = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
            if neighbor not in solid_cells:
                continue
            nearest = np.clip(centroid, np.array(neighbor), np.array(neighbor) + 1.0)   # the nearest point of the neighbor's box
            rank = (float(np.sum((centroid - nearest) ** 2)), sum(abs(x) for x in d), neighbor)
            if best is None or rank < best[0]:
                best = (rank, neighbor)
        if best is None:
            dropped += 1
        else:
            owner[cell] = best[1]
            reassigned += 1
    chunk_of = {cell: cell for cell in solid_cells}   # the voxels that have no pieces are plain boxes, but they are solid: a chunk next to one needs a cap
    chunk_of.update(owner)

    # 4. the geometry of every chunk: its pieces, and the caps of the cells' faces that border another solid
    parity = PointParity(outer_tris) if len(outer_tris) else None
    chunks = defaultdict(lambda: {"vertices": [], "triangles": []})   # voxel cell -> lists of (position, normal) and of triples of vertex numbers
    cap_count = 0

    def add_triangle(chunk, corners):
        base = len(chunk["vertices"])
        chunk["vertices"].extend(corners)
        chunk["triangles"].append((base, base + 1, base + 2))

    for cell, cell_pieces in pieces.items():
        if cell not in owner:
            continue
        chunk = chunks[owner[cell]]
        for _, polygon in cell_pieces:
            for i in range(1, len(polygon) - 1):
                add_triangle(chunk, [(polygon[0][:3], polygon[0][3:]), (polygon[i][:3], polygon[i][3:]), (polygon[i + 1][:3], polygon[i + 1][3:])])
        polygons = [polygon for _, polygon in cell_pieces]
        for axis, side in FACES:
            step = [0, 0, 0]
            step[axis] = side
            across = (cell[0] + step[0], cell[1] + step[1], cell[2] + step[2])
            if chunk_of.get(across) == owner[cell]:
                continue                                   # the same chunk goes on there: no face
            if across not in solid_cells and across not in pieces:
                continue                                   # no solid there: the surface of the mesh is the end
            plane = cell[axis] + (1.0 if side > 0 else 0.0)
            u, v = (axis + 1) % 3, (axis + 2) % 3
            segments = slice_segments(polygons, axis, plane - side * CAP_SLICE)

            def is_inside(pu, pv, axis=axis, u=u, v=v, plane=plane, side=side):
                point = [0.0, 0.0, 0.0]
                point[axis] = plane - side * CAP_SLICE
                point[u] = pu
                point[v] = pv
                return parity.inside(point)

            for tri in cap_polygons(segments, cell[u], cell[u] + 1.0, cell[v], cell[v] + 1.0, is_inside):
                corners = []
                for pu, pv in (tri if side > 0 else (tri[0], tri[2], tri[1])):   # counterclockwise in (u, v) looks along -axis from +axis: normal +axis
                    point = [0.0, 0.0, 0.0]
                    point[axis] = plane
                    point[u] = pu
                    point[v] = pv
                    corners.append((tuple(point), tuple(float(side) if a == axis else 0.0 for a in range(3))))
                add_triangle(chunk, corners)
                cap_count += 1

    # 4b. a voxel that has no pieces of its own but owns the pieces of the cells around it (the surface passes outside of it) is solid all through: its faces
    #     toward other chunks are whole squares
    for cell in sorted(set(owner.values()) - set(pieces)):
        chunk = chunks[cell]
        for axis, side in FACES:
            step = [0, 0, 0]
            step[axis] = side
            across = (cell[0] + step[0], cell[1] + step[1], cell[2] + step[2])
            if chunk_of.get(across) == cell:
                continue
            if across not in solid_cells and across not in pieces:
                continue
            plane = cell[axis] + (1.0 if side > 0 else 0.0)
            u, v = (axis + 1) % 3, (axis + 2) % 3
            u0, v0 = cell[u], cell[v]
            square = [(u0, v0), (u0 + 1.0, v0), (u0 + 1.0, v0 + 1.0), (u0, v0 + 1.0)]
            for tri in ((square[0], square[1], square[2]), (square[0], square[2], square[3])):
                corners = []
                for pu, pv in (tri if side > 0 else (tri[0], tri[2], tri[1])):
                    point = [0.0, 0.0, 0.0]
                    point[axis] = plane
                    point[u] = pu
                    point[v] = pv
                    corners.append((tuple(point), tuple(float(side) if a == axis else 0.0 for a in range(3))))
                add_triangle(chunk, corners)
                cap_count += 1

    # 5. out: relative to the center of the voxel, in the units of the output, welded
    result = ChunkSet()
    positions, normals_out, triangles_out = [], [], []
    for voxel_cell in sorted(chunks, key=lambda c: solid_cells[c]):
        chunk = chunks[voxel_cell]
        center = np.array(voxel_cell, dtype=np.float64) + 0.5
        welded = {}
        local_positions, local_normals, local_triangles = [], [], []
        remap = []
        for position, normal in chunk["vertices"]:
            p = (np.array(position) - center) * size
            n = np.array(normal)
            length = np.linalg.norm(n)
            n = n / length if length > 1e-12 else n
            key = (tuple(np.round(p * WELD_POSITION).astype(np.int64)), tuple(np.round(n * WELD_NORMAL).astype(np.int64)))
            if key not in welded:
                welded[key] = len(local_positions)
                local_positions.append(p)
                local_normals.append(n)
            remap.append(welded[key])
        for a, b, c in chunk["triangles"]:
            a, b, c = remap[a], remap[b], remap[c]
            if a != b and b != c and a != c:
                local_triangles.append((a, b, c))
        if not local_triangles:
            continue
        result.voxel_numbers.append(solid_cells[voxel_cell])
        positions.extend(local_positions)
        normals_out.extend(local_normals)
        triangles_out.extend(local_triangles)
        result.vertex_start.append(result.vertex_start[-1] + len(local_positions))
        result.triangle_start.append(result.triangle_start[-1] + len(local_triangles))
    result.positions = np.array(positions, dtype=np.float64).reshape(-1, 3)
    result.normals = np.array(normals_out, dtype=np.float64).reshape(-1, 3)
    result.triangles = np.array(triangles_out, dtype=np.int64).reshape(-1, 3)
    result.stats = {
        "chunks": result.count,
        "vertices": len(result.positions),
        "triangles": len(result.triangles),
        "cap_triangles": cap_count,
        "outside_cells_given_to_a_voxel": reassigned,
        "pieces_lost": dropped,
        "cavity_triangles_left_out": int(len(tris) - len(outer)),
    }
    return result
