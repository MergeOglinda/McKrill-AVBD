#!/usr/bin/env python3
"""Writes the three meshes that the samples of the JSON Mesh Import Demo were voxelized from, as binary STL files: Sphere.stl, Torus.stl and Cube.stl.

    python3 Tools/Voxelize/make_sample_meshes.py some/folder

They are z-up like the output of most modeling tools, the way the samples were made (see the commands in README.md). Voxelizing these meshes gives exactly the voxels
and joints of the files in Samples~/JsonMeshImportDemo/Resources/JsonMeshImportDemo (the test suite checks it), and with --smooth the chunks.

  * Sphere: a UV sphere of radius 1.5 with 32 segments and 16 rings, the poles on the z axis.
  * Torus: major radius 1.2, minor radius 0.5, 48 x 16 quads, the axis along z.
  * Cube: a box from (0, 0, 0) to (2, 2, 2) with a closed cavity (a box from 0.5 to 1.5 whose triangles face into the cavity), to show that the cavity is filled.
"""
import os
import struct
import sys

import numpy as np


def uv_sphere(radius, segments, rings):
    """Triangles facing outward (counterclockwise seen from outside); the poles are on +z and -z."""
    def vertex(ring, segment):
        theta = np.pi * ring / rings
        phi = 2.0 * np.pi * (segment % segments) / segments
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]) * radius

    triangles = []
    for ring in range(rings):
        for segment in range(segments):
            a, b, c, d = vertex(ring, segment), vertex(ring, segment + 1), vertex(ring + 1, segment + 1), vertex(ring + 1, segment)
            if ring > 0:
                triangles.append((a, d, b))
            if ring < rings - 1:
                triangles.append((b, d, c))
    return np.array(triangles)


def torus(major_radius, minor_radius, major_segments, minor_segments):
    def point(i, j):
        u = 2.0 * np.pi * (i % major_segments) / major_segments
        w = 2.0 * np.pi * (j % minor_segments) / minor_segments
        ring = major_radius + minor_radius * np.cos(w)
        return np.array([ring * np.cos(u), ring * np.sin(u), minor_radius * np.sin(w)])

    triangles = []
    for i in range(major_segments):
        for j in range(minor_segments):
            a, b, c, d = point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)
            triangles += [(a, b, c), (a, c, d)]
    return np.array(triangles)


def box(lo, hi, inward=False):
    lo, hi = np.array(lo, float), np.array(hi, float)
    corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles = []
    for a, b, c, d in faces:
        triangles += [(corners[a], corners[b], corners[c]), (corners[a], corners[c], corners[d])]
    triangles = np.array(triangles)
    return triangles[:, ::-1] if inward else triangles


def outward(triangles):
    """The winding turned so that the mesh has a positive volume (outward facing)."""
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    return triangles if volume > 0 else triangles[:, ::-1]


def write_binary_stl(path, triangles, name):
    with open(path, "wb") as f:
        f.write(name.encode("ascii").ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        for t in triangles:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            length = np.linalg.norm(n)
            n = n / length if length > 0 else n
            f.write(struct.pack("<12fH", *n, *t.reshape(-1), 0))


def sample_meshes():
    """The meshes by name."""
    return {
        "Sphere": outward(uv_sphere(1.5, 32, 16)),
        "Torus": outward(torus(1.2, 0.5, 48, 16)),
        "Cube": np.concatenate([box((0, 0, 0), (2, 2, 2)), box((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), inward=True)]),
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    os.makedirs(argv[1], exist_ok=True)
    for name, triangles in sample_meshes().items():
        path = os.path.join(argv[1], name + ".stl")
        write_binary_stl(path, triangles, "solid " + name)
        print("%s: %d triangles" % (path, len(triangles)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
