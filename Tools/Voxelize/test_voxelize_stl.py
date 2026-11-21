#!/usr/bin/env python3
"""Tests of voxelize_stl.py: python3 -m unittest Tools/Voxelize/test_voxelize_stl.py   (or run this file). Needs numpy."""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voxelize_stl as vox  # noqa: E402


# ---- meshes ----------------------------------------------------------------------------------------------------------

def box_mesh(lo, hi, inward=False):
    """The 12 triangles of an axis-aligned box, facing outward (or inward, for the wall of a cavity)."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    corners = np.array([[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
    # the corners are numbered x + 2 y + 4 z; each face is two triangles, counter-clockwise seen from outside
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    triangles = []
    for a, b, c, d in faces:
        triangles += [(corners[a], corners[b], corners[c]), (corners[a], corners[c], corners[d])]
    triangles = np.array(triangles)
    return triangles[:, ::-1] if inward else triangles


def icosphere(radius, subdivisions):
    t = (1 + 5 ** 0.5) / 2
    vertices = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    vertices = [np.array(v, float) / np.linalg.norm(v) for v in vertices]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    triangles = [(vertices[a], vertices[b], vertices[c]) for a, b, c in faces]
    for _ in range(subdivisions):
        refined = []
        for a, b, c in triangles:
            ab, bc, ca = (a + b), (b + c), (c + a)
            ab, bc, ca = ab / np.linalg.norm(ab), bc / np.linalg.norm(bc), ca / np.linalg.norm(ca)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        triangles = refined
    return np.array(triangles) * radius


def write_binary(path, triangles):
    with open(path, "wb") as f:
        f.write(b"solid test".ljust(80, b" "))  # a binary STL may well start with 'solid'
        f.write(struct.pack("<I", len(triangles)))
        for t in triangles:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            f.write(struct.pack("<12fH", *n, *t.reshape(-1), 0))


def write_ascii(path, triangles):
    with open(path, "w") as f:
        f.write("solid test\n")
        for t in triangles:
            f.write("facet normal 0 0 0\n outer loop\n")
            for v in t:
                f.write("  vertex %.9g %.9g %.9g\n" % tuple(v))
            f.write(" endloop\nendfacet\n")
        f.write("endsolid test\n")


def write_obj(path, triangles):
    with open(path, "w") as f:
        f.write("# test\n")
        for t in triangles:
            for v in t:
                f.write("v %.17g %.17g %.17g\n" % tuple(v))
        for n in range(len(triangles)):
            f.write("f %d %d %d\n" % (3 * n + 1, 3 * n + 2, 3 * n + 3))


class VoxelizerTests(unittest.TestCase):
    def test_reads_binary_and_ascii_alike(self):
        mesh = icosphere(1.0, 1).astype(np.float32).astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            write_binary(os.path.join(d, "b.stl"), mesh)
            write_ascii(os.path.join(d, "a.stl"), mesh)
            np.testing.assert_allclose(vox.read_stl(os.path.join(d, "b.stl")), mesh, atol=1e-6)
            np.testing.assert_allclose(vox.read_stl(os.path.join(d, "a.stl")), mesh, atol=1e-6)

    def test_rejects_a_file_that_is_not_an_stl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.stl")
            with open(path, "w") as f:
                f.write("hello")
            with self.assertRaises(ValueError):
                vox.read_stl(path)

    def test_a_cube_fills_the_grid_and_every_neighbor_pair_is_joined_once(self):
        solid, origin, size = vox.voxelize(box_mesh((0, 0, 0), (1, 1, 1)), voxel_size=0.25)
        self.assertEqual(solid.shape, (4, 4, 4))
        self.assertTrue(solid.all())
        np.testing.assert_allclose(origin, 0.0)
        voxels, joints = vox.build_joints(solid, 6)
        self.assertEqual(len(voxels), 64)
        self.assertEqual(len(joints), 3 * 4 * 4 * 3)  # 3 directions, 4 x 4 lines of 3 pairs
        self.assertEqual(len({tuple(sorted(j)) for j in joints.tolist()}), len(joints), "a pair appears once")
        d = np.abs(voxels[joints[:, 0]] - voxels[joints[:, 1]])
        self.assertTrue(np.all(d.sum(axis=1) == 1), "6-neighborhood: the two voxels share a face")

    def test_connectivity_18_and_26_add_the_edge_and_corner_neighbors(self):
        solid = np.ones((2, 2, 2), bool)
        self.assertEqual(len(vox.build_joints(solid, 6)[1]), 12)
        self.assertEqual(len(vox.build_joints(solid, 18)[1]), 12 + 12)  # + the 12 face diagonals (2 on each of the 6 faces)
        self.assertEqual(len(vox.build_joints(solid, 26)[1]), 12 + 12 + 4)  # + the 4 space diagonals

    def test_voxels_are_numbered_in_the_order_i_j_k(self):
        solid = np.zeros((3, 2, 2), bool)
        solid[0, 0, 1] = solid[0, 1, 0] = solid[2, 0, 0] = True
        voxels, _ = vox.build_joints(solid, 6)
        self.assertEqual(voxels.tolist(), [[0, 0, 1], [0, 1, 0], [2, 0, 0]])

    def test_a_hollow_cube_with_a_cavity_is_filled(self):
        # an outer box and a reversed box inside it (a closed cavity): the voxelized solid has no hole
        mesh = np.concatenate([box_mesh((0, 0, 0), (3, 3, 3)), box_mesh((1, 1, 1), (2, 2, 2), inward=True)])
        for surface in ("center", "touch"):
            solid, _, _ = vox.voxelize(mesh, voxel_size=0.5, surface=surface)
            self.assertEqual(solid.shape, (6, 6, 6))
            self.assertTrue(solid.all(), surface)

    def test_a_surface_only_sphere_is_solid_and_has_about_the_volume_of_the_sphere(self):
        mesh = icosphere(1.0, 3)
        volume = vox.mesh_volume(mesh)
        center, _, size = vox.voxelize(mesh, voxel_size=0.05, surface="center")
        touch, _, _ = vox.voxelize(mesh, voxel_size=0.05, surface="touch")
        center_volume = center.sum() * size ** 3
        self.assertAlmostEqual(center_volume / volume, 1.0, delta=0.03)
        self.assertGreater(touch.sum() * size ** 3, volume, "touch is conservative")
        self.assertGreater(touch.sum(), center.sum())
        # solid: the middle is filled, and the line through the middle has no gap
        c = np.array(center.shape) // 2
        self.assertTrue(center[c[0], c[1], :].all() or center[c[0], c[1], 1:-1].all())
        # every voxel of center is also in touch
        self.assertTrue(np.all(touch[center]))

    def test_a_tilted_closed_mesh_does_not_depend_on_the_winding(self):
        mesh = icosphere(1.0, 2)
        a, _, _ = vox.voxelize(mesh, voxel_size=0.1)
        b, _, _ = vox.voxelize(mesh[:, ::-1], voxel_size=0.1)
        np.testing.assert_array_equal(a, b)

    def test_resolution_sets_the_number_of_voxels_along_the_longest_side(self):
        mesh = box_mesh((0, 0, 0), (4, 2, 1))
        solid, _, size = vox.voxelize(mesh, resolution=16)
        self.assertEqual(solid.shape, (16, 8, 4))
        self.assertAlmostEqual(size, 0.25)

    def test_voxel_size_and_resolution_are_exclusive_and_one_is_needed(self):
        mesh = box_mesh((0, 0, 0), (1, 1, 1))
        with self.assertRaises(ValueError):
            vox.voxelize(mesh)
        with self.assertRaises(ValueError):
            vox.voxelize(mesh, voxel_size=0.1, resolution=10)
        with self.assertRaises(ValueError):
            vox.voxelize(mesh, voxel_size=-1.0)

    def test_refuses_a_grid_over_the_limit(self):
        with self.assertRaises(ValueError):
            vox.voxelize(box_mesh((0, 0, 0), (1, 1, 1)), voxel_size=0.01, max_cells=1000)

    def test_an_open_mesh_is_reported(self):
        closed = box_mesh((0, 0, 0), (1, 1, 1))
        self.assertEqual(vox.open_edge_count(closed), 0)
        self.assertGreater(vox.open_edge_count(closed[:-2]), 0)  # a face missing

    def test_the_fill_is_not_a_bounding_box(self):
        # a bar and a post that do not touch (two closed meshes): the empty corner beside the post stays empty, in both modes
        mesh = np.concatenate([box_mesh((0, 0, 0), (3, 1, 1)), box_mesh((0, 1.5, 0), (1, 3, 1))])
        solid, _, _ = vox.voxelize(mesh, voxel_size=0.5, surface="center")
        self.assertFalse(solid[2, 5, 0])
        self.assertFalse(solid[4, 4, 0])
        self.assertFalse(solid[0, 2, 0], "the gap between the bar and the post")
        self.assertTrue(solid[0, 0, 0] and solid[0, 5, 1] and solid[5, 0, 0])
        # not aligned with the grid, so that no face grazes a neighboring voxel: touch leaves the corner empty too
        solid, _, _ = vox.voxelize(mesh, voxel_size=0.4, surface="touch")
        self.assertFalse(solid[5, 6, 0])
        self.assertTrue(solid[0, 0, 0] and solid[0, 6, 1])

    def test_z_up_to_y_up_turns_the_mesh(self):
        mesh = box_mesh((0, 0, 0), (4, 2, 1))          # x 4, y 2, z 1 (z is the height in a z-up tool)
        turned = vox.transform(mesh, 1.0, True)
        extent = np.ptp(turned.reshape(-1, 3), axis=0)
        np.testing.assert_allclose(extent, [4, 1, 2])  # the height is now y
        self.assertAlmostEqual(vox.mesh_volume(turned), 8.0)
        v0 = np.einsum("ij,ij->i", turned[:, 0], np.cross(turned[:, 1], turned[:, 2])).sum()
        v1 = np.einsum("ij,ij->i", mesh[:, 0], np.cross(mesh[:, 1], mesh[:, 2])).sum()
        self.assertAlmostEqual(v0, v1, msg="a rotation keeps the orientation")

    def test_scale_changes_the_units(self):
        solid, origin, size = vox.voxelize(vox.transform(box_mesh((0, 0, 0), (1000, 1000, 1000)), 0.001, False), voxel_size=0.25)
        self.assertEqual(solid.shape, (4, 4, 4))

    def test_the_command_line_writes_a_file_that_agrees_with_itself(self):
        mesh = icosphere(1.0, 2).astype(np.float32).astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "ball.stl")
            out = os.path.join(d, "ball.json")
            write_binary(stl, mesh)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(vox.main([stl, "-o", out, "--voxel-size", "0.2", "--connectivity", "6"]), 0)
            with open(out) as f:
                data = json.load(f)
            self.assertEqual(data["format"], vox.FORMAT_NAME)
            self.assertEqual(data["version"], vox.FORMAT_VERSION)
            n, m = data["voxelCount"], data["jointCount"]
            self.assertEqual(len(data["voxels"]), 3 * n)
            self.assertEqual(len(data["joints"]), 2 * m)
            voxels = np.array(data["voxels"]).reshape(-1, 3)
            joints = np.array(data["joints"]).reshape(-1, 2)
            self.assertTrue(np.all((joints >= 0) & (joints < n)))
            self.assertTrue(np.all(np.abs(voxels[joints[:, 0]] - voxels[joints[:, 1]]).sum(axis=1) == 1))
            dims = np.array(data["gridDimensions"])
            self.assertTrue(np.all(voxels >= 0) and np.all(voxels < dims))
            self.assertAlmostEqual(data["voxelSize"], 0.2)
            np.testing.assert_allclose(data["origin"], mesh.reshape(-1, 3).min(axis=0), atol=1e-6)

    def test_the_command_line_warns_about_an_open_mesh_and_fails_cleanly_on_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "open.stl")
            write_ascii(stl, box_mesh((0, 0, 0), (1, 1, 1))[:-2])
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(vox.main([stl, "-o", os.path.join(d, "o.json"), "--resolution", "8"]), 0)
            self.assertIn("not closed", err.getvalue())
            missing = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(missing):
                self.assertEqual(vox.main([os.path.join(d, "nothing.stl"), "--resolution", "8"]), 1)
            self.assertIn("error", missing.getvalue())

    # ---- OBJ input ----

    def test_reads_an_obj_with_the_face_forms_comments_quads_and_negative_indices(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.obj")
            with open(path, "w") as f:
                f.write("# a comment\nmtllib m.mtl\no thing\nv 0 0 0\nv 1 0 0 1.0\nv 1 1 0\nv 0 1 0 # trailing comment\nvt 0 0\nvn 0 0 1\ng part\ns off\n"
                        "f 1/1/1 2/1/1 3/1/1 4/1/1\n"   # a quad: two triangles
                        "f -4//1 -3//1 \\\n -2//1\n"    # negative indices, v//vn, a line that goes on
                        "l 1 2\np 3\nusemtl x\n")
            tris = vox.read_obj(path)
            self.assertEqual(tris.shape, (3, 3, 3))
            np.testing.assert_array_equal(tris[0], [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
            np.testing.assert_array_equal(tris[1], [[0, 0, 0], [1, 1, 0], [0, 1, 0]])
            np.testing.assert_array_equal(tris[2], [[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    def test_an_obj_is_read_when_the_name_ends_with_obj_in_any_case_and_an_stl_otherwise(self):
        mesh = box_mesh((0, 0, 0), (1, 2, 3))
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.obj", "b.OBJ"):
                write_obj(os.path.join(d, name), mesh)
                np.testing.assert_allclose(vox.read_mesh(os.path.join(d, name)), mesh, atol=1e-9)
            write_binary(os.path.join(d, "c.stl"), mesh)
            np.testing.assert_allclose(vox.read_mesh(os.path.join(d, "c.stl")), mesh, atol=1e-6)

    def test_an_obj_and_an_stl_of_the_same_mesh_give_the_same_voxels(self):
        mesh = icosphere(1.0, 2).astype(np.float32).astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            write_obj(os.path.join(d, "ball.obj"), mesh)
            write_binary(os.path.join(d, "ball.stl"), mesh)
            from_obj = vox.voxelize(vox.read_mesh(os.path.join(d, "ball.obj")), voxel_size=0.2)
            from_stl = vox.voxelize(vox.read_mesh(os.path.join(d, "ball.stl")), voxel_size=0.2)
            np.testing.assert_array_equal(from_obj[0], from_stl[0])
            np.testing.assert_allclose(from_obj[1], from_stl[1], atol=1e-6)

    def test_a_bad_obj_is_refused_with_the_line_number(self):
        with tempfile.TemporaryDirectory() as d:
            for text, expected in (("v 0 0 0\nv 1 0 0\nf 1 2 3\n", "line 3"),          # vertex 3 does not exist
                                   ("v 0 0 0\nv 1 0\n", "line 2"),                      # a vertex with two numbers
                                   ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n", "line 4"),  # not an index
                                   ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n", "line 4"),    # two corners
                                   ("v 0 0 0\nv 1 0 0\nv 0 1 0\n", "no faces")):
                path = os.path.join(d, "bad.obj")
                with open(path, "w") as f:
                    f.write(text)
                with self.assertRaises(ValueError) as caught:
                    vox.read_obj(path)
                self.assertIn(expected, str(caught.exception))

    def test_the_command_line_voxelizes_an_obj(self):
        mesh = box_mesh((0, 0, 0), (1, 1, 1))
        with tempfile.TemporaryDirectory() as d:
            obj = os.path.join(d, "box.obj")
            write_obj(obj, mesh)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(vox.main([obj, "--voxel-size", "0.25"]), 0)
            with open(os.path.join(d, "box.voxels.json")) as f:
                data = json.load(f)
            self.assertEqual(data["voxelCount"], 64)
            self.assertEqual(data["gridDimensions"], [4, 4, 4])
            self.assertEqual(data["source"], "box.obj")


if __name__ == "__main__":
    unittest.main()
