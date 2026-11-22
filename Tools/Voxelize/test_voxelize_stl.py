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
import smooth_chunks as chunks_module  # noqa: E402
import make_sample_meshes as samples  # noqa: E402


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

    def test_connect_isolated_joins_voxels_without_a_face_neighbor_to_their_edge_and_corner_neighbors(self):
        # a diagonal line of voxels: no face neighbors at all, so each one joins its edge/corner neighbors
        solid = np.zeros((4, 4, 4), bool)
        for i in range(4):
            solid[i, i, i] = True
        self.assertEqual(len(vox.build_joints(solid, 6)[1]), 0)
        voxels, joints = vox.build_joints(solid, 6, connect_isolated=True)
        self.assertEqual(len(joints), 3)
        self.assertTrue(np.all(np.abs(voxels[joints[:, 0]] - voxels[joints[:, 1]]).sum(axis=1) == 3))
        # voxels that share an edge
        solid = np.zeros((3, 3, 3), bool)
        solid[0, 0, 0] = solid[1, 1, 0] = True
        self.assertEqual(len(vox.build_joints(solid, 6, connect_isolated=True)[1]), 1)

    def test_connect_isolated_adds_nothing_to_voxels_that_have_a_face_neighbor(self):
        # a 2 x 2 x 2 block: every voxel has face neighbors, so no face diagonals or space diagonals
        solid = np.ones((2, 2, 2), bool)
        self.assertEqual(len(vox.build_joints(solid, 6, connect_isolated=True)[1]), 12)
        # two face-connected voxels and a third one that touches only at a corner of the second
        solid = np.zeros((4, 4, 4), bool)
        solid[0, 0, 0] = solid[1, 0, 0] = True
        solid[2, 1, 1] = True  # corner neighbor of (1, 0, 0), no face neighbor
        voxels, joints = vox.build_joints(solid, 6, connect_isolated=True)
        pairs = {tuple(sorted(map(tuple, voxels[j].tolist()))) for j in joints}
        self.assertEqual(pairs, {((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (2, 1, 1))})

    def test_connect_isolated_tries_corner_neighbors_only_without_edge_neighbors(self):
        # A = (2, 2, 2) has an edge neighbor B and a corner neighbor C = (1, 1, 1), which has an edge neighbor D: nobody falls back to corners
        solid = np.zeros((4, 4, 4), bool)
        for v in [(2, 2, 2), (3, 3, 2), (1, 1, 1), (0, 0, 1)]:
            solid[v] = True
        voxels, joints = vox.build_joints(solid, 6, connect_isolated=True)
        pairs = {tuple(sorted(map(tuple, voxels[j].tolist()))) for j in joints}
        self.assertEqual(pairs, {((2, 2, 2), (3, 3, 2)), ((0, 0, 1), (1, 1, 1))})
        # without the edge neighbors the corner neighbors are used
        solid[3, 3, 2] = solid[0, 0, 1] = False
        voxels, joints = vox.build_joints(solid, 6, connect_isolated=True)
        self.assertEqual({tuple(sorted(map(tuple, voxels[j].tolist()))) for j in joints}, {((1, 1, 1), (2, 2, 2))})

    def test_connect_isolated_off_and_with_18_or_26(self):
        solid = np.zeros((2, 2, 2), bool)
        solid[0, 0, 0] = solid[1, 1, 1] = True
        self.assertEqual(len(vox.build_joints(solid, 6)[1]), 0)
        self.assertEqual(len(vox.build_joints(solid, 6, connect_isolated=False)[1]), 0)
        with self.assertRaises(ValueError):
            vox.build_joints(solid, 18, connect_isolated=True)
        with self.assertRaises(ValueError):
            vox.build_joints(solid, 26, connect_isolated=True)

    def test_connect_isolated_with_no_voxels(self):
        voxels, joints = vox.build_joints(np.zeros((2, 2, 2), bool), 6, connect_isolated=True)
        self.assertEqual((len(voxels), len(joints)), (0, 0))

    def test_connect_isolated_option_on_the_command_line(self):
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "b.stl")
            write_binary(stl, box_mesh((0, 0, 0), (1, 1, 1)))
            plain, flagged = os.path.join(d, "p.json"), os.path.join(d, "f.json")
            self.assertEqual(vox.main([stl, "-o", plain, "--voxel-size", "0.25"]), 0)
            self.assertEqual(vox.main([stl, "-o", flagged, "--voxel-size", "0.25", "--connect-isolated"]), 0)
            with open(plain) as f:
                a = json.load(f)
            with open(flagged) as f:
                b = json.load(f)
            self.assertNotIn("connectIsolated", a)
            self.assertTrue(b["connectIsolated"])
            self.assertEqual(b["connectivity"], 6)
            self.assertEqual(a["jointCount"], b["jointCount"], "a solid box has no isolated voxel")
            with self.assertRaises(SystemExit):
                with redirect_stderr(io.StringIO()):
                    vox.main([stl, "-o", flagged, "--voxel-size", "0.25", "--connect-isolated", "--connectivity", "18"])
            self.assertEqual(vox.main([stl, "-o", flagged, "--voxel-size", "0.25", "--connect-isolated", "--connectivity", "6"]), 0)

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


# ---- the smooth surface: chunks --------------------------------------------------------------------------------------

SAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Packages", "com.mckrill.avbd", "Samples~", "JsonMeshImportDemo", "Resources", "JsonMeshImportDemo")


def bake(mesh, voxel_size, surface="center", crease=50.0):
    solid, origin, size = vox.voxelize(mesh, voxel_size=voxel_size, surface=surface)
    voxels, _ = vox.build_joints(solid, 6)
    return chunks_module.bake(mesh, origin, size, solid, voxels, crease), voxels, size


def chunk_arrays(chunks, i):
    p = chunks.positions[chunks.vertex_start[i]:chunks.vertex_start[i + 1]]
    n = chunks.normals[chunks.vertex_start[i]:chunks.vertex_start[i + 1]]
    t = chunks.triangles[chunks.triangle_start[i]:chunks.triangle_start[i + 1]]
    return p, n, t


def volume_and_area_vector(chunks, i):
    """The volume of the closed surface of chunk i (a sum of tetrahedra to the origin, the center of the voxel) and its area vector, which is zero for a closed surface."""
    p, _, t = chunk_arrays(chunks, i)
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0), np.cross(b - a, c - a).sum(axis=0) / 2.0


class SmoothChunkTests(unittest.TestCase):
    def assert_closed_and_adds_up(self, mesh, voxel_size, surface="center", relative=2e-3, expected_volume=None):
        chunks, voxels, size = bake(mesh, voxel_size, surface)
        self.assertGreater(chunks.count, 0)
        total = 0.0
        for i in range(chunks.count):
            volume, area_vector = volume_and_area_vector(chunks, i)
            self.assertGreater(volume, 0.0, "chunk %d faces outward" % i)
            # the cap is cut a hair (1e-4 of a cell) inside the cell, so the area vector is not exactly zero; a missing face would be about a tenth of a cell face
            self.assertLess(np.linalg.norm(area_vector), 1e-3 * size ** 2, "chunk %d is closed (the area vector of a closed surface is zero)" % i)
            total += volume
        total += (len(voxels) - chunks.count) * size ** 3     # the voxels without a chunk are whole boxes
        expected = vox.mesh_volume(mesh) if expected_volume is None else expected_volume
        self.assertAlmostEqual(total / expected, 1.0, delta=relative, msg="the chunks and the boxes of the inside make up the mesh")
        return chunks, voxels, size

    def test_the_chunks_of_a_sphere_are_closed_and_make_up_the_mesh(self):
        chunks, _, _ = self.assert_closed_and_adds_up(icosphere(1.0, 2), 0.25)
        self.assertGreater(chunks.stats["outside_cells_given_to_a_voxel"], 0, "the mesh passes through cells whose center is outside")
        self.assertEqual(chunks.stats["pieces_lost"], 0)
        self.assertEqual(chunks.stats["cavity_triangles_left_out"], 0)

    def test_it_also_works_when_every_cell_the_mesh_touches_is_a_voxel(self):
        chunks, _, _ = self.assert_closed_and_adds_up(icosphere(1.0, 2), 0.25, surface="touch")
        self.assertEqual(chunks.stats["outside_cells_given_to_a_voxel"], 0)

    def test_a_box_of_whole_voxels_has_cubes_as_chunks(self):
        chunks, voxels, size = self.assert_closed_and_adds_up(box_mesh((0, 0, 0), (1, 1, 1)), 0.25)
        self.assertEqual(chunks.count, 64 - 8, "the 8 voxels in the middle have no surface")
        for i in range(chunks.count):
            self.assertAlmostEqual(volume_and_area_vector(chunks, i)[0] / size ** 3, 1.0, delta=2e-3)
            p, _, _ = chunk_arrays(chunks, i)
            self.assertLessEqual(np.abs(p).max(), 0.5 * size * (1 + 1e-3), "the chunk of a cell of the box does not leave its cell")

    def test_a_cavity_is_not_a_hole_and_its_triangles_are_left_out(self):
        mesh = np.concatenate([box_mesh((0, 0, 0), (3, 3, 3)), box_mesh((1, 1, 1), (2, 2, 2), inward=True)])
        chunks, _, _ = self.assert_closed_and_adds_up(mesh, 0.5, expected_volume=27.0)   # the object is solid: the volume without the cavity
        self.assertEqual(chunks.stats["cavity_triangles_left_out"], 12)

    def test_the_winding_and_the_normals_agree_and_face_outward(self):
        chunks, _, size = bake(icosphere(1.0, 2), 0.25)
        for i in range(chunks.count):
            p, n, t = chunk_arrays(chunks, i)
            self.assertLess(abs(np.linalg.norm(n, axis=1) - 1.0).max(), 2e-3, "unit normals")
            face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
            mean_normal = n[t].sum(axis=1)
            self.assertTrue(np.all(np.einsum("ij,ij->i", face, mean_normal) > 0.0), "the winding agrees with the normals")

    def test_the_winding_of_the_mesh_does_not_matter(self):
        a, _, _ = bake(icosphere(1.0, 2), 0.25)
        b, _, _ = bake(icosphere(1.0, 2)[:, ::-1], 0.25)
        self.assertEqual(a.voxel_numbers, b.voxel_numbers)
        self.assertEqual(a.stats["triangles"], b.stats["triangles"])
        order_a = np.lexsort(np.round(a.positions * 1e5).T)
        order_b = np.lexsort(np.round(b.positions * 1e5).T)
        np.testing.assert_allclose(a.positions[order_a], b.positions[order_b], atol=2e-5)   # the same vertices, in another order

    def test_a_chunk_reaches_out_of_its_cell_by_less_than_a_cell(self):
        chunks, _, size = bake(icosphere(1.0, 2), 0.25)
        for i in range(chunks.count):
            self.assertLessEqual(np.abs(chunk_arrays(chunks, i)[0]).max(), 1.5 * size * (1 + 1e-3))

    def test_a_torus_with_a_hole_works_too(self):
        self.assert_closed_and_adds_up(samples.outward(samples.torus(1.2, 0.5, 24, 12)), 0.2, relative=5e-3)

    def test_normals_are_smooth_on_a_sphere_and_sharp_on_a_box(self):
        mesh = icosphere(1.0, 2)
        smooth = chunks_module.corner_normals(mesh, 50.0)
        corners = mesh.reshape(-1, 3)
        np.testing.assert_allclose(smooth.reshape(-1, 3), corners / np.linalg.norm(corners, axis=1)[:, None], atol=0.06)  # about the normals of a sphere
        sharp = chunks_module.corner_normals(box_mesh((0, 0, 0), (1, 1, 1)), 50.0)
        face = np.cross(box_mesh((0, 0, 0), (1, 1, 1))[:, 1] - box_mesh((0, 0, 0), (1, 1, 1))[:, 0], box_mesh((0, 0, 0), (1, 1, 1))[:, 2] - box_mesh((0, 0, 0), (1, 1, 1))[:, 0])
        face /= np.linalg.norm(face, axis=1)[:, None]
        np.testing.assert_allclose(sharp, np.repeat(face[:, None, :], 3, axis=1), atol=1e-9)   # the corners of a box keep the normal of their face

    def test_the_inside_test_for_points(self):
        mesh = chunks_module.to_cell_units(icosphere(1.0, 2), np.array([-1.0, -1.0, -1.0]), 0.25, (8, 8, 8))
        parity = chunks_module.PointParity(mesh)
        self.assertTrue(parity.inside([4.0, 4.0, 4.0]))
        self.assertTrue(parity.inside([4.0, 4.0, 6.5]))
        self.assertTrue(parity.inside([4.0, 4.0, 7.9]))    # the sphere has a radius of 4 cells
        self.assertFalse(parity.inside([4.0, 4.0, 8.3]))
        self.assertFalse(parity.inside([0.2, 0.2, 4.0]))

    def test_caps(self):
        full = chunks_module.cap_polygons([], 0.0, 1.0, 0.0, 1.0, lambda u, v: True)
        self.assertAlmostEqual(sum(0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) for t in full), 1.0)
        self.assertEqual(len(full), 2)
        self.assertEqual(chunks_module.cap_polygons([], 0.0, 1.0, 0.0, 1.0, lambda u, v: False), [])

        def area(triangles):
            return sum(0.5 * ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) for t in triangles)

        # a bent cut from the left edge to the top edge: the inside is below it, or above it
        cut = [((0.0, 0.5), (0.4, 0.6)), ((0.4, 0.6), (0.7, 1.0))]
        below = lambda u, v: v < (0.5 + 0.25 * u if u < 0.4 else 0.6 + (u - 0.4) / 0.3 * 0.4)
        lower = chunks_module.cap_polygons(cut, 0.0, 1.0, 0.0, 1.0, below)
        upper = chunks_module.cap_polygons(cut, 0.0, 1.0, 0.0, 1.0, lambda u, v: not below(u, v))
        self.assertAlmostEqual(area(lower) + area(upper), 1.0, places=9)
        self.assertAlmostEqual(area(upper), 0.5 * 0.7 * 0.5 + 0.5 * 0.4 * 0.1 * 0 + 0.0, delta=1.0)   # the two parts are not empty, and they are counterclockwise
        self.assertTrue(all(((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])) > 0 for t in lower + upper))
        self.assertLess(len(lower) + len(upper), 12, "the polygons are cut into few triangles")
        # a square hole in the middle (a closed loop): the slabs; the inside is everything but the hole
        hole = [((0.4, 0.4), (0.6, 0.4)), ((0.6, 0.4), (0.6, 0.6)), ((0.6, 0.6), (0.4, 0.6)), ((0.4, 0.6), (0.4, 0.4))]
        outside_hole = chunks_module.cap_polygons(hole, 0.0, 1.0, 0.0, 1.0, lambda u, v: not (0.4 < u < 0.6 and 0.4 < v < 0.6))
        self.assertAlmostEqual(area(outside_hole), 1.0 - 0.04, places=9)
        island = chunks_module.cap_polygons(hole, 0.0, 1.0, 0.0, 1.0, lambda u, v: 0.4 < u < 0.6 and 0.4 < v < 0.6)
        self.assertAlmostEqual(area(island), 0.04, places=9)

    def test_ear_clipping_of_a_concave_polygon(self):
        points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 0.5), (0.0, 2.0)]   # an arrow head pointing down: concave at (1, 0.5)
        triangles = chunks_module.triangulate_polygon(points)
        self.assertEqual(len(triangles), 3)
        total = sum(0.5 * ((points[b][0] - points[a][0]) * (points[c][1] - points[a][1]) - (points[b][1] - points[a][1]) * (points[c][0] - points[a][0])) for a, b, c in triangles)
        self.assertAlmostEqual(total, 4.0 - 1.5, places=9)

    def test_the_command_line_with_smooth_writes_chunks_that_agree_with_the_voxels(self):
        mesh = icosphere(1.0, 2).astype(np.float32).astype(np.float64)
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "ball.stl")
            plain = os.path.join(d, "plain.json")
            smooth = os.path.join(d, "smooth.json")
            write_binary(stl, mesh)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(vox.main([stl, "-o", plain, "--voxel-size", "0.2"]), 0)
                self.assertEqual(vox.main([stl, "-o", smooth, "--voxel-size", "0.2", "--smooth"]), 0)
            with open(plain) as f:
                a = json.load(f)
            with open(smooth) as f:
                b = json.load(f)
            self.assertNotIn("chunkCount", a, "without --smooth the file is as before")
            for key in a:
                self.assertEqual(a[key], b[key], key + " is the same with and without --smooth")
            count = b["chunkCount"]
            self.assertGreater(count, 0)
            self.assertEqual(len(b["chunkVoxels"]), count)
            self.assertEqual(len(b["chunkVertexStart"]), count + 1)
            self.assertEqual(len(b["chunkTriangleStart"]), count + 1)
            vertex_start, triangle_start = np.array(b["chunkVertexStart"]), np.array(b["chunkTriangleStart"])
            self.assertEqual(vertex_start[0], 0)
            self.assertEqual(triangle_start[0], 0)
            self.assertTrue(np.all(np.diff(vertex_start) > 0) and np.all(np.diff(triangle_start) > 0))
            self.assertEqual(len(b["chunkPositions"]), 3 * vertex_start[-1])
            self.assertEqual(len(b["chunkNormals"]), 3 * vertex_start[-1])
            self.assertEqual(len(b["chunkTriangles"]), 3 * triangle_start[-1])
            voxel_numbers = np.array(b["chunkVoxels"])
            self.assertTrue(np.all(np.diff(voxel_numbers) > 0), "ascending and without repeats")
            self.assertTrue(voxel_numbers.min() >= 0 and voxel_numbers.max() < b["voxelCount"])
            triangles = np.array(b["chunkTriangles"]).reshape(-1, 3)
            for i in range(count):
                local = triangles[triangle_start[i]:triangle_start[i + 1]]
                self.assertTrue(local.min() >= 0 and local.max() < vertex_start[i + 1] - vertex_start[i], "the indices of a chunk count from its first vertex")

    def test_the_command_line_says_so_when_a_part_is_thinner_than_a_voxel(self):
        # a flat plate much thinner than a voxel, in a cell no voxel is next to: the voxelizer leaves it out and so does the baking, without failing
        plate = box_mesh((0, 0, 0), (1, 1, 0.01))
        with tempfile.TemporaryDirectory() as d:
            stl = os.path.join(d, "plate.stl")
            write_ascii(stl, np.concatenate([plate, box_mesh((5, 0, 0), (7, 2, 2))]))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(vox.main([stl, "-o", os.path.join(d, "p.json"), "--voxel-size", "0.5", "--smooth"]), 0)

    def test_the_shipped_samples_are_what_the_generated_meshes_give(self):
        # the files of the JSON Mesh Import Demo were baked from make_sample_meshes.py: the voxels and joints are the ones the meshes give, and the chunks are in them
        for name, size, zup in (("Sphere", 0.25, True), ("Torus", 0.2, True), ("Cube", 0.25, False)):
            path = os.path.join(SAMPLE_FOLDER, name + ".json")
            if not os.path.exists(path):
                self.skipTest("the sample folder is not next to the tools")
            with open(path) as f:
                shipped = json.load(f)
            mesh = vox.transform(samples.sample_meshes()[name].astype(np.float32).astype(np.float64), 1.0, zup)
            solid, origin, voxel_size = vox.voxelize(mesh, voxel_size=size)
            voxels, joints = vox.build_joints(solid, 6)
            self.assertEqual(shipped["voxels"], voxels.reshape(-1).tolist(), name)
            self.assertEqual(shipped["joints"], joints.reshape(-1).tolist(), name)
            baked = chunks_module.bake(mesh, origin, voxel_size, solid, voxels)
            self.assertEqual(shipped.get("chunkCount"), baked.count, name + ": re-run the voxelizer with --smooth (see README.md)")
            self.assertEqual(shipped["chunkVoxels"], baked.voxel_numbers, name)
            self.assertEqual(len(shipped["chunkTriangles"]), 3 * baked.stats["triangles"], name)


if __name__ == "__main__":
    unittest.main()
