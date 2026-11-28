# STL and OBJ voxelizer

`voxelize_stl.py` turns a triangle mesh (an STL or OBJ file) into a **solid** set of voxels and writes them, together with the
joints between neighboring voxels, as JSON. In Unity (and in the AVBD package) each voxel is a box body and each joint
is a ball socket joint that links two neighboring boxes.

The only dependency is `numpy`. Nothing here needs Unity.

```
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --resolution 64 --zup-to-yup
python3 Tools/Voxelize/voxelize_stl.py model.obj -o model.voxels.json --voxel-size 0.1
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1 --smooth
```

## Options

| Option | Meaning |
|---|---|
| `mesh` | The mesh file. A name that ends with `.obj` (any case) is read as a Wavefront OBJ, anything else as an STL (binary or ASCII, detected from the file). |
| `-o`, `--output` | Output file. Default: the mesh file's name with `.voxels.json`. |
| `--voxel-size S` | Grid size as the edge length of a voxel, in output units (mesh units times `--scale`). |
| `--resolution N` | Grid size as the number of voxels along the longest side of the mesh. Exactly one of `--voxel-size` and `--resolution` is required. |
| `--surface center` (default) | A voxel is solid if its **center** is inside the mesh, that is inside at least one of its shells (see *How the fill works*). The voxel volume is close to the mesh volume. |
| `--surface touch` | A voxel is solid if it touches or is inside the mesh. Conservative: the result contains the mesh. |
| `--connectivity 6` (default) | Joint every voxel to its 6 face neighbors. `18` adds the 12 edge neighbors, `26` adds the 8 corner neighbors. |
| `--scale F` | Multiplies the mesh first, for example `0.001` for a mesh in millimeters. |
| `--zup-to-yup` | Rotates the mesh from a z-up frame (most CAD tools) to Unity's y-up frame: `(x, y, z) -> (x, z, -y)`. |
| `--smooth` | Also bake the smooth surface of the mesh into the voxels on the surface (see *Smooth surface* below). Larger files. |
| `--crease D` | With `--smooth`: the angle in degrees up to which the normals of neighboring triangles are smoothed (default 50); a sharper edge keeps its crease. |
| `--max-cells N` | Refuses a grid with more cells than this (default 200 million) instead of running out of memory. |

The grid starts at the minimum corner of the (scaled, rotated) mesh. The script prints the voxel count, the joint count
and the voxel volume next to the mesh volume, which is a quick check of the resolution.

## OBJ files

Only the vertices (`v`) and the faces (`f`) of an OBJ are used. A face corner can be `v`, `v/vt`, `v//vn` or `v/vt/vn`, with 1-based or negative (relative) indices, and a line can continue on the next one with a trailing `\`. A face with more than three corners is split into a fan of triangles: that is right for triangles and convex quads (the usual case) and wrong for a concave polygon. Normals, texture coordinates, materials, groups and objects are ignored, so all objects of a file become one mesh. The same rules as for an STL apply to the mesh: it should be closed, and an OBJ is often y-up already, so `--zup-to-yup` is only for files from a z-up tool. A bad file stops the script with the line number.

## How the fill works

1. Every voxel that a triangle touches is marked (triangle against box overlap test).
2. The space outside the mesh is found by a flood fill from a corner of the padded grid.
3. Everything that is not outside is solid. This fills the interior of a surface-only mesh, and also **closed cavities**
   inside the mesh (there is no way to get in, so they are treated as solid).
4. With `--surface center`, voxels on the boundary whose center is outside the mesh are removed again. Whether a center
   is inside is decided by counting ray crossings, which does not depend on the winding (orientation) of the triangles.
   The crossings are counted **for every shell on its own**, and a center is inside if it is inside any shell. A shell
   is a set of triangles that are connected through shared vertices (vertices at the same place, to a ten-millionth of the
   size of the mesh); a mesh built from several pieces has one shell per piece. Pieces that are put partly inside each
   other therefore make up the *union* of the pieces: counted together, the crossings in the overlap would cancel each
   other and the voxels there, next to the outside, would be missing.

## Output format

JSON, `"format": "mckrill-avbd-voxels"`, `"version": 1`. The arrays are flat arrays of integers so that Unity's
`JsonUtility` can read them.

```json
{
  "format": "mckrill-avbd-voxels",
  "version": 1,
  "source": "model.stl",
  "voxelSize": 0.1,
  "origin": [0.0, 0.0, 0.0],
  "gridDimensions": [10, 4, 6],
  "connectivity": 6,
  "surface": "center",
  "voxelCount": 3,
  "jointCount": 2,
  "voxels": [0,0,0, 1,0,0, 2,0,0],
  "joints": [0,1, 1,2]
}
```

| Field | Meaning |
|---|---|
| `voxelSize` | Edge length of a voxel (and of its box body). |
| `origin` | Minimum corner of voxel `(0,0,0)`, in the output frame. |
| `gridDimensions` | Number of cells along x, y, z. |
| `voxels` | `3 * voxelCount` integers: the `(i, j, k)` index of each solid voxel, ordered by `i`, then `j`, then `k`. The position of a triple in this list (0, 1, 2, ...) is the **voxel number**. |
| `joints` | `2 * jointCount` integers: pairs of voxel numbers. Each pair of neighbors appears once. |

Voxel `n` with index `(i, j, k)` is a box of size `voxelSize` on all three sides, centered at

```
origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize
```

The anchor of a joint is the midpoint of the two voxel centers: for face neighbors the center of the shared face, for
edge neighbors the middle of the shared edge, for corner neighbors the shared corner.

With `--smooth` the file also has the arrays of the chunks (see below). They are optional and only added, so the format stays version 1 and a reader that does not know them ignores them.

### Reading it in Unity

```csharp
[System.Serializable]
class VoxelData
{
    public string format;
    public int version;
    public float voxelSize;
    public float[] origin;
    public int[] gridDimensions;
    public int connectivity;
    public int voxelCount;
    public int jointCount;
    public int[] voxels;
    public int[] joints;
}

var data = JsonUtility.FromJson<VoxelData>(json);
for (int n = 0; n < data.voxelCount; n++)
{
    var center = new Vector3(
        data.origin[0] + (data.voxels[3 * n] + 0.5f) * data.voxelSize,
        data.origin[1] + (data.voxels[3 * n + 1] + 0.5f) * data.voxelSize,
        data.origin[2] + (data.voxels[3 * n + 2] + 0.5f) * data.voxelSize);
    // spawn a box body of size data.voxelSize at center
}
for (int m = 0; m < data.jointCount; m++)
{
    int a = data.joints[2 * m], b = data.joints[2 * m + 1];
    // ball socket joint between body a and body b, anchored at the midpoint of their centers
}
```

The package's *JSON Mesh Import Demo* sample does this: it reads a list of voxel files, spawns the boxes and joints on the GPU solver and lets you shoot or blow them up (`Packages/com.mckrill.avbd/Samples~/JsonMeshImportDemo`, which also has a ready-made `VoxelMeshData` class that reads and checks the file).

## Smooth surface (`--smooth`)

The voxels are boxes, so an object made of them looks like a pile of cubes. With `--smooth` the script also bakes, for each voxel on the surface, a **chunk**: the piece of the original mesh's surface that lies in that voxel's cell, as a closed triangle mesh. A renderer that draws every surface block as its chunk (rigidly bound to the block) shows the smooth surface of the STL while the object is intact, and a block that breaks off is still a block: a curved outside and flat faces where it was cut from its neighbors. Voxels that are not on the surface get no chunk and stay plain cubes. The physics is not affected: the bodies are still the boxes.

How a chunk is made (`smooth_chunks.py`):

1. The triangles are clipped (exactly, Sutherland-Hodgman) against the cell boxes they cross. The corner normals are smoothed over the neighbors of a corner (angle weighted, up to `--crease` degrees) before the clipping, so the normals are continuous across the cells.
2. The triangles of closed cavities (the script fills them) are left out: they cannot be seen.
3. Mesh in a cell that has no voxel (the surface bulges past the voxels' edge, with `--surface center`) goes to the nearest voxel around it, so it is not lost. That piece can lie up to one cell away from its block.
4. On every face of a cell that borders a cell of **another chunk** that is solid there, the chunk gets a flat cap: the cross-section of the mesh with that face, triangulated. Both neighbors get a cap, so each chunk is closed and, alone, a solid piece.

All chunks of a file are stored in a few flat arrays:

| Field | Meaning |
|---|---|
| `chunkCount` | Number of chunks. |
| `chunkVoxels` | `chunkCount` voxel numbers, increasing: chunk `c` belongs to voxel `chunkVoxels[c]`. |
| `chunkVertexStart`, `chunkTriangleStart` | `chunkCount + 1` integers each, starting at 0: chunk `c` has the vertices `chunkVertexStart[c]` to `chunkVertexStart[c + 1] - 1` and the triangles `chunkTriangleStart[c]` to `chunkTriangleStart[c + 1] - 1`. |
| `chunkPositions`, `chunkNormals` | Three floats for each vertex: the position **relative to the center of the voxel**, in the units of the file (the pose of the block moves it, so the positions are valid for any block pose), and the unit normal. Positions have 5 decimals and normals 3. |
| `chunkTriangles` | Three integers for each triangle, counted from the first vertex **of the chunk**. Counterclockwise seen from outside, so a left-handed engine such as Unity sees them clockwise from outside, which is its front face. |

Numbers from the shipped samples (voxel size 0.25 and 0.2): about 40 to 46 triangles per chunk (60% of them caps), 250 to 870 KB for the whole file. The triangle count follows the resolution of the STL: a finely tessellated mesh gives many tiny triangles in each chunk, so decimate the mesh first if the file is too large.

Limits of the method: the mesh must be closed (the caps need a reliable inside test); a part thinner than a cell can be left without its own chunk; the sum of the chunk volumes equals the mesh volume only up to about 0.1% (the script nudges the mesh by a tiny amount so that no triangle lies exactly on a cell face).

### The sample meshes

`make_sample_meshes.py` writes the STL meshes of the shipped samples (`Sphere.stl`, `Torus.stl` and `Cube.stl`, with a cavity). The files in `Packages/com.mckrill.avbd/Samples~/JsonMeshImportDemo/Resources/JsonMeshImportDemo` were made with:

```
python3 Tools/Voxelize/make_sample_meshes.py some/folder
D=Packages/com.mckrill.avbd/Samples~/JsonMeshImportDemo/Resources/JsonMeshImportDemo
python3 Tools/Voxelize/voxelize_stl.py some/folder/Sphere.stl -o $D/Sphere.json --voxel-size 0.25 --zup-to-yup --smooth
python3 Tools/Voxelize/voxelize_stl.py some/folder/Torus.stl  -o $D/Torus.json  --voxel-size 0.2  --zup-to-yup --smooth
python3 Tools/Voxelize/voxelize_stl.py some/folder/Cube.stl   -o $D/Cube.json   --voxel-size 0.25 --smooth
```

The tests check that the voxels, joints and chunks of the shipped files are what these commands give.

## Limits

- The mesh should be **closed** (watertight). For an open mesh the flood fill can leak into the inside and leave it
  empty; the script warns when it finds open edges.
- Pieces that share vertices with each other (two boxes that touch in a corner or along an edge, with the vertices at the same place) are one shell, so
  the overlap rule does not apply between them. `--smooth` decides the cut of its chunks with a ray count over the whole mesh, so a mesh whose pieces overlap
  can still get chunks that disagree with the voxels in the overlap.
- Features thinner than about one voxel can vanish with `--surface center`; use `--surface touch` or a smaller voxel.
- The grid has to fit in memory: `--max-cells` stops runs that would not.
- Large voxel counts mean many bodies and joints. At `--connectivity 6` there are about 3 joints per voxel.

## Tests

```
python3 -I Tools/Voxelize/test_voxelize_stl.py
```
