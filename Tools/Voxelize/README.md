# STL and OBJ voxelizer

`voxelize_stl.py` turns a triangle mesh (an STL or OBJ file) into a **solid** set of voxels and writes them, together with the
joints between neighboring voxels, as JSON. In Unity (and in the AVBD package) each voxel is a box body and each joint
is a ball socket joint that links two neighboring boxes.

The only dependency is `numpy`. Nothing here needs Unity.

```
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --resolution 64 --zup-to-yup
python3 Tools/Voxelize/voxelize_stl.py model.obj -o model.voxels.json --voxel-size 0.1
```

## Options

| Option | Meaning |
|---|---|
| `mesh` | The mesh file. A name that ends with `.obj` (any case) is read as a Wavefront OBJ, anything else as an STL (binary or ASCII, detected from the file). |
| `-o`, `--output` | Output file. Default: the mesh file's name with `.voxels.json`. |
| `--voxel-size S` | Grid size as the edge length of a voxel, in output units (mesh units times `--scale`). |
| `--resolution N` | Grid size as the number of voxels along the longest side of the mesh. Exactly one of `--voxel-size` and `--resolution` is required. |
| `--surface center` (default) | A voxel is solid if its **center** is inside the mesh. The voxel volume is close to the mesh volume. |
| `--surface touch` | A voxel is solid if it touches or is inside the mesh. Conservative: the result contains the mesh. |
| `--connectivity 6` (default) | Joint every voxel to its 6 face neighbors. `18` adds the 12 edge neighbors, `26` adds the 8 corner neighbors. |
| `--scale F` | Multiplies the mesh first, for example `0.001` for a mesh in millimeters. |
| `--zup-to-yup` | Rotates the mesh from a z-up frame (most CAD tools) to Unity's y-up frame: `(x, y, z) -> (x, z, -y)`. |
| `--no-colors` | Does not write the colors of the surface voxels, even if the mesh has colors (see Colors). |
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
| `colorCount`, `colorVoxels`, `colors` | Only if the mesh has colors: `colorVoxels` are the numbers of the voxels on the surface that have a color (ascending), `colors` their colors as `3 * colorCount` integers 0 to 255 (red, green, blue, sRGB). Voxels that are not listed have no color. The three fields are absent for a mesh without colors, and the version stays 1: a reader that does not know them ignores them. |

Voxel `n` with index `(i, j, k)` is a box of size `voxelSize` on all three sides, centered at

```
origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize
```

The anchor of a joint is the midpoint of the two voxel centers: for face neighbors the center of the shared face, for
edge neighbors the middle of the shared edge, for corner neighbors the shared corner.

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

The package's *JSON Mesh Import Demo* sample does this (and also reads the colors, see Colors below): it reads a list of voxel files, spawns the boxes and joints on the GPU solver and lets you shoot or blow them up (`Packages/com.mckrill.avbd/Samples~/JsonMeshImportDemo`, which also has a ready-made `VoxelMeshData` class that reads and checks the file).

## Colors

An OBJ can carry colors, and the voxelizer passes them on to the voxels **on the surface**, that is the voxels with a face that no other voxel shares. The voxels below the surface get no color.

For a surface voxel the script finds the point of the mesh that is closest to the center of the voxel (among the triangles that touch the voxel or one of its six face neighbors) and takes the color of the mesh at that point. It is one sample for each voxel, not an average over the voxel, so a texture with detail finer than a voxel is sampled, not filtered. The color of a triangle comes from, in this order:

1. its **texture**: the `map_Kd` image of the material (from the `mtllib` file) at the texture coordinates (`vt`) of the face, with bilinear filtering and repeat. Reading an image needs the [Pillow](https://pypi.org/project/pillow/) package (`pip install pillow`); without it the script warns and uses the next source. The only other dependency stays numpy.
2. its **vertex colors**: `v x y z r g b` (the common extension of MeshLab and others), interpolated between the corners. The values are 0 to 1, or 0 to 255 if any value in the file is above 1.
3. the diffuse color `Kd` of its material (`usemtl`).

A surface voxel whose closest triangle has none of them has no color (a part of the mesh without a material, for example). A missing `.mtl` or texture file is reported as a warning, and the material's `Kd` is used. STL files have no colors, so the output of an STL has none. Colors are written as they are in the file (sRGB). Colors do not change which voxels exist, or the joints.

The *JSON Mesh Import Demo* sample applies them (`Packages/com.mckrill.avbd/Samples~/JsonMeshImportDemo`).

## Limits

- The mesh should be **closed** (watertight). For an open mesh the flood fill can leak into the inside and leave it
  empty; the script warns when it finds open edges.
- Features thinner than about one voxel can vanish with `--surface center`; use `--surface touch` or a smaller voxel.
- The grid has to fit in memory: `--max-cells` stops runs that would not.
- Large voxel counts mean many bodies and joints. At `--connectivity 6` there are about 3 joints per voxel.

## Tests

```
python3 -I Tools/Voxelize/test_voxelize_stl.py
```
