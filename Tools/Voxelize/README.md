# STL voxelizer

`voxelize_stl.py` turns a triangle mesh (an STL file) into a **solid** set of voxels and writes them, together with the
joints between neighboring voxels, as JSON. In Unity (and in the AVBD package) each voxel is a box body and each joint
is a ball socket joint that links two neighboring boxes.

The only dependency is `numpy`. Nothing here needs Unity.

```
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --voxel-size 0.1
python3 Tools/Voxelize/voxelize_stl.py model.stl -o model.voxels.json --resolution 64 --zup-to-yup
```

## Options

| Option | Meaning |
|---|---|
| `stl` | Binary or ASCII STL (detected from the file). |
| `-o`, `--output` | Output file. Default: the STL's name with `.voxels.json`. |
| `--voxel-size S` | Grid size as the edge length of a voxel, in output units (STL units times `--scale`). |
| `--resolution N` | Grid size as the number of voxels along the longest side of the mesh. Exactly one of `--voxel-size` and `--resolution` is required. |
| `--surface center` (default) | A voxel is solid if its **center** is inside the mesh. The voxel volume is close to the mesh volume. |
| `--surface touch` | A voxel is solid if it touches or is inside the mesh. Conservative: the result contains the mesh. |
| `--connectivity 6` (default) | Joint every voxel to its 6 face neighbors. `18` adds the 12 edge neighbors, `26` adds the 8 corner neighbors. |
| `--scale F` | Multiplies the mesh first, for example `0.001` for an STL in millimeters. |
| `--zup-to-yup` | Rotates the mesh from a z-up frame (most CAD tools) to Unity's y-up frame: `(x, y, z) -> (x, z, -y)`. |
| `--max-cells N` | Refuses a grid with more cells than this (default 200 million) instead of running out of memory. |

The grid starts at the minimum corner of the (scaled, rotated) mesh. The script prints the voxel count, the joint count
and the voxel volume next to the mesh volume, which is a quick check of the resolution.

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

A Unity importer that does this for the AVBD package is not part of this script.

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
