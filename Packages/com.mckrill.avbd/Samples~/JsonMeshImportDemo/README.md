# JSON Mesh Import Demo

Spawns the objects of every voxel file in a folder on the GPU solver. A voxel file is the JSON that `Tools/Voxelize/voxelize_stl.py` writes from an STL mesh (see `Tools/Voxelize/README.md` for the script and the file format). Every voxel becomes a box body and every joint of the file becomes a ball-socket joint with a hard angle lock between two neighboring boxes, at the middle between their centers. The joints fracture like those of the *GPU Fracture Wall Demo*, which this demo is based on: shoot boxes at the objects, or blow them up.

The objects stand on the ground in a row along x, **5 m apart** (center to center), the row centered on x = 0. Each object is placed so that the middle of its bounding box is on its spot in x and z and its lowest voxel rests on the ground.

The demo uses only the package's public API (`AvbdBodyBuffers`, `AvbdConstraintBuffers`, `AvbdSolver.Step()`), and draws every box with one instanced draw call, as the fracture wall demo does. It contains no collision or fracture code. `VoxelMeshData.cs` reads and checks the JSON.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *JSON Mesh Import Demo*.
2. Open the imported scene `JsonMeshImport` and press **Play**, or add the *JSON Mesh Import Demo* component to any scene (**GameObject > McKrill AVBD > JSON Mesh Import Demo**).

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support.

## The files

There is no list of files. The demo spawns **every file in the `Resources/JsonMeshImportDemo` folder**, one object each, in the order of the file names (ordinal, so `Cube`, `Sphere`, `Torus`). In the imported sample that is `Assets/Samples/McKrill AVBD/<version>/JSON Mesh Import Demo/Resources/JsonMeshImportDemo`; any `Resources/JsonMeshImportDemo` folder in the project counts, and all of them are read together. The folder ships with three files: a sphere, a torus lying flat and a cube. The cube was made from an STL that is hollow (a surface for the outside and another, facing inward, for a cavity), to show that the script fills the inside.

To use your own mesh, run the script and put its output (a `.json` file) in that folder:

```
python3 Tools/Voxelize/voxelize_stl.py engine.stl -o engine.voxels.json --voxel-size 0.25 --zup-to-yup
```

To leave an object out, remove its file from the folder (or move it out of any `Resources` folder). There is nothing to edit in the code.

Every text asset in the folder is read as a voxel file, so keep other text files out of it. A file that cannot be read (not a voxel file, wrong version, inconsistent arrays) is reported in the console and in the panel and is left out; the other files are still spawned. If the folder has no files, the console says so and the scene has only the ground. **Reset** reads the folder again, so a file you changed or added while playing is picked up with `R` (in the editor the asset must be imported first, which happens when the editor window regains focus). Objects are placed 5 m apart along x, centered on x = 0.

An object wider than 5 m in x or z overlaps its neighbor: the demo warns in the console, and the two objects then collide.

## Controls

The same as in the *GPU Fracture Wall Demo*, except for the volley (there is no wall to aim it at).

| Input | Action |
| --- | --- |
| Left click (outside the panel) | Shoot a box in the direction the camera looks, or, in explosion mode, set off an explosion where the camera ray hits |
| Right drag | Look around |
| `W` `A` `S` `D`, `Q` `E` | Move, down and up. `Shift` is three times faster |
| Mouse wheel | Move forward and back |
| `Space`, `N` | Pause / play, single step |
| `R` | Read the files again, rebuild the objects with the panel's settings and remove the shots |

## The panel

The click mode button and its shot and explosion settings, the solver's iterations and the joint settings are as in the fracture wall demo. The object and joint settings take effect on **Reset**, because the package cannot change a joint after it was added.

| Control | What it does |
| --- | --- |
| Objects | One line for each file of the `Resources/JsonMeshImportDemo` folder: the number of voxels and joints and the size of the object, or why it was not loaded |
| Voxel mass | 0.01 to 10 kg (log scale), default 0.25 kg like a block of the fracture wall. The mass of every voxel; the density of an object follows from it and the size of its voxels |
| Rigid joints, Joint stiffness, Fracture threshold | As in the fracture wall demo (defaults: stiffness 5e3, threshold 40). **They are the wall's numbers, not tuned for voxels**: the multiplier a joint sees depends on the voxel size and mass, so if nothing breaks lower the threshold or shoot harder, and if an object falls apart by itself raise it |
| Plastic yield, Yield threshold, Bend break, Bend limit | As in the fracture wall demo |

## Voxels and joints

- A voxel is a box of the file's `voxelSize` less a gap of 1 cm (at most 5% of the voxel size): neighbors start a little apart, so that voxels that touch only along an edge or at a corner (they are not joined at the default connectivity) do not start inside each other. Joint anchors are the middle between the two voxel centers, as in the file format.
- A joint holds two voxels together: while it holds they do not collide with each other. Voxels that are not joined collide like any other pair. At the default connectivity of 6 only face neighbors are joined, so an object is as stiff as a lattice of ball-socket joints with an angle lock can make it; `--connectivity 18` or `26` in the script gives it more joints per voxel.
- The scene is y-up. A mesh from a z-up tool needs `--zup-to-yup` in the script, or it lies on its side.

## Limits

- Every voxel is a body: the files that ship have 500 to 900 voxels each, and a finer grid grows with the cube of the resolution. The package's limits on bodies, joints and contact manifolds apply (the demo sizes its buffers from the files).
- The shot size slider goes up to 2 m and shots are cubes.
- Drawing is instanced and has **no shadows**, as in the fracture wall demo.
- Written without a Unity editor: not compiled, not run.
