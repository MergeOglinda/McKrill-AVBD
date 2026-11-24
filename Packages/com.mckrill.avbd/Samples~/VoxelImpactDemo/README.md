# Voxel Impact Demo

Six hollow objects made of voxels fly at one point from a ring around it and collide there all at once, on the GPU solver. It is the scene of the performance test `Scenario6_VoxelImpact35k` (`Tests/Performance/VoxelImpactScene.cs`): the same bodies, joints, capacities and solver settings, so what you see is what the test times.

- **Objects:** two cubes (4.8 m), two spheres (radius 2.4 m) and two toruses (tube radius 1.4 m around a circle of radius 2.2 m, standing upright with its axis along z), in the order cube, sphere, torus, sphere, torus, cube around the ring. Each is a **skin two voxels thick** (0.4 m), made of 0.2 m voxels: the shapes are generated (`VoxelShape.cs`), so there are no voxel files. With a 2-voxel skin the scene has **34 880 voxels** and **241 872 joints** (cube 5 824 / 38 040, sphere 4 288 / 30 564, torus 7 328 / 52 332, two of each).
- **Impact:** the middle of every object starts 11 m from a meeting point 8 m above the ground, on a ring (60 degrees apart), and flies at the point at 25 m/s. The three pairs that face each other close in at **50 m/s**; neighbors on the ring close in at 25 m/s and touch first (the first voxels touch after about 13 steps, all six objects by step 22).
- **Voxels and joints** are those of the *JSON Mesh Import Demo*: every voxel is a box (0.19 m, 0.25 kg), and every pair of voxels that share a face or an edge is linked by a ball-socket joint with a rigid angle lock, at the middle between their centers (about 6.9 joints per voxel). The joint settings are the JSON demo's defaults: rigid joints, fracture 600, plastic yield 400, bend break at 30 degrees, lowest penalty 2000. The differences are the ones the test asks for: the **linear fracture (600) and the linear yield (400, with the 4 cm stretch break) are on** and the solver runs **25 iterations** per step.

The demo uses only the package's public API (`AvbdBodyBuffers`, `AvbdConstraintBuffers`, `AvbdSolver.Step()`) and draws every box with one instanced draw call, as the *GPU Brick Ring Demo* does. It contains no collision or fracture code.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *Voxel Impact Demo*.
2. Open the imported scene `VoxelImpact` and press **Play**, or add the *Voxel Impact Demo* component to any scene (**GameObject > McKrill AVBD > Voxel Impact Demo**).

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support. Building the scene (the shapes, the batch spawn and 242 000 joints) takes a moment on the CPU; the panel shows the time.

## Controls

| Input | Action |
| --- | --- |
| Left click (outside the panel) | Shoot a box in the direction the camera looks |
| Right drag | Look around |
| `W` `A` `S` `D`, `Q` `E` | Move, down and up. `Shift` is three times faster |
| Mouse wheel | Move forward and back |
| `Space`, `N` | Pause / play, single step |
| `R` | Rebuild the scene with the panel's settings and remove the shots |

## The panel

| Control | What it does |
| --- | --- |
| Shot: speed, size, mass | The box that a click fires (a voxel weighs 0.25 kg) |
| Iterations | Solver iterations per step, 1 to 30 (25 in the test). Takes effect at the next step |
| Fuse the dual update | `AvbdSolver.FuseDualUpdate`: the dual update of the joints runs inside the next primal sweep. The same result with fewer dispatches; switch it to compare frame times |
| Speed of each object | 0 to 60 m/s, default 25 (twice that head-on). **Reset to apply** |
| Skin | 1 to 4 voxels (0.2 to 0.8 m). 2 is the test's scene (34 880 voxels, 241 872 joints); 1 has 19 264 voxels and 92 592 joints, 3 has 46 304 and 352 864, 4 has 53 760 and 428 528. **Reset to apply** |
| Plastic yield, Linear fracture, Linear yield | The joint options of the test. The thresholds are serialized fields of the component (**Reset to apply**) |
| Play / Pause, Step, Clear shots, Reset | |
| Count broken | Reads the number of broken joints back from the GPU. It waits for the GPU, so it is a button and not a live number |

The panel also shows the number of voxels and joints, the step number and the simulated time, the build time, and the CPU-side step and frame times.

## Notes

- At 50 m/s a pair of objects closes in by 0.83 m per step, a little more than the two skins are thick together (0.8 m). A deep first touch can push voxels through the other object's skin, so a few voxels get through. Lower the speed, thicken the skin, or raise the iterations to see fewer. A shot at 80 m/s moves 1.3 m per step and passes through thin skins in the same way (the panel warns when a shot moves farther than its size per step).
- The objects are not rotated, so a joint's local anchors are the offsets from the voxel centers to the middle between them, as in the JSON demo.
- The `VoxelShape.cs` of the demo and of the performance test are the same file (apart from the namespace).
- Written without access to a Unity editor or a GPU: not compiled, not run, not measured.
