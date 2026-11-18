# GPU Brick Ring Demo

The brick ring of three-avbd (paper Fig. 1) on the GPU solver: 40 courses of bricks laid tangentially in running bond on rings from a radius of 80 m outwards, 10 bricks deep at the bottom and one fewer every 4 courses, so the outside is stepped and the inside sheer. With the defaults that is **110 440 bricks**, 20 m high and 160 m across. The bricks rest exactly on each other, so the ring stands from the first step. An 8 m cube (the volume and density of the original's 8 m ball) starts just outside the stepped face and flies in at 30 m/s. Click to shoot boxes from the camera as well.

This is the scene of the performance test `Scenario5_BrickRing110k` (`Tests/Performance/BrickRingScene.cs`): the same bodies, the same capacities and 4 solver iterations, so what you see is what the test times. The world is y-up and the ball is a cube, because the package has no sphere shape (so it slides instead of rolling).

The demo uses only the package's public API: `AvbdBodyBuffers` holds the bodies, `AvbdConstraintBuffers` has room for contacts (and no joints), and `AvbdSolver.Step()` advances them. It contains no collision code. Drawing is as in the *GPU Fracture Wall Demo*: one instanced draw call, with the box shader reading the poses from the solver's body buffer.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *GPU Brick Ring Demo*.
2. Open the imported scene `BrickRing` and press **Play**, or add the *GPU Brick Ring Demo* component to any scene (**GameObject > McKrill AVBD > GPU Brick Ring Demo**).

## Controls

The same as in the *GPU Fracture Wall Demo*: left click shoots, right drag looks, `W` `A` `S` `D` `Q` `E` move (`Shift` is three times faster), the mouse wheel moves forward and back, `Space` pauses, `N` steps once, `R` resets. The camera starts outside the stepped face, to the side of the ball's path, and moves faster than in the wall demo because the scene is 160 m across.

## The panel

| Control | What it does |
| --- | --- |
| Shot speed, size, mass | As in the wall demo; the mass goes up to 2000 kg because the ring is heavy |
| Iterations | Solver iterations per step, 1 to 30 (4, as in the performance test) |
| C, D, R | Courses (4 to 40), bricks deep at the bottom (1 to 10) and the inner radius (20 to 100 m); the number of bricks after **Reset** is shown under them. Smaller rings need much less GPU memory |
| Z-curve body order | Off (the default): the bodies are in the GPU arrays in the order the ring is built, course by course. On: in Z-curve order of their positions (`AvbdBodyBuffers.SpawnBatch`), so that bricks that touch are near each other in memory. Takes effect on **Reset**. Use it to compare frame times |
| Ball, Ball speed | Whether the cube is there, and its speed (0 to 80 m/s). Takes effect on **Reset** |

The panel shows the number of bodies, the CPU time of the last build (which includes sorting the batch when Z-curve order is on), and the step time and frame time. The step time is the time the CPU spent in `Step()`, not the GPU time: the frame time is the better number. At full size the scene takes much more than 16 ms per step on most GPUs, so the demo runs one step per frame and says when it is slower than real time.

## Limits

- **Memory.** With the defaults the contact pool is the largest the package allows (500 000 manifold slots), which takes a few GB. Lower C, D or R on a smaller GPU.
- **The pool is too small for this ring.** It has more touching pairs than the pool holds, and a pair that finds no slot has no contacts: some bricks may then not rest on each other as they should. The performance test reports the overflow (`Manifold overflow`); the demo has no readback of it.
- The panel's step time is CPU-side. For GPU time use a GPU profiler or the performance test.
