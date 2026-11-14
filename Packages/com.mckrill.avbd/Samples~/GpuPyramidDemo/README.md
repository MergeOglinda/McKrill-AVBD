# GPU Pyramid Demo

A pyramid of boxes on the GPU solver. Click to shoot a box from the camera into it.

The demo uses only the package's public API: `AvbdBodyBuffers` holds the bodies and `AvbdSolver.Step()` advances them.
It contains no collision code, because collisions (broadphase, narrowphase, contact solve) belong inside `AvbdSolver.Step()`.

**Until those stages are part of `Step()`, the boxes fall through the ground and through each other.** The demo is written
for the solver with contacts; with the stages that exist today you can check the controls, the UI and the free fall, not the stack.
For a pyramid that stands today, use the *Reference Solver Demo* (CPU reference solver, scene *Pyramid*).

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *GPU Pyramid Demo*.
2. Open the imported scene `PyramidShooter` and press **Play**. The scene holds one object with the *GPU Pyramid Demo* component;
   the component creates a camera and a light if the scene has none.
   Or add it to any scene with **GameObject > McKrill AVBD > GPU Pyramid Demo**.

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support.
Input is read through IMGUI events, so it works with the Input Manager and with the Input System package.

## Controls

| Input | Action |
| --- | --- |
| Left click (outside the panel) | Shoot a box in the direction the camera looks. The crosshair marks that direction |
| Right drag | Look around |
| `W` `A` `S` `D`, `Q` `E` | Move, down and up. `Shift` is three times faster |
| Mouse wheel | Move forward and back |
| `Space`, `N` | Pause / play, single step |
| `R` | Reset the pyramid and remove the shots |

## The panel

| Control | What it does |
| --- | --- |
| Speed | Speed of the next shot, 5 to 80 m/s |
| Size | Edge length of the next shot, a cube, 0.2 to 2 m |
| Mass | Mass of the next shot, 0.1 to 200 kg on a log scale. The density follows from mass and size |
| Iterations | Solver iterations per step (1 to 30; default 10, upstream's value) |
| Pyramid layers | 7 to 12 layers, which is 140 to 650 blocks. Takes effect on *Reset* |
| Pause, Step, Clear shots, Reset | As named. *Clear shots* removes every shot and leaves the pyramid |

A warning appears when a shot travels farther than its own size in one step (speed times 1/60 s above the size). Collision detection
looks at the state at the end of each step, so such a shot can pass through thin boxes.

## The scene

- **Ground**: a static slab 200 m by 200 m whose top is the plane y = 0.
- **Pyramid**: layer k is a square grid of `layers - k` blocks per side, centered over the layer below, so each block rests on four.
  Blocks are 1 x 0.5 x 1 m with density 1 (0.5 kg) and 0.01 m gaps. Seven layers are 7^2 + 6^2 + ... + 1 = 140 blocks.
- **Shots**: up to 100 are alive at once; shooting more removes the oldest.
- Gravity is (0, -10, 0). The scene is y-up (the package default), so solver positions are Unity world positions.
- The shot masses are in the scene's units: a block weighs 0.5 kg, so a 5 kg shot is ten blocks heavy.

## How it draws

Every frame the demo reads the pose of every body back from the GPU (one `GetData` of the used slots) and sets one cube per body.
That is a stall and one GameObject per body: fine for several hundred boxes, and not how the package will render. The demo also reads
the public `AvbdBodyBuffers.State` buffer with its own copy of the state record and checks the record size at start, so a change of
that layout fails with a message instead of drawing garbage.

Each shot is a spawn and the oldest is a remove. A spawn or remove changes the set of solved bodies, so the next step rebuilds the
constraint lists and the graph coloring (a readback). That is cheap at this size.
