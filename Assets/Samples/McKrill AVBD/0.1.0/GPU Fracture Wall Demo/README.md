# GPU Fracture Wall Demo

A brick wall of 225 or more blocks on the GPU solver. The blocks are held together by joints that break under load. Click to shoot a box from the camera into the wall.

The controls are the same as in the *GPU Pyramid Demo*, and the panel has the wall's properties on top of the shot's.

The demo uses only the package's public API: `AvbdBodyBuffers` holds the bodies, `AvbdConstraintBuffers` holds the joints and has room for contacts, and `AvbdSolver.Step()` advances them. It contains no collision or fracture code.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *GPU Fracture Wall Demo*. (Package Manager copies the sample into `Assets/Samples`. After a package update, import it again to get the new version.)
2. Open the imported scene `FractureWall` and press **Play**, or add the *GPU Fracture Wall Demo* component to any scene (**GameObject > McKrill AVBD > GPU Fracture Wall Demo**).

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support.

## Controls

| Input | Action |
| --- | --- |
| Left click (outside the panel) | Shoot a box in the direction the camera looks |
| Right drag | Look around |
| `W` `A` `S` `D`, `Q` `E` | Move, down and up. `Shift` is three times faster |
| Mouse wheel | Move forward and back |
| `Space`, `N` | Pause / play, single step |
| `R` | Rebuild the wall with the panel's settings and remove the shots |

## The panel

The shot's speed, size and mass, the solver's iterations and the buttons are as in the pyramid demo. The wall's settings take effect on **Reset**, because the package cannot change a joint after it was added.

| Control | What it does |
| --- | --- |
| Columns, Rows | 15 to 22 columns and 15 to 18 rows: 225 (the default, 15 x 15) to 396 blocks |
| Block density | 0.1 to 10 (log scale). A block is 1 x 0.5 x 0.5 m, so density 1 is 0.25 kg. Does not apply to the bottom row, which is immobile |
| Rigid joints | Joints with infinite stiffness. Off: the joints' linear part is a spring with the stiffness below |
| Joint stiffness | 1e2 to 1e8 (log scale), used when the joints are not rigid |
| Fracture threshold | A joint breaks when the norm of its angular multiplier is above this (1 to 1e4, log scale) |

The panel also shows how many joints have broken, out of all.

## The wall

- **Walls.** The *Walls* slider (1 to 10, default 1) builds several identical walls one behind the other along z, 4 m apart (the first stands in front, nearest the starting camera). Like the other wall settings it takes effect on **Reset**. The block, joint and body counts grow with the number of walls, so a scene of many walls is a bigger scene for the solver.
- Blocks are 1 x 0.5 x 0.5 m in a running bond: odd rows are shifted by half a block, so every block rests on two blocks of the row under it. The wall stands on a static ground slab whose top is the plane y = 0. **The bottom row is immobile** (static bodies, no mass); the blocks of the row above are joined to it, and its blocks have no joints to each other.
- Every block has a ball-socket joint to the block on its left and to the two blocks under it, at the middle of the face they share. Each joint has a **rigid angle lock**. The fracture test looks at the angular multiplier, and only a rigid angle lock stores a multiplier, so the angle lock is not an option in the panel.
- A joint breaks in the step its angular multiplier exceeds the threshold. The next step removes it from the solver's lists, and from the step after that its two blocks collide with each other again, like any other pair (the reference solver's timing). Blocks that were joined do not collide while the joint holds.
- **The threshold has no calibrated unit.** The multiplier depends on the joint's penalty and on the constraint's torque arm, so the number to enter is found by trying: if nothing breaks, lower it or shoot a heavier, faster box; if the wall breaks apart by itself, raise it. The default is 100.

## Limits

- The wall is limited to 396 blocks because that is the largest size that was tried. The constraint lists are built on the GPU (stage G) and nothing is read back for them; the graph coloring still reads two small things back every step (the number of colors), which is the remaining stall.
- The "joints broken" counter is a counter on the GPU that the panel reads once per step.
- **Drawing is instanced and stays on the GPU.** One `Graphics.RenderMeshPrimitives` call draws every box with the package's `AvbdInstancedBoxes` shader, which reads each box's pose from the solver's body state buffer: no pose readback and no GameObject per body (the pyramid demo still does both). The shader is unlit-style, with one directional light (the scene's sun if there is one) and ambient, so it works in the built-in pipeline and in URP, but **it has no shadows**. A player build needs the shader included, which it is: the package's shader set references it.
