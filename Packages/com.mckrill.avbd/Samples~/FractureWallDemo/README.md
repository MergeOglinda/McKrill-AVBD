# GPU Fracture Wall Demo

A brick wall (75 x 75 blocks by default, in ten walls) on the GPU solver. The blocks are held together by joints that break under load, or that can bend and then break (plastic joints). Click to shoot a box from the camera into the wall, or switch the click to explosions.

The controls are the same as in the *GPU Pyramid Demo*, and the panel has the wall's properties on top of the shot's.

The demo uses only the package's public API: `AvbdBodyBuffers` holds the bodies, `AvbdConstraintBuffers` holds the joints and has room for contacts, and `AvbdSolver.Step()` advances them. It contains no collision or fracture code.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *GPU Fracture Wall Demo*. (Package Manager copies the sample into `Assets/Samples`. After a package update, import it again to get the new version.)
2. Open the imported scene `FractureWall` and press **Play**, or add the *GPU Fracture Wall Demo* component to any scene (**GameObject > McKrill AVBD > GPU Fracture Wall Demo**).

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support.

## Controls

| Input | Action |
| --- | --- |
| Left click (outside the panel) | Shoot a box in the direction the camera looks, or, in explosion mode, set off an explosion where the camera ray hits |
| Right drag | Look around |
| `W` `A` `S` `D`, `Q` `E` | Move, down and up. `Shift` is three times faster |
| Mouse wheel | Move forward and back |
| `Space`, `N` | Pause / play, single step |
| `V` | Fire a volley: 15 shots at once at the first wall (the panel's **Fire volley** button) |
| `R` | Rebuild the wall with the panel's settings and remove the shots |

## The panel

The click mode button, the shot's speed, size and mass (or the explosion's radius, force and depth), the solver's iterations and the buttons are as in the pyramid demo. The wall's settings take effect on **Reset**, because the package cannot change a joint after it was added.

| Control | What it does |
| --- | --- |
| Click mode button | Switches the left click between shooting boxes and explosions. In explosion mode the panel stops showing the shot's Speed, Size and Mass and shows the explosion's Radius, Force and Depth instead (the shot's values are kept) |
| Radius | 1 to 50 m. A block whose center is farther from the explosion than this is not touched |
| Force | 0.1 to 1000 N s (log scale): the impulse at the center of the explosion. A body's velocity changes by this times a falloff `(1 - r/radius)^2` divided by its mass. The label shows the speed a default block (0.25 kg) at the center would get |
| Depth | 0 to 20 m: how far behind the surface the camera ray hits the explosion goes off, along the ray. 0 is on the surface |
| C, R, W | Columns (15 to 75), rows (15 to 75) and walls (1 to 10), three sliders on one row; the defaults are 75, 75 and 10. The total block count is shown under them |
| Block density | 0.1 to 10 (log scale). A block is 1 x 0.5 x 0.5 m, so density 1 is 0.25 kg. Does not apply to the bottom row, which is immobile |
| Fire volley | Fires 15 shots at once from a grid of 5 by 3 in front of the first wall (x = -30 to 30 m, y = 6, 19 and 32 m, 10 m away, at the default wall size; a smaller wall gets a smaller grid), along +z, with the panel's shot speed, size and mass. It is the volley of the performance tests `Scenario1` to `Scenario4` (`Tests/Performance/FractureWallScene.cs`): with a 2 m or a 1 m shot size and *Plastic yield* off or on, the demo shows the scenario the test times |
| Z-curve body order | Off (the default): the bodies are in the GPU arrays in the order the wall is built, row by row. On: in Z-curve order of their positions (`AvbdBodyBuffers.SpawnBatch`), so that blocks that touch are near each other in memory. Takes effect on **Reset**; use it to compare frame times |
| Rigid joints | Joints with infinite stiffness. Off: the joints' linear part is a spring with the stiffness below |
| Joint stiffness | 1e2 to 1e8 (log scale), used when the joints are not rigid |
| Fracture threshold | A joint breaks when the norm of its angular multiplier is above this (1 to 1e4, log scale) |
| Plastic yield | Off by default. On: the joints are plastic. The package then allocates the plastic array and runs the joint kernels that read it; off, it runs without any plastic code |
| Yield threshold | 0.1 to 1e4 (log scale). A plastic joint yields when the norm of its angular multiplier is above this: it gives way and keeps the bend instead of springing back. Keep it **below the fracture threshold**: above it a joint breaks before it can yield (the panel says so, and the package warns once) |
| Bend break, Bend limit | A plastic joint that has yielded breaks once it is bent more than the limit (1 to 180 degrees) from where it started |

The panel also shows how many joints have broken, out of all. Plastic yield, the yield threshold and the bend limit are greyed out until *Plastic yield* is on. Like the other joint settings they take effect on **Reset**.

## Explosions

In explosion mode a left click casts a ray from the center of the camera. The demo reads the poses back and tests the ray against the ground, the blocks and the shots (oriented boxes, on the CPU, once per click). The explosion goes off at the nearest hit plus *Depth* meters along the ray. If the ray hits nothing, nothing happens and the panel says so.

The physics is `AvbdSolver.Explode`: a radial impulse applied in the next step's predict stage, so the joints and contacts resist it in the same step. The push falls off to zero at the radius, is radial (no upward bias and no shock shell), and has a small off-center angular kick, hashed from the body's slot, that makes the blocks tumble. The immobile bottom row and the ground do not move. The explosion has no visual or audio effect. While the simulation is paused, the explosion waits for the next step (**Step** or **Space**). At most 16 explosions are applied per step.

## The wall

- **Walls.** The *W* slider (1 to 10, default 10) builds several identical walls one behind the other along z, 4 m apart (the first stands in front, nearest the starting camera). Like the other wall settings it takes effect on **Reset**. The block, joint and body counts grow with the number of walls, so a scene of many walls is a bigger scene for the solver.
- Blocks are 1 x 0.5 x 0.5 m in a running bond: odd rows are shifted by half a block, so every block rests on two blocks of the row under it. The wall stands on a static ground slab whose top is the plane y = 0. **The bottom row is immobile** (static bodies, no mass); the blocks of the row above are joined to it, and its blocks have no joints to each other.
- Every block has a ball-socket joint to the block on its left and to the two blocks under it, at the middle of the face they share. Each joint has a **rigid angle lock**. The fracture test looks at the angular multiplier, and only a rigid angle lock stores a multiplier, so the angle lock is not an option in the panel.
- A joint breaks in the step its angular multiplier exceeds the threshold. The next step removes it from the solver's lists, and from the step after that its two blocks collide with each other again, like any other pair (the reference solver's timing). Blocks that were joined do not collide while the joint holds.
- **The threshold has no calibrated unit.** The multiplier depends on the joint's penalty and on the constraint's torque arm, so the number to enter is found by trying: if nothing breaks, lower it or shoot a heavier, faster box; if the wall breaks apart by itself, raise it. The default is 100.

## Limits

- The wall is limited to 75 x 75 blocks in 10 walls (56 250 blocks) because that is the largest size that was tried. The default is that size, which needs a lot of GPU memory for the contact pool: lower W or the number of columns and rows on a smaller GPU. The constraint lists are built on the GPU (stage G) and nothing is read back for them; the graph coloring still reads two small things back every step (the number of colors), which is the remaining stall.
- The "joints broken" counter is a counter on the GPU that the panel reads once per step.
- **Drawing is instanced and stays on the GPU.** One `Graphics.RenderMeshPrimitives` call draws every box with the package's `AvbdInstancedBoxes` shader, which reads each box's pose from the solver's body state buffer: no pose readback and no GameObject per body (the pyramid demo still does both). The shader is unlit-style, with one directional light (the scene's sun if there is one) and ambient, so it works in the built-in pipeline and in URP, but **it has no shadows**. A player build needs the shader included, which it is: the package's shader set references it.
