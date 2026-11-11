# Reference Solver Demo

A live view of the CPU reference solver (`McKrill.Avbd.Reference`) running the upstream avbd-demo3d scenes.
It exists so you can watch the port behave inside Unity before the GPU solver exists. It is a visualization
of the test oracle, not of the GPU solver: the reference is managed code with an O(n^2) broadphase and the
demo draws one GameObject per body, so expect it to slow down on large scenes. The GPU package will not
use one GameObject per body.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD** under *In Project* (or *Packages*), open the **Samples** tab and press **Import** next to *Reference Solver Demo*.
2. In an open scene choose **GameObject > McKrill AVBD > Reference Solver Demo** (or add the *Reference Solver Demo* component to any empty GameObject).
3. Press **Play**.

The demo uses the scene's main camera (and creates a camera and a light if the scene has none). It takes the
camera over while playing and restores it afterwards; untick *Control Camera* on the component to keep your own.
Works with the Input Manager and with the Input System package, because input is read through IMGUI events.

## Controls

| Input | Action |
| --- | --- |
| Panel: scene buttons, or keys `1`-`5` | Ground, Stack, Pyramid, Rope, Spring (the upstream demo scenes) |
| Panel: *Sequential / Colored*, or `C` | Primal sweep order. *Sequential* is upstream's order. *Colored* is the graph-colored order the GPU solver will use |
| Panel: iterations slider | Solver iterations per step (upstream: 10) |
| `Space`, `N` | Pause / play, single step |
| `B` or *Drop box* | Drop a random box above the scene |
| `R` or *Reset* | Reload the current scene |
| Left or right drag, mouse wheel | Orbit and zoom |

## Reading the picture

- **Tint > Graph color** colors each box by its graph color in *Colored* mode. Boxes with the same color are
  solved independently, which is what becomes one GPU dispatch per color. Switch to *Sequential* and the tint turns neutral.
- **Contact points** are drawn on top of the geometry: green for contacts whose friction is sticking, yellow for sliding.
- **Joints and springs** are drawn as lines with a cyan dot at each joint anchor (cyan lines are joints, orange lines are springs). A joint that fractures disappears.
- The panel shows body, manifold and contact counts, the number of colors, the smoothed time of one `Step()`, and
  *solve failures* if the 6x6 system ever lost positive definiteness (it should stay at zero).
- If the solver cannot keep up with 60 Hz, the demo caps the steps per frame and shows *running slower than real time*.

## Things worth trying

- **Stack**, *Sequential* versus *Colored*, at 10 iterations: the colored sweep converges a little differently, which is expected; both should settle.
- **Stack**, drop iterations to 1-3: the tower sags and wobbles; raise them and it tightens.
- **Rope**, drop boxes onto the chain: it is a 20-link hinged chain pinned to a static first link.
- **Pyramid**: the base size is a component field (upstream uses 16; the default here is 10 so the managed solver stays interactive).

## Frame conventions

The solver is z-up like upstream (gravity `(0, 0, -10)`). The demo parents its boxes under a child transform
rotated -90 degrees about X, so z-up solver space shows up as Unity's y-up world. Solver positions and rotations
are copied to the boxes unchanged.
