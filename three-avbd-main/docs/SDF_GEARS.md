# Contact-driven gear train

Run `pnpm dev` and open `/sdf-gears.html`.

The gold 20-tooth wheel is the only driven body. Its kinematic motor uses `Body.moveTo`
with a bounded speed and acceleration; the 12- and 28-tooth followers are dynamic.
Two ball joints per follower constrain the shaft while leaving its axial rotation free.
There are no gear-ratio constraints, follower pose writes, or follower velocity commands
in the demo. Reverse, handwheel gestures, and arrow-key nudges all operate on the input.

The profiles share a module and 25-degree pressure angle, with involute flanks, a radial
root extension and clearance for the interpolated field. Each watertight extruded mesh
is baked at resolution 80. Polygon area/second-moment integration provides the mass and
principal inertia at unit density. Collision fields and visible teeth share the same
geometry; face hardware is decorative.

## Controls and evidence

- Auto drive starts by default (paused for reduced-motion preferences). The speed slider
  covers 0–8 rpm, and Reverse smoothly changes the input's direction.
- Drag the gold gear or the small handwheel to take manual control. Focus the handwheel
  and press arrow keys to nudge it; Home holds its current position. Manual speed is capped
  to avoid driving the discrete collision search through a tooth in one step.
- Contact points are sampled at 10 Hz. Their color/size maps the magnitude of the solver's
  normal-force multiplier logarithmically from cool to warm, saturating at 50 N. This is a
  load visualization, not a penetration-depth display. Body rendering reads GPU poses directly.
- Tooth collisions resets the train with collision masks on/off. With collisions disabled,
  the input still turns while both followers remain stationary. This distinguishes contact
  transmission from an animation with prescribed ratios.
- Physics uses 120 Hz fixed steps, 20 iterations and at most four catch-up steps per frame.
  The footer reports browser frame cadence and simulation-time/wall-time separately.
  Hidden tabs suspend work; Pause stops physics and repeated readbacks.

`tests/sdf-gears.test.ts` checks shared pitch, initial tooth/gap phases, profile bounds and
positive symmetric mass/inertia. `tests-gpu/sdf-gears.gpu.test.ts` runs a bounded, sequential
real-Dawn experiment with forward drive, reversal, manual input and disconnected controls.
It checks accumulated rotation ratios (within 0.14 rad), aligned shafts, finite state,
nonzero contact loads, zero capacity overflow and axle drift below 0.035 world units.

This is an experimental collision demonstration, not a precision gear-design tool. Voxel
interpolation, approximate contact search and compliant ball joints cause small speed ripple,
backlash and shaft displacement. The root construction is not a manufactured undercut profile.
A full browser FPS number alone does not guarantee real-time physics; check the simulation
rate too. Initial shader compilation may stall before warmed playback.

## Local validation (2026-10-24)

The full `pnpm check` gate passed: typecheck, 162 CPU tests, 179 real-Dawn GPU tests
(no skips), Vite build and library build. A temporary test launcher serialized test
files without modifying the repository's test commands. The gear test includes the
8 rpm slider maximum. The existing chair/bunny settling regressions remain green.

The built page and all directly linked assets return HTTP 200. Browser checks exercised
auto drive, reversal, direct gear dragging, handwheel dragging, keyboard nudges, collision
disabling/reset, contact toggling and Pause. A 390×844 layout keeps controls and labels
visible. The production browser console reported no warnings/errors. Warmed local M4 Max
playback observed approximately 118–120 browser frames/s and 0.98–1.00 simulated seconds
per wall second with the overlay enabled. These short observations are not a sustained
benchmark or a result for other hardware.
