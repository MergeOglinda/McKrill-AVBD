# GPU Fracture Wall Sphere Demo

The *GPU Fracture Wall Demo* with a sphere in place of every brick: a wall of spheres (75 x 75 by default, in ten walls) on the GPU solver, held together by joints that break under load, or that can bend and then break (plastic joints). Click to shoot a cube from the camera into the wall, or switch the click to explosions.

The wall is built from `BodyDescriptor.Sphere`; the ground and the shots are boxes. The solver's contact stages handle both shapes in one scene: sphere-sphere contacts in the wall and sphere-box contacts for the shots and the ground. Like the box demo, this one contains no collision or fracture code.

The controls and the panel are those of the *GPU Fracture Wall Demo* (see its README); this page lists what is different.

## Run it

1. Open **Window > Package Manager**, select **McKrill AVBD**, open the **Samples** tab and press **Import** next to *GPU Fracture Wall Sphere Demo*.
2. Open the imported scene `FractureWallSphere` and press **Play**, or add the *GPU Fracture Wall Sphere Demo* component to any scene (**GameObject > McKrill AVBD > GPU Fracture Wall Sphere Demo**).

The project needs a render pipeline that draws a `Universal Render Pipeline/Lit` or `Standard` material, and compute shader support.

## What is different

- **Packing.** The spheres of a row touch (0.01 m apart), and the odd rows are shifted by half a sphere, so every sphere rests in the gap of two spheres of the row under it: a hexagonal packing, with the rows sqrt(3) / 2 of a sphere apart. A wall of 75 rows is about 66 m high (the brick wall is 38 m), and a row of 75 spheres of radius 0.5 m is as wide as a row of the 1 m bricks.
- **Joints.** As in the brick wall: every sphere has a ball-socket joint with a rigid angle lock to the sphere on its left and to the two spheres under it, at the middle between them, so every sphere is joined to all six spheres that touch it. The fracture threshold default is 19, the value the performance scenarios use.
- **Mass.** *Radius* is 0.5 m and *Sphere density* 0.5 by default: a sphere then weighs 0.26 kg, about as much as a brick (0.25 kg), so the wall has about the mass of the brick wall and the same shots break it. The panel shows the mass per sphere. The bottom row is immobile.
- **Shots** are cubes, like in the brick wall, so the same shots hit both walls.
- **Fire volley** is the volley of the sphere performance scenarios `SphereWallPerformanceTests.Scenario1` to `Scenario4` (`Tests/Performance/SphereWallScene.cs`): 15 cubes of 2 m or 1 m, with *Plastic yield* off or on, at the same grid as the brick wall's.
- **Explosions.** The GPU ray hits a sphere at its surface (the ray cast knows the shape).

## Drawing

Two instanced draw calls draw everything with the package's `AvbdInstancedBoxes` shader, which reads each body's pose from the solver's body state buffer: one for the wall with a sphere mesh (a low-polygon sphere of 192 triangles, built by the demo, whose instance size is the diameter), and one for the ground and the shots with a cube mesh. There is no pose readback and no GameObject per body. The shader is unlit-style, with one directional light and ambient, and **has no shadows**: the spheres are tinted by row, and as a sphere has no visible texture, its rotation can only be seen from how the wall breaks up.

## Limits

- The wall is limited to 75 x 75 spheres in 10 walls (56 250 spheres), like the brick wall.
- The numbers have not been tuned for spheres: the fracture threshold and the joint stiffness are those of the brick wall.
