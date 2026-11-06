# Third-Party Notices

This package contains code derived from the works listed below. Their license texts are reproduced in full as required by their licenses.

## avbd-demo3d (Chris Giles)

The CPU reference solver in `Reference/Solver` (`ReferenceSolver`, `AvbdBody`, `AvbdForce`, `AvbdManifold`, `AvbdJoint`, `AvbdSpring`, `AvbdIgnoreCollision`, `BoxCollision`, `AvbdScenes`) and the quaternion and 3x3 helpers in `Reference/Math` are a C# port of the AVBD 3D demo solver (`solver.cpp`, `manifold.cpp`, `collide.cpp`, `joint.cpp`, `spring.cpp`, `rigid.cpp`, `maths.h`, `scenes.h`). The port keeps upstream's algorithms, constants and feature-key layout; deviations are documented where they occur in the code (gravity as a vector, a skipped update instead of NaNs on a failed solve, graph-colored sweep ordering).

`Tests/Reference/UpstreamGolden.cs` contains numeric trajectories recorded by running the upstream solver on its own demo scenes.

The upstream source is vendored in this repository under `savant117-avbd-demo3d/avbd-demo3d-main`.

```
MIT License

Copyright (c) 2025 Chris Giles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pending

Any further reference implementation or paper whose code or data layout is adapted (for example when the GPU kernels are ported) must be added here before release.
