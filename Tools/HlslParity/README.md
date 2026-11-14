# HLSL parity harness

Compiles the package's vector-only contact HLSL as C++ (through `hlsl_shim.h`, which provides `float3`, `float4`, `float3x3`, `mul`, ...) next to upstream avbd-demo3d's unmodified solver and compares the two. It needs `g++` and `perl`; no GPU, Unity or OpenGL.

```
Tools/HlslParity/build.sh /some/output/dir
/some/output/dir/collide_parity 40 200000     # [scene steps] [random cases]
/some/output/dir/contact_parity 200000        # [random cases]
```

- `collide_parity` runs the demo scenes for a number of steps, then random box pairs, boxes on the ground, grids and crossing bars. It compares `AvbdCollideBoxes` with upstream's `Manifold::collide`: contact count, feature keys, rA, rB and normal. Expected: bit-identical.
- `contact_parity` compares `AvbdInitContacts` (warm start), `AvbdEvaluateContact` and `AvbdStampContact` (the primal terms: right-hand side and Hessian) and `AvbdContactDual` (multiplier, penalty, stick) with upstream's `Manifold::initialize`, `updatePrimal` and `updateDual`. Expected: bit-identical. The one input that cannot match is the angular constraint: upstream multiplies quaternions in a different order than `AvbdQuatDifference`, which differs by rounding, and penalties up to 1e8 amplify it to a few percent of the right-hand side. The harness therefore feeds the HLSL side upstream's own angular difference; `AvbdQuatDifference` itself is checked by the stage P and U tests.

What it does not check: that Unity compiles the HLSL (`half` and other reserved words, resource arguments), the compute shaders' buffer glue, the C# code.

The build adds an `f` to the HLSL float literals (they are `double` in C++), and uses `-ffp-contract=off` so the results do not depend on fused multiply-add.
