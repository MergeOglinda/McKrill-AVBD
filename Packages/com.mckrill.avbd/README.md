# McKrill AVBD

GPU-resident Augmented Vertex Block Descent (AVBD) physics for Unity 6.6 or newer, rendered through URP.
Rigid bodies, joints and soft bodies run entirely on the GPU. Bodies are rows in GPU buffers, never one GameObject each.

> **Status: 0.1.0, work in progress.** The CPU reference solver (a C# port of upstream avbd-demo3d, with sequential and graph-colored sweeps) is in and tested against the upstream C++ solver. The GPU solver is not written yet. To watch the reference run, import the **Reference Solver Demo** sample (see below).

## Requirements

- Unity 6.6 (6000.6) or newer
- Universal Render Pipeline (URP 17.6.0 ships with Unity 6.6)
- Reference configuration: Windows, DX12, RTX 3060 Ti. Windows Vulkan is also Tier 1.

## Install

Add the package from the git URL in the Package Manager (Add package from git URL):

```
https://github.com/MergeOglinda/McKrill-AVBD.git?path=/Packages/com.mckrill.avbd
```

Append `#0.1.0` (or another tag) to pin a version.

## Samples

Open **Window > Package Manager > McKrill AVBD > Samples** and import **Reference Solver Demo**. It runs the CPU reference solver live on the upstream demo scenes. See the sample's README for controls.

## Layout

| Folder | Assembly | Purpose |
| --- | --- | --- |
| `Runtime/` | `McKrill.Avbd.Runtime` | Solver core. No URP or Editor references. |
| `Runtime.URP/` | `McKrill.Avbd.Runtime.URP` | Thin URP render-graph adapter. |
| `Reference/` | `McKrill.Avbd.Reference` | CPU reference solver (port of upstream avbd-demo3d) used as the test oracle. |
| `Editor/` | `McKrill.Avbd.Editor` | Editor-only code. |
| `Shaders/Compute/` | | One `.compute` file per pipeline stage. |
| `Shaders/Include/` | | Shared `.hlsl` headers, included by package path. |
| `Shaders/Rendering/` | | URP-compatible draw shaders. |
| `Tests/` | `McKrill.Avbd.Tests.*` | Edit Mode and Play Mode tests. |
| `Samples~/` | | Importable samples (not compiled until imported). |

## License

MIT. See [LICENSE.md](LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
