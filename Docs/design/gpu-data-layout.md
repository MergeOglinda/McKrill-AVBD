# GPU data layout and buffer lifecycle (task 3)

Design spec for the package's GPU buffers. It is written step by step; each section is added when its step is done.

| Step | Section | Status |
| --- | --- | --- |
| 1 | Read/write matrix | Done |
| 2 | Packing rules | Done (this version) |
| 3 | Body arrays | Done for rigid boxes (section 3) |
| 4 | Other buffers | Joints and springs done (section 12); contacts and the other buffers not started |
| 5 | Capacity calculator | Done for the body arrays (section 5) |
| 6 | Slot allocation and edits | Done (section 6) |
| 7 | Growth and overflow policy | Overflow policy done, growth not started (section 7) |
| 8 | Buffer owner | Done for the body arrays (section 8) |
| 9 | HLSL/C# layout contract | Done for the body arrays and edit commands (section 9) |
| 10 | Tests and review | Tests written for the body arrays; not yet run in Unity (section 10) |

All sizes in this document are estimates from the working design. They are not measured.

**Revision note.** Sections 1 and 2 were corrected against the reference solver (savant117 avbd-demo3d, vendored in this repo). The step-start pose is hot (contacts read it every iteration), the previous velocity is added for the adaptive warm start, the warm-start snapshot must carry anchors and the stick flag, the solver constants are completed, and joints are noted as a separate pattern. See findings 1, 4, 5, 6, 9, 10 and the changed matrix rows.

## 1. Read/write matrix

### Scope

This matrix covers dynamic rigid bodies and their contacts. Soft-body particles and elements and static colliders follow the same pattern and are added in step 4. Joints do not: they evaluate the constraint from the current poses (no step-start pose), use a quaternion-difference angular constraint scaled by a torque arm (needs both body sizes), add a geometric-stiffness term, carry stiffness caps and a fracture flag, and may be anchored to a world point with no body A. Step 4 defines a separate joint record. Static colliders are tested directly per dynamic body in the narrowphase and never enter the pair list.

### Stages

| Code | Stage | What it does |
| --- | --- | --- |
| E | Edits | Applies edit commands (spawn, remove, property change) and uploads kinematic transforms |
| S | Sort | Computes a Morton key per active body and radix-sorts the active list |
| B | Broadphase | Builds the hashed grid and generates body pairs |
| N | Narrowphase | Turns pairs into contacts and warm-starts them from the previous step |
| G | Gather | Builds per-body constraint lists (CSR) |
| C | Coloring | Assigns colors with temporal coherence and buckets bodies by color |
| P | Predict | Computes the inertial target and the initial guess |
| V | Solve | Primal update of each body, once per color per iteration |
| D | Dual update | Updates multiplier and penalty of each constraint, once per iteration |
| U | Velocity update | Derives velocity from displacement, applies the speed limit, snapshots warm-start data |
| Z | Sleep | Updates sleep counters and compacts the active list |
| R | Render | Culls and bins instances, then draws them |

Order within one step: E, S, B, N, G, C, P, then (V, D) repeated per iteration, then U and Z. R runs once per frame.

Cell codes: R = reads, W = writes, RW = both, blank = untouched. Hot means touched in every solver iteration (V or D). Cold means touched at most once per step.

### Body fields

Persistent per-body state.

| Field | E | S | B | N | G | C | P | V | D | U | Z | R | Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Position | W | R | R | R | | | RW | RW | R | R | | R | Hot |
| Rotation | W | | R | R | | | RW | RW | R | R | | R | Hot |
| Mass and principal inertia | W | | | | | | | R | | | | | Hot |
| Inertial target (position, rotation) | | | | | | | W | R | | | | | Hot |
| Step-start pose | W | | | | | | W | R | R | R | | R | Hot |
| Linear and angular velocity | W | | R | | | | R | | | W | R | | Cold |
| Previous linear velocity (adaptive warm start) | W | | | | | | R | | | W | | | Cold |
| Shape type | W | | R | R | | | | | | | | R | Cold |
| Shape dimensions | W | | R | R | | | | | | | | R | Cold |
| Group mask | W | | R | | | | | | | | | | Cold |
| Flags (kinematic, asleep, ...) | W | | R | R | | | R | | | | RW | | Cold |
| Friction | W | | | R | | | | | | | | | Cold |
| Graph color | | | | | | RW | | | | | | | Cold |
| Constraint-list range (offset, count) | | | | | W | | | R | | | | | Hot |
| Sleep state | W | | | | | | | | | | RW | | Cold |
| Appearance id | W | | | | | | | | | | | R | Cold |

Notes:

- V writes only the pose of the body it solves. A body is solved by exactly one thread, and neighbours are read from other color groups, so no atomics are needed.
- V reads the position, rotation and step-start pose of the other bodies in each of its contact constraints, because the contact constraint is evaluated as C0 plus a Jacobian times the displacement since the step start, for both bodies. The partner needs no mass, inertia, shape or friction: contact points are stored in body-local coordinates and friction is stored in the contact. Joints evaluate C from the current poses and do not need the step-start pose.
- The kinematic upload in E also writes pose and velocity for kinematic bodies. P skips them through the flags.
- P overwrites the pose with the initial guess. Everything that needs the pose from the start of the step (S, B, N) must run before P, or read the step-start pose.
- P computes the adaptive warm-start weight from the change between the previous and the current linear velocity, so it reads both. U copies the current velocity to the previous-velocity field before it overwrites the velocity.

### Derived buffers

Per-step buffers that are not part of body state.

| Buffer | E | S | B | N | G | C | P | V | D | U | Z | R | Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Active list | | R | | | | | R | | | R | RW | | Cold |
| Sort keys and sorted order | | W | R | | | R | | | | | | | Cold |
| Grid cells | | | RW | | | | | | | | | | Cold |
| Pair list, binned by pair type | | | W | R | | | | | | | | | Cold |
| Contacts (current step) | | | | W | R | R | | R | RW | R | | | Hot |
| Warm-start data (previous step) | | | | R | | | | | | W | | | Cold |
| Constraint-list entries (CSR) | | | | | W | | | R | | | | | Hot |
| Color buckets and per-color counts | | | | | | W | | R | | | | | Hot |
| Control counters | | W | W | W | W | W | | | | | W | | Hot |
| Indirect arguments | | W | W | W | W | W | | | | | W | W | Hot |
| Status and overflow flags | | W | W | W | W | W | | | | | W | | Cold |
| Draw lists and indirect draw arguments | | | | | | | | | | | | RW | Cold |

Control counters and indirect arguments are tiny and stay in cache, so they are hot by access pattern but cost almost no bandwidth. Counter writes happen in the stage that produces the count. The next stage reads the counters through indirect arguments.

### Contact fields

Only the stages that touch contacts are shown.

| Field | N | G | C | V | D | U | Class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Body A and body B | W | R | R | R | R | R | Hot |
| Feature id | RW | | | | | R | Cold |
| Normal (tangents derived) | W | | | R | R | | Hot |
| Local anchors, one per body (kept from the previous step while sticking) | RW | | | R | R | | Hot |
| Initial constraint value (step start) | W | | | R | R | | Hot |
| Penalty, one per row | W | | | R | RW | R | Hot |
| Multiplier, one per row | W | | | R | RW | R | Hot |
| Friction and flags (includes the stick flag, set by D) | RW | | | R | RW | R | Hot |

N reads the previous step's warm-start data and writes the new multiplier and penalty scaled by the warm-start factors. If the matched contact was sticking in the previous step, N keeps its old anchors instead of the new ones (this is what holds static friction). The solver parameters (alpha, linear and angular beta, gamma, minimum and maximum penalty, collision margin, stick threshold, iteration counts) are global constants in one small constant buffer, not per-record fields.

### Findings

These findings change or refine the proposed body arrays in task 3, step 3.

1. **The solver also reads mass and inertia.** The proposal says the solver reads only the pose and inertial arrays, but V needs the mass and the principal inertia of the body it solves. The hot self-read set is pose (28 B of payload), inertial target (28 B) and mass properties (16 B), 72 B in total, plus the constraint-list range. The mass properties join the hot set. Finding 6 adds the step-start pose (28 B), making it about 100 B.
2. **The constraint-list range is hot and must not share an array with cold fields.** The proposal groups CSR offsets with sort key, color and sleep state in one "Extras" record. V reads the range in every iteration, so keeping it there would pull cold fields into every solver read.
3. **The sort key is transient.** S writes it and the radix sort consumes it within the same stage, so it needs no persistent per-body field. It lives in the sort scratch buffers. The remaining Extras (graph color, sleep state) are cold and only touched by C and Z.
4. **Constraint reads dominate solver traffic.** With the proposal's estimate of 12 contacts per body and 96 to 128 B per contact, one body reads roughly 1.5 to 1.9 KB of contact records and partner poses in V (12 x 96 to 128 B, plus 12 x 64 B for partner pose and step-start pose, see finding 6), against about 100 B for its own state. The contact record size and layout matter far more than how the body arrays are grouped.
5. **The warm-start snapshot must carry anchors and the stick flag.** From the previous step N needs the key (bodies, feature id), multipliers and penalties, and also the local anchors and the stick flag: the reference keeps the old anchors of a sticking contact, which is how static friction holds, and the stick flag is set by D. A snapshot without them loses static friction and contacts creep. This makes the compact snapshot about 64 B per contact, not 48 B, giving about 461 MB in the 200k-body example instead of about 422 MB; double-buffering full contact records is about 614 MB. It is still allowed under P3 only if read in index order. It costs an extra write pass and needs a lookup structure for matching, so step 4 should compare both options.
6. **Ordering constraint.** N must run before P, because P overwrites the pose with the initial guess. The step-start pose is written by P and read by V and D (contact constraints are evaluated from the displacement since the step start, for both bodies of every contact), by U and by motion vectors. It is hot. Step 3 decides whether to interleave it with the pose in one 64 B record (two sectors for both) or keep a separate array; the partner gather in V reads both either way.
7. **Gap in the proposal.** Rendering needs a per-instance appearance id (material or tint), which no proposed array holds. It fits in the Meta record or in a user-supplied buffer. Step 3 should decide.
8. **Neighbour reads are gathers.** V reads partner poses by slot id, so with stable slots those reads are scattered. This is the cost that physically reordering bodies would reduce, which is why decision 1 (stable slots versus reordering) needs the benchmark in task 19.

9. **Previous velocity is missing from the proposal.** The adaptive warm start needs last step's linear velocity. It is written by U and read by P, the same stages as velocity, so it can share the velocity record (about 16 B). If adaptive warm start is dropped, that must be a deliberate decision, since it breaks parity with the CPU reference.
10. **Solver constants and joints.** The constant buffer needs separate linear and angular beta (joints use the angular one), the collision margin, the stick threshold and the minimum penalty. Joints are a separate record, see the scope note.

Everything the matrix marks cold for the solver (velocity, previous velocity, shape, group mask, friction, flags, color, sleep state, appearance id) can stay grouped by access phase as proposed.

With these changes the body state comes to roughly 230 to 245 B per body (the previous velocity adds about 16 B to the proposal's 216 B). The exact figure is fixed in step 3.

## 2. Packing rules

These rules apply to every GPU buffer. No struct is defined until they are written down, and later steps cite them by number. They come from the Unity constraints for this task: structured buffer strides must be a multiple of 4 and below 2048 bytes, should be a multiple of 16, and should be built from float4-sized fields because mixed variable sizes can lay out differently per graphics API.

### Rules

| Rule | Statement | Why |
| --- | --- | --- |
| P1 | **Vector words only.** Every structured-buffer element is built only from `float4` and `uint4` members. No bare `float`, `float2`, `float3`, `uint2` or `uint3` members, and no explicit padding variables. Exception: a homogeneous scalar buffer (`uint` or `float`, stride 4) holding one kind of scalar, such as index lists, sort keys, counters and per-element counts. | Mixed member sizes can lay out differently per graphics API. A single scalar type has no layout ambiguity. |
| P2 | **Member typing.** A `float4` member holds floats only. A `uint4` member holds ids, flags, counts, and any float passed through `asfloat`/`asuint`. Ids, flags and sentinels are never stored in `float4` members. | `0xFFFFFFFF` is a NaN when read as a float, and a float path is not guaranteed to preserve NaN bit patterns. |
| P3 | **Stride.** Every stride is a multiple of 16 B and below 2048 B. Records read by index in V, D or N (gathers) use 16 B or a multiple of 32 B, so that no record straddles a 32 B sector on the reference GPU. Records read in index order by consecutive threads (streams) may use any multiple of 16 B. | A gathered 48 B record always touches two sectors for 48 B of payload. A stream pays no such cost. |
| P4 | **Spare lanes.** The unused `w` of a `float3` may carry a scalar only if every stage that reads or writes that scalar also reads or writes the vector in the same `float4`. Otherwise the lane stays unused. | A spare lane is free only when it is fetched anyway. |
| P5 | **Grouping by access phase.** Fields share a record only if the same stages touch them (section 1). A hot record never grows to hold a cold field. Fields written in the solver iterations (pose, multipliers, penalties) go in different arrays from fields that are read-only in the iterations, wherever the split adds no sector to any hot stage. | One cold field in a hot record is paid for in every iteration. |
| P6 | **Invalid id.** `AVBD_INVALID = 0xFFFFFFFFu` means no body, no contact, empty slot or no color. Every capacity and valid id stays below `0x7FFFFFFF`, so the top bit is free for tagging (for example references into the static collider table, decided in step 4) and `AVBD_INVALID` can never be a valid tagged id. The capacity calculator enforces the limit. | One sentinel everywhere, and room to tag ids later without a layout change. |
| P7 | **Flags.** Flags live in one `uint` per record: the low 24 bits are flags and the high 8 bits are an enum kind (shape type for bodies, constraint type for constraints). Bit positions are named in one place, the layout contract (step 9), with a C# `[Flags]` mirror. Each flag bit has exactly one owning writer stage. Unused bits are zero, and a stage changes only the bits it owns, by masking. | One word per record, no cross-stage write races. |
| P8 | **Numeric conventions.** 32-bit floats throughout, no 16-bit types. Rotations are full unit quaternions stored as (x, y, z, w) with the Unity.Mathematics convention. A body's position is its center of mass, which equals the shape center in v1. Body axes are the principal axes, so inertia is diagonal in body space. Contact anchors are stored in body space. World space follows Unity (left-handed, Y up). | The CPU reference (task 4) and the GPU kernels must mean the same thing by every field. |
| P9 | **No reliance on initial contents.** The package does not assume new buffers are zeroed. Buffers that kernels might read before writing are cleared explicitly at creation and after growth. A free slot is marked by a flag bit and `AVBD_INVALID` in its id fields. | Keeps behavior identical across APIs and drivers. |
| P10 | **One definition per record.** Each record has one HLSL definition and one C# mirror built from `float4`/`uint4` words. Kernels access records only through accessor functions, never raw field offsets (step 9). | Layout cannot drift between kernels, C# and the CPU reference. |

### Size classes

| Size | Use |
| --- | --- |
| 4 B scalar | Index lists, sort keys, counters, per-element counts (the P1 exception) |
| 16 B | A single vector word, small ranges, control blocks |
| 32 B | Standard body record: one sector. Pose, inertial target, step-start pose, velocity, shape and meta records are expected to use this class |
| 64, 96, 128 B | Constraint records (contacts, joints), always a multiple of 32 B when gathered. The exact size per record is set in step 4 |
| 2048 B or more | Not allowed |

### How each rule is checked

| Rule | Check |
| --- | --- |
| P1, P2, P3, P10 | Layout round-trip test (steps 9 and 10) on DX12 and Vulkan, plus a stride assertion in C# that each mirror struct has the expected size |
| P4, P5 | Review against the section 1 matrix whenever a record is defined or changed. There is no automated check |
| P6 | Capacity calculator tests (step 5) |
| P7 | Test that the C# flag enum has no overlapping bits, and a debug assertion for owned-bit writes |
| P8 | CPU reference parity test (task 4) |
| P9 | Growth and clear tests (step 10) |

### Carried into step 3

- **Appearance id (section 1, finding 7).** Default: it lives in the Meta record. It is read only by R, so under P4 it can sit in a spare lane only if R also reads that record's vector. Step 3 places it.
- **Compact warm-start snapshot (section 1, finding 5).** It is about 64 B because it carries anchors and the stick flag, about 461 MB for the 200k-body example. It is a multiple of 32 B, so it is also safe to gather by key under P3. Step 4 decides between it and full double-buffered contact records.
- **Step-start pose and previous velocity (findings 6 and 9).** Step 3 places both: interleaved with the pose or separate, and inside the velocity record.
- **Joint record (scope note).** Step 4 defines it separately from contacts.

## 3. Body arrays

Scope: rigid boxes. Contacts, joints and the other buffers are step 4. Seven arrays, indexed by body slot, 224 B per body in total. Each record follows the packing rules in section 2 (float4 and uint4 words only, unused lanes zero). The C# mirrors are in `Runtime/Layout/BodyRecords.cs`, the HLSL in `Shaders/Include/AvbdBodyRecords.hlsl`.

| Array | Stride | Words | Class | Written by | Read by |
| --- | --- | --- | --- | --- | --- |
| State | 64 B | position, rotation, step-start position, step-start rotation | Hot | E (spawn, pose edits), P, V | S, B, N, V, D, U, R |
| Inertial | 32 B | inertial position, inertial rotation | Hot | P | V |
| Mass | 16 B | principal moments (xyz), mass (w) | Hot | E | V |
| Motion | 48 B | linear velocity, angular velocity, previous linear velocity | Cold | E, U | B, P, Z |
| Shape | 16 B | size (xyz), friction (w) | Cold | E | B, N, R |
| Meta | 32 B | flags and kind, group mask, appearance id; graph color, sleep counter | Cold | E (first word group), C, Z | B, N, P, R |
| Range | 16 B | constraint-list offset and count | Hot | G | V |

Decisions made here, where section 1 left them open:

- **Step-start pose is interleaved with the pose** in the 64 B State record, so the partner gather in V and D is one record, not two arrays. The record is two sectors; V writes only the first two words.
- **Previous linear velocity is in the Motion record** (finding 9). Motion is 48 B. It is read by P and Z by slot through the active list and is cold (once per step), so rule P3 does not require 32 B alignment for it. P3 names V, D and N as the strict gathers.
- **Friction is in the spare lane of the size vector** (rule P4). Its stages (E, N) are a subset of the size stages (E, B, N, R).
- **Mass and the three principal moments share one word.** V reads them together.
- **Appearance id** is in the Meta record (finding 7). Group mask is there too.
- **Free slots**: the Free flag is set, the appearance id and the graph color are `AVBD_INVALID`, the mass is zero and rotations are identity (rule P9).
- **Mass 0 means static.** A static body needs no flag. Static colliders in the sense of section 1 (the separate table) are still step 4.

Matrix changes from this step: E also writes the step-start pose (spawn and pose edits, so a teleport leaves no motion streak and a new body has a valid start pose before P runs), and E writes the previous linear velocity (spawn and velocity edits, so the adaptive warm start sees no acceleration). E never writes the inertial target.

## 5. Capacity calculator

`BodyBufferCapacity` (`Runtime/Buffers/BodyBufferCapacity.cs`) holds the cost and the limits of the body arrays.

- Cost: 224 B per body, so 1,000,000 bodies take 224 MB for the body arrays alone.
- Limit: the largest array is 64 B per body, so the capacity is capped at `int.MaxValue / 64` = 33,554,431 bodies to keep every buffer under 2 GiB. This is far below `0x7FFFFFFF`, so rule P6 holds.
- Minimum capacity is 1. Capacity is fixed when the buffers are created (section 7).
- The edit command buffer (128 B per command, 1024 commands by default) is separate and small.

The calculator covers the body arrays only. Contacts, constraint lists and scratch buffers are added with step 4.

## 6. Slot allocation and edit commands

A body is a slot (a row in every array) plus a handle. `BodyHandle` carries the slot and a generation; freeing a slot bumps its generation, so an old handle never edits a newer body that reused the slot. The default handle is invalid.

**Slot allocation** (`BodySlotAllocator`, CPU only): a freed slot is reused before a never-used one, most recently freed first, so live bodies stay packed near the start. `HighWaterMark` is one past the highest slot ever used. Allocation happens when `Spawn` is called, so the caller has a usable handle before the GPU sees anything.

**Edit commands** (`EditCommand`, 128 B, one header word and seven payload words) come in three kinds:

| Kind | Effect |
| --- | --- |
| Spawn | Writes every persistent field of the slot. Clears the Free flag. Sets the step-start pose and the previous velocity. Graph color is invalid, sleep counter zero, range empty |
| Remove | Sets the Free flag, keeps the shape kind, clears the other flags, zeroes the mass and the range, sets the appearance id and graph color to invalid |
| Set | Overwrites the fields named in a field mask: pose, velocity, shape (size and density), friction, group mask, appearance id, kinematic flag |

Pose and velocity edits also reset the step-start pose and the previous velocity (see section 3). A kinematic transform upload is a pose edit plus a velocity edit. Kinematic bodies have no counterpart in the CPU reference, so their contact behavior is untested against an oracle.

**One command per slot per batch.** The kernel runs one thread per command with no ordering, so two commands for the same slot would race. `BodyEditQueue` therefore merges everything done to a slot in one batch by its final effect, compared with what the GPU holds at the start of the batch:

| GPU slot at batch start | Slot at the end of the batch | Command |
| --- | --- | --- |
| Free | Free (spawned and removed inside the batch) | none |
| Free | Live | Spawn |
| Live | Free | Remove |
| Live | Live, a new body spawned into the reused slot | Spawn (overwrites everything) |
| Live | Live, edited | Set, with the union of the edited fields |

Commands are emitted in the order slots were first touched.

**Rejected input.** A descriptor or edit with a non-finite or out-of-range value throws `ArgumentException` and uses no slot. An edit through an invalid or stale handle returns false and changes nothing. Rotations are normalized on the CPU. Mass properties are computed on the CPU with the same formulas and operation order as upstream, so they match the reference solver exactly.

## 7. Growth and overflow policy

Done so far: **overflow**. Capacity is fixed. `Spawn` on a full buffer returns the invalid handle and increments `OverflowCount`; nothing else changes. Not done: **growth** (reallocating larger buffers and copying the old contents), including the clear after growth that rule P9 requires. Until it is built, capacity must be chosen when the buffers are created.

## 8. Buffer owner (stage E)

`AvbdBodyBuffers` (`Runtime/Buffers/AvbdBodyBuffers.cs`) owns the seven arrays and the command buffer.

- Creating it allocates the arrays and runs the `ClearBodies` kernel over every slot (rule P9; chunked, so more than 65,535 thread groups are covered).
- `Spawn`, `Remove` and the `Set*` methods only record edits on the CPU. `ApplyEdits()` drains the queue, uploads the commands in chunks of the command batch size and dispatches the `ApplyEdits` kernel once per chunk.
- The other stages bind the arrays through the read-only properties `State`, `Inertial`, `Mass`, `Motion`, `Shape`, `Meta` and `Range`. Only stage E writes the fields it owns.
- Nothing is read back during a step. Tests read back synchronously.
- The owner is single-threaded (main thread) and `IDisposable`.

## 9. HLSL/C# layout contract

One definition per record on each side, field for field: `BodyRecords.cs` and `EditCommand.cs` against `AvbdBodyRecords.hlsl` and `AvbdEditCommand.hlsl`. The constants (flag bits, edit kinds, field bits, `AVBD_INVALID`, group size) are defined in both. Kernels write the records with plain struct assignment for now; accessor functions (rule P10) are added when a second kernel needs the same field. Section 10 lists the tests that keep the two sides in step.

## 10. Tests

| Test | Needs a GPU | Checks |
| --- | --- | --- |
| `BodyLayoutTests` | No | Struct sizes equal the strides, rules P3 and P6, flag bits, and a parse of the HLSL includes: the constants and every struct member (type, name, order) equal the C# side |
| `BodySlotAllocatorTests` | No | Allocation order, reuse, stale handles, double free, exhaustion |
| `BodyEditQueueTests` | No | Encoding of every field, validation, and every row of the merge table in section 6 |
| `BodyEditsOracleTests` | No | The behavior of each command on the CPU oracle, and mass properties equal to `AvbdBody` |
| `BodyEditsGpuTests` | Yes | The arrays after creation, spawn and remove, and 40 random edit batches replayed on the oracle, compared bit for bit |

The CPU oracle is `Reference/Edits/CpuBodyArrays.cs`. It must change together with `AvbdBodyEdits.compute`. The GPU tests have not been run: no Unity editor or GPU was available when they were written.

## 11. Solver stages

The solver stages run in the order E, P, then per iteration V and D, then U (section 1). They are added one at a time. Every stage has a C# oracle in the Reference assembly that does the same arithmetic on `CpuBodyArrays`, and the GPU kernel is tested against it. Results are compared with a small relative tolerance, not bit for bit, because GPU arithmetic may differ in the last bits (fused multiply-add, reciprocal square root).

### Solver constants

`AvbdSolverSettings` holds the global constants: time step, gravity vector, iterations, alpha, linear and angular beta, gamma, and the minimum and maximum penalty. The kernels read them as plain uniforms (`_SolverParams0`, `_SolverParams1`, `_Gravity`, declared in `AvbdSolverParams.hlsl`), set by `SolverUniforms` before each dispatch. The iteration count is not a uniform: the iteration loop runs on the CPU, one dispatch per pass. The default gravity is `(0, -10, 0)` for Unity's y-up world; upstream's demo scenes are z-up and need `(0, 0, -10)`.

### Stage P: predict

One thread per slot, over `[0, HighWaterMark)`. Free slots, kinematic bodies and bodies with mass 0 are skipped completely: their pose and inertial target are not touched. For every other body:

| Writes | Value |
| --- | --- |
| Inertial target | position + v dt + g dt^2; rotation advanced by the angular velocity |
| Step-start pose | the pose before this stage |
| Pose | position + v dt + g (w dt^2) with the adaptive weight w; rotation advanced by the angular velocity |

The adaptive weight w is the fraction of gravity the body actually accelerated by last step: clamp(dot(a, g) / |g|^2, 0, 1) with a = (v - previous v) / dt, and 0 if the result is not finite or g is zero. The previous velocity comes from the Motion record. It is set by stage U, and by edits (so a new or just-edited body has w = 0).

Statics keep their stale inertial target because no stage reads it for a static body.

The oracle is `CpuPredict`. With zero iterations the reference solver runs exactly this phase, so `PredictTests` compares the oracle against `ReferenceSolver` directly, and then the GPU kernel against the oracle.

### Stage V: primal update

One thread per solved body, one dispatch per graph color, repeated every iteration, so a sweep is `colors` dispatches in order. A body is solved if it is not free, not kinematic and has mass above zero; every other body is a fixed anchor. A body writes only its own pose, and every neighbour it reads belongs to another color, so the kernel needs no atomics, except a counter of failed solves (`Status`, word 0).

For the body it solves, the kernel builds the 6x6 system exactly like the reference solver:

| Part | Value |
| --- | --- |
| Inertia | `lhsLin = diag(m / dt^2)`, `lhsAng = diag(I / dt^2)`, `rhsLin = (m / dt^2)(x - x_inertial)`, `rhsAng = (I / dt^2)(q - q_inertial)` (upstream's quaternion difference, no sign flip) |
| Constraints | every entry of the body's list adds its terms, newest constraint first (section 12) |
| Solve | 6x6 LDL^T without pivoting, reading only the lower triangle. A pivot not above 1e-20 (or NaN) fails the solve: the body keeps its pose and the failure counter increments |
| Update | `x += dx.lin`, `q = normalize(q + 0.5 (dx.ang, 0) q)` |

The joint terms are the reference's: linear rows with a Jacobian of plus or minus identity and a skew-symmetric angular part, the diagonal geometric-stiffness term, and the optional angle row scaled by the torque arm. The geometric-stiffness sum is written in closed form, `H = -dot(F, r) I + r F^T`, which equals the reference's three-term sum (checked numerically). Springs skip the update when the anchors are closer than 1e-6.

The oracle is `CpuPrimal` (it reuses `Mat6.TrySolveLDLT` of the reference). `PrimalTests` checks it against `ReferenceSolver` in `Colored` mode with one iteration, which produces exactly the predict phase, the constraint warm start and one primal sweep as far as the body poses go. The GPU kernel is then compared with the oracle on the rope, the spring scene and random lattices of boxes with joints, springs and world pins. This needs the per-step constraint warm start (C0 and the decayed multipliers and penalties), which is the reference's `Joint.Initialize`. Its oracle (`CpuConstraintInit`) exists so the tests can prepare records; its kernel is part of stage D.

Not part of this stage: the dual update, the warm start kernel, the velocity update, contacts, and GPU-built lists and coloring.

### Stage D: constraint warm start and dual update

`AvbdDual.compute` has two kernels, both one thread per constraint over the constraint records. A thread writes only its own record and reads body poses that no stage writes meanwhile, so there is no coloring, no ordering inside a dispatch and no atomics. Springs and broken joints are skipped.

| Kernel | When | What |
| --- | --- | --- |
| `ConstraintInit` | Once per step, after stage P | Evaluates each joint at the **start-of-step** pose (the state record's start fields, written by P) and stores C0. Warm-starts the multipliers (`lambda * alpha * gamma`) and penalties (`clamp(penalty * gamma, min, max)`), then caps the penalties of non-hard rows at the stiffness. The reference's `Joint.Initialize` |
| `Dual` | After every primal sweep | For each row with a non-zero penalty: a hard row is stabilized (`C -= C0 * alpha`) and stores `lambda = penalty * C + lambda`; the penalty then grows by `|C| * beta` up to the stiffness (the maximum penalty for a hard row). Then the fracture test. The reference's `Joint.UpdateDual` |

A step of `AvbdSolver.Step()` is now: apply edits, rebuild constraints if needed, P, `ConstraintInit`, then per iteration the primal sweep (one dispatch per color) and `Dual` (one dispatch).

**Flags.** `Broken` is owned by `Dual`: when a fracturable joint's angular multiplier exceeds the threshold (`|lambdaAng|^2 > fracture^2`, tested for every fracturable joint, whatever its penalty), the kernel zeroes its penalties and multipliers and sets the flag. A broken joint has zero penalty, so the primal stage adds nothing for it, and both stage D kernels skip it. It stays in the lists and the coloring (see section 12). `Fracturable` is a flag plus a finite threshold; a joint without it never breaks.

**Encoding.** The stiffness caps follow the flags, not infinity: `HardLinear` or `HardAngular` selects "no stiffness cap" (the penalty maximum), anything else uses the stored finite stiffness. A non-hard angular stiffness of 0 disables the angle row, because the warm start caps its penalty at 0.

**Check.** The oracles are `CpuConstraintInit` and `CpuDual`. `DualTests` compares them with `ReferenceSolver` over a whole step (colored order, 1 and 3 iterations): body poses, C0, penalties, multipliers and the broken flag, on the rope, a fracture scene and random lattices (some joints fracture at once, some never). It then compares the GPU solver's `Step()` with the oracles on the same scenes, including every constraint record read back. Records go to the GPU without the warm start, so the kernel is tested. Comparing a second step with the reference is not possible: the velocity update (stage U) is missing, and a joint that broke would leave the reference's lists and coloring.

## 12. Constraints (step 4: joints and springs)

Contacts are still open. Joints and springs share one record, a constraint, with the kind in the high 8 bits of the flags word (rule P7).

### Record

`ConstraintRecord` is 160 B, ten words (`Runtime/Layout/ConstraintRecords.cs`, `Shaders/Include/AvbdConstraintRecords.hlsl`). It is gathered by id, so it is a multiple of 32 B (rule P3). V and D read nearly every field of it, so there is no hot/cold split (rule P5).

| Word | Type | Content |
| --- | --- | --- |
| 0 | `uint4` | body A (`AVBD_INVALID` = the world), body B, flags and kind, unused |
| 1 | `float4` | anchor on A (a world point if A is the world); joint: linear stiffness, spring: stiffness |
| 2 | `float4` | anchor on B; joint: angular stiffness, spring: rest length |
| 3 | `float4` | linear constraint at the step start (C0); torque arm |
| 4 | `float4` | angular constraint at the step start; fracture threshold |
| 5, 6 | `float4` | penalty of the linear and the angular rows |
| 7, 8 | `float4` | multiplier of the linear and the angular rows |
| 9 | `float4` | reserved, zero |

The spare lanes follow rule P4: stiffness shares a word with the anchor and torque arm with C0 (read by the same stages). A spring uses words 0 to 2 only, so a spring wastes 112 B; a separate smaller spring record is an option for later.

Flags: `HardLinear`, `HardAngular`, `Fracturable` (set when the constraint is created) and `Broken` (owner: stage D). **Infinite stiffness is a flag.** The reference solver uses `+infinity` for a hard constraint and tests it with `isinf`; GPU fast-math is not guaranteed to keep infinity, so the record holds a flag and stores 0 as the stiffness. The behaviors stay the same: a hard constraint has no stiffness cap in the warm start (its penalty is limited by the maximum penalty), a hard constraint uses stabilization and stores its multiplier, and the dual step caps the penalty at the maximum penalty. Fracture works the same way (a flag and a finite threshold, instead of an infinite one).

### Lists and colors

`AvbdConstraintBuffers` (`Runtime/Constraints`) owns four buffers: the records, the per-body lists, the bodies bucketed by color, and the status words. Per-body ranges (offset, count) go into the body arrays' Range array, whose owner is stage G.

- The list is a scalar `uint` buffer (the P1 exception) of constraint ids, grouped by body. Bit 31 is reserved to tag contact entries later (rule P6).
- Only solved bodies get a list. A constraint between a solved body and a fixed one is listed for the solved body only.
- Entries are newest constraint first, so the floating-point sums run in the reference's order.
- Colors come from a greedy first-fit coloring: bodies from the highest slot down, lowest color not used by a colored neighbor. This is the reference's `GraphColoring`, and `ConstraintTopologyTests` checks the two against each other. Fixed bodies and world pins create no edge.
- Bodies are bucketed by color in one list; the CPU knows each color's range, so a color is one dispatch.

**Both are built on the CPU for now** (`ConstraintTopology`, called by `Rebuild()`), because stages G and C do not exist. `Rebuild()` applies the pending body edits and reads body data back (sizes, poses, mass, flags), so it is for when the constraint set changes, not for every frame. `AvbdSolver.Step()` calls it when constraints are pending. Known limits until G and C exist: constraints cannot be removed, removing a body that has constraints is not supported, and a joint that fractures (stage D) stays in the lists and the coloring, so the colored order can differ from the reference's from the next step on (the reference removes a broken joint at the next step's warm start and colors again). The graph color is not written to the Meta record yet (stage C's field).

### Tests

| Test | Needs a GPU | Checks |
| --- | --- | --- |
| `BodyLayoutTests` (constraint cases) | No | Record size and rule P3, flag bits, HLSL constants and struct members against the C# mirror |
| `ConstraintTopologyTests` | No | Lists and coloring against `ReferenceSolver` on the rope, the spring scene and 12 random scenes; fixed bodies; stiffness encoding |
| `PrimalTests` (oracle) | No | `CpuPrimal` against `ReferenceSolver` after one colored iteration, including C0 of every joint |
| `PrimalTests` (GPU) | Yes | The kernel against the oracle on the same scenes; `AddJoint` and `AddSpring` completing the records; a joint with a known closed-form result; static bodies never moving |
| `DualTests` (oracle) | No | `CpuConstraintInit` and `CpuDual` against `ReferenceSolver` after one step (1 and 3 iterations), a hand-computed dual update, fracture |
| `DualTests` (GPU) | Yes | `Step()` against the oracles on the rope, the fracture scene and random lattices, including all constraint records; springs untouched by stage D |

None of the stage V or stage D code or tests has been compiled or run: there was no Unity editor, compiler or GPU in the environment where it was written.
