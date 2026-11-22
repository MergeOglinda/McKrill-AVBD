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
| U | Velocity update | Derives velocity from displacement, snapshots warm-start data. A speed limit is planned (see section 11, stage U) |
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
| P1 | **Vector words only.** Every structured-buffer element is built only from `float4` and `uint4` members. No bare `float`, `float2`, `float3`, `uint2` or `uint3` members, and no explicit padding variables. Exceptions: a homogeneous scalar buffer (`uint` or `float`, stride 4) holding one kind of scalar, such as index lists, sort keys, counters and per-element counts; and a record that is a row of a small table that every thread reads in order (never gathered by slot), whose scalar members (`float`, `uint`, `float2`) fill whole 16 B rows with no implicit padding: `ExplosionRecord` (a `float4`, then `float`, `uint` and `float2`). The test of such a record says so (`allowScalars` in `BodyLayoutTests`). | Mixed member sizes can lay out differently per graphics API. A single scalar type has no layout ambiguity. |
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

Scope: rigid boxes. Contacts, joints and the other buffers are step 4. Eight arrays, indexed by body slot, 256 B per body in total. Each record follows the packing rules in section 2 (float4 and uint4 words only, unused lanes zero). The C# mirrors are in `Runtime/Layout/BodyRecords.cs`, the HLSL in `Shaders/Include/AvbdBodyRecords.hlsl`.

| Array | Stride | Words | Class | Written by | Read by |
| --- | --- | --- | --- | --- | --- |
| State | 64 B | position, rotation, step-start position, step-start rotation | Hot | E (spawn, pose edits), P, V | S, B, N, V, D, U, R |
| Pose | 32 B | position, rotation (a copy of the first two words of State) | Hot | E, P, the group mover, V (every writer of State's pose) | V, D |
| Inertial | 32 B | inertial position, inertial rotation | Hot | P | V |
| Mass | 16 B | principal moments (xyz), mass (w) | Hot | E | V |
| Motion | 48 B | linear velocity, angular velocity, previous linear velocity | Cold | E, U | B, P, Z |
| Shape | 16 B | size (xyz), friction (w) | Cold | E | B, N, R |
| Meta | 32 B | flags and kind, group mask, appearance id; graph color, sleep counter | Cold | E (first word group), C, Z | B, N, P, R |
| Range | 16 B | constraint-list offset, count and number of contact entries | Hot | G | V |

Decisions made here, where section 1 left them open:

- **Compact pose array.** The primal and dual stages read the pose of the neighbors of a joint or spring by slot, and of the 64 B State record they use only the first 32 B. `Pose` is a 32 B copy of position and rotation (`BodyPoseRecord`, `_BodyPose`), so twice as many bodies fit in a cache line (and in the cache). It is a mirror: State stays the one record with the step-start pose, which a manifold needs (and the narrow phase and the velocity stage read), and every kernel that writes State's pose writes the copy too (`WriteFreeSlot`, spawn and pose edits in `AvbdBodyEdits.compute`, `Predict`, the group mover's `Move` and `HoldApply`, and `Primal`, which writes 32 B more per solved body), so the two never differ (`PoseMirror` in the tests checks it after those stages). Readers of the copy: the joint and spring stamps of `Primal` and `Dual` (current pose). `ConstraintInit` (start pose), the manifold stamps and `DualContact` (start pose of both bodies), the narrow phase, the broad phase, the velocity stage and the ray cast still read State. Cost: 32 B per body of memory, 32 B per solved body per primal sweep of writes; saving: half the bytes and half the cache lines per neighbor pose of a joint or spring. Not measured.
- **Step-start pose is interleaved with the pose** in the 64 B State record, so the partner gather in V and D is one record, not two arrays. The record is two sectors; V writes only the first two words.
- **Slot order is spatial when a scene is spawned in one batch.** Finding 8: partner poses are read by slot, so the order of the slots decides how far apart neighbors are in memory. `AvbdBodyBuffers.SpawnBatch(descriptors, handles)` allocates the slots in Z-curve (Morton) order of the body positions (`SpatialOrder`: 21 bits per axis on a cube grid over the bounding box of the batch, ties in input order) and returns the handles in the order of the descriptors, so joints and visuals that hold handles need no change. The order is made once, when the bodies are spawned; nothing is remapped while the simulation runs (a runtime permutation of the iteration order was tried in PR #27 and was slower). It only works on fresh slots: the allocator reuses freed slots first, so a batch spawned into a buffer with holes is placed in Morton order into whatever slots are free. Bodies spawned one by one with `Spawn` keep the order of the calls. Not measured; the performance tests `Scenario5_BrickRing110k` and `Scenario1_Shots2m_PlasticYieldOff` use it, and their `_BuilderOrder` variants are the baseline.
- **Previous linear velocity is in the Motion record** (finding 9). Motion is 48 B. It is read by P and Z by slot through the active list and is cold (once per step), so rule P3 does not require 32 B alignment for it. P3 names V, D and N as the strict gathers.
- **Friction is in the spare lane of the size vector** (rule P4). Its stages (E, N) are a subset of the size stages (E, B, N, R).
- **Mass and the three principal moments share one word.** V reads them together.
- **Appearance id** is in the Meta record (finding 7). Group mask is there too.
- **Free slots**: the Free flag is set, the appearance id and the graph color are `AVBD_INVALID`, the mass is zero and rotations are identity (rule P9).
- **Mass 0 means static.** A static body needs no flag. Static colliders in the sense of section 1 (the separate table) are still step 4.

Matrix changes from this step: E also writes the step-start pose (spawn and pose edits, so a teleport leaves no motion streak and a new body has a valid start pose before P runs), and E writes the previous linear velocity (spawn and velocity edits, so the adaptive warm start sees no acceleration). E never writes the inertial target.

## 5. Capacity calculator

`BodyBufferCapacity` (`Runtime/Buffers/BodyBufferCapacity.cs`) holds the cost and the limits of the body arrays.

- Cost: 256 B per body, so 1,000,000 bodies take 256 MB for the body arrays alone.
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

#### Explosions

`AvbdSolver.Explode(center, radius, impulse)` queues a one-shot radial impulse (at most 16 per step, `AvbdSolver.MaxExplosionsPerStep`; `Explode` returns false when the queue is full). The next step's stage P applies the queue before it computes the inertial target, so contacts and joints resist the blast within the same step. A step with an empty queue runs the plain `Predict` kernel; one with explosions runs the `PredictExplosions` kernel, a variant of the same code compiled with `AVBD_EXPLOSIONS` (`AvbdExplosions.hlsl`), so the plain steps carry no explosion code and are bit-identical to before. The queue is uploaded as `ExplosionRecord` (32 B: center and radius, impulse, seed).

For each solved body (not free, not kinematic, mass above 0) and each explosion, with `r` the distance from the body's center to the explosion's center and `R` its radius:

| Part | Value |
| --- | --- |
| Reach | nothing happens if `r >= R` |
| Linear | `dv = impulse (1 - r/R)^2 / mass` along `(x - c) / r`. At `r <= 1e-4` the direction is a hashed one. There is no upward bias and no shock shell: the direction is purely radial |
| Angular | a small off-center kick. The impulse acts at an offset `random[-1,1)^3 * 0.25 * 0.5 * |size|` from the center of mass; `L = cross(offset, impulse)` and `dw = Rot (Rot^T L / principal moments)` |
| Warm start | `dv` is added to the previous velocity as well, so the adaptive warm start does not read the blast as an acceleration |

The random offset is a PCG hash of `(slot, seed)` with 23-bit integers scaled by powers of two, so the GPU and the oracle agree bit for bit on it, and the solver hands out a new seed for each explosion. Two runs that issue the same explosions in the same order give the same result.

The oracle is `CpuExplosionMath` (called by `CpuPredict.Run`). `PredictTests` checks the arithmetic (impulse over mass, falloff, no upward bias, spin, a body at the center, static and kinematic bodies, an empty queue) and the GPU kernel against the oracle on random bodies, and that an explosion is one-shot.

#### Force fields

`AvbdSolver.AddForceField(in ForceFieldDescriptor)` adds a force that acts on a group of bodies in every step until `RemoveForceField` (`UpdateForceField` changes it, for a thrust or a gravity scale that changes while it runs). At most 16 (`AvbdSolver.MaxForceFields`); a full table returns an invalid handle.

*The label is the group mask.* A body belongs to a group through its group mask (the meta record's `ident.y`, `BodyDescriptor.groupMask`, `AvbdBodyBuffers.SetGroupMask`). It is already in every body and nothing else reads it, so labelling bodies adds no per-body data. A field targets the bodies whose group mask has a bit in common with the field's mask, so 32 groups are available and a body can be in several (it gets the sum of their fields). A new body has all bits set and is therefore in every group: give the bodies the masks of their groups (for an airplane, say 1 for the blocks of the wings and 2 for the rest, and a wing field with the mask 1). The field table is the only GPU memory added: 16 records of 64 B (`ForceFieldRecord`, only `float4` and `uint4` members), created when the first field is added and uploaded when the table changes.

*The kernel.* A step with fields runs the `PredictForces` kernel (`PredictForcesExplosions` with explosions queued too, which apply first, so the fields see the velocity they left), a variant of the predict kernel compiled with `AVBD_FORCES`. A step without fields runs the plain kernel and is bit-identical to before. One thread per body loops over the table (up to its last used slot), skips empty slots and fields of other groups, adds up the acceleration of the fields and adds `a dt^2` to the inertial target, the place of gravity (Eq. 2), so contacts and joints resist it like gravity. The warm start does not include it: the adaptive weight reads the acceleration of the last step from the velocity change, so a body held up by lift has no gravity in its warm start. Static and kinematic bodies are skipped like in the rest of stage P.

| Kind | Acceleration |
| --- | --- |
| Constant | `F / m`, with the force F in N in the world's frame or the body's. It acts on each body of the group (it is not a total shared among them) |
| GravityScale | `g (s - 1)`, so that gravity plus the field is `s g`: 0 is weightless, 1 changes nothing, negative falls up. Uses the solver's gravity |
| Drag | `F / m` with `F = -(c1 + c2 \|v\|) v` for the velocity `v` of the body relative to the wind. The acceleration is limited to `\|v\| / dt`, so one step cannot reverse the relative velocity |
| Spring | pulls the group as a whole toward a target `T` with stiffness `k` and damping ratio `z`: every body gets `a = k (T - C) / m - 2 z sqrt(k / m) v`, where `C` is the centroid (the mean position) of the bodies of the group at the start of the step. The same force `k (T - C)` acts on each body, so the group moves as one and keeps its shape; a point attractor per body would collapse it onto the point. The spring part is limited to `\|T - C\| / dt^2` (never past the target in one step), the damping part to `\|v\| / dt`. With a line direction `u` (unit, in `axis`) and a gain `g` (`coefficients.z`) the stiffness is `k (1 + g d)`, where `d` is the distance of the centroid from the line through `T` along `u` (`d = \|(C - T) - ((C - T) . u) u\|`); a grab uses the line of sight, so the pull grows the farther the group is from the center of the screen |
| Wing | a flat plate of area S with normal `n` in the body's frame, density of air `rho`, at relative velocity `v` with direction `vhat` and `q = 0.5 rho S \|v\|^2`: `F = -q (k (vhat . n) n + cd0 vhat)` with the lift slope `k` and the drag `cd0` at no angle of attack. The part across the flow is lift and the induced drag comes from the same term; the same limit as drag applies. There is no stall: the force keeps growing with the sine of the angle of attack |

*The centroid of a spring.* A step with a spring first runs two small kernels of `AvbdPredict.compute` (`ForceFieldClear`, `ForceFieldAccumulate`), before the predict kernel: for each spring field, one pass over the bodies adds the offsets `position - target` of the bodies of its group (not static, not kinematic, not free: the bodies the predict kernel moves) with `InterlockedAdd`, and counts them. The offsets are summed as integers in 1/1024 m (`AVBD_FORCE_FIXED_POINT`), so the sum does not depend on the order the threads add in and the oracle can reproduce it exactly; the sum of the offsets of a group must stay below about 2 million m. The sums are 16 fields times 4 words (256 B, created with the first field). A step without a spring does not run the two kernels.

*Labelling bodies near a point.* `AvbdGroupLabeler.Label(center, radius, setBits, clearBits)` (kernel `Label` of `AvbdGroupLabel.compute`) changes the group mask of every body whose box is within `radius` of `center` to `(mask | setBits) & ~clearBits`, writing `ident.y` of the meta record on the GPU; `ClearAll(bits)` clears bits everywhere. Together with `AvbdRaycaster` this selects a group of bodies around a ray hit without the CPU reading any body. `GroupLabelerTests` checks the distances by hand.

*Moving a group directly.* `AvbdGroupMover.Move(mask, fields, translation, linearVelocity, angularVelocity)` (kernel `Move` of `AvbdGroupMove.compute`) is the other way to act on a group: no force, no kinematic flag. For every body whose group mask shares a bit with `mask` it adds a translation to the position and/or sets the linear and angular velocity, and nothing else: mass, moments of inertia, shape, friction, flags and the mask stay, so the body keeps responding to contacts and joints as the same body. A position change is a teleport like the `SetPose` edit (the step-start pose follows, so the step does not read the jump as a motion), a linear velocity change sets the previous linear velocity too, like `SetVelocity`. Free slots and static bodies (mass 0) are skipped; kinematic bodies are moved. The inertial target is made by the predict kernel from the new pose and velocity, so it needs no edit. *Holding a group.* `Move` adds a distance, so a group that a step has pulled or pushed (gravity moves a body by g dt^2 in a step even from rest) stays off by what the steps did: a grab that moves the group by the distance of the camera every frame sinks, and is wrong when the solver runs a different number of steps than the frames the move was made for. `BeginHold(mask, point)` and `Hold(mask, point, linearVelocity)` place the group at absolute positions instead, each body on its own. Two kernels of `AvbdGroupMove.compute`: `HoldCapture` (only in `BeginHold`, dispatched over every slot of the buffers) writes, for every slot, a float4 in a buffer of the mover: the offset `position - point` of the body and `w = 1` if the body is in the group (the bodies `Move` moves), else zero; `HoldApply` (only in `Hold`, over the high-water mark) puts every body of the group whose `w` is 1 at `point + offset` (a teleport: the step-start pose follows) and sets its linear and previous linear velocity to `linearVelocity`. The rotations, the angular velocity and everything else stay. A body that joined the group after the capture (`w` is 0) is not held, and a hold before a capture does nothing. No readback: the group and the offsets stay on the GPU, at 16 B per body slot. The oracle is `CpuGroupMove.HoldCapture`/`Hold`.

The oracle is `CpuGroupMove`; `GroupMoverTests` checks it by hand and against the oracle.

*Limits.* Forces act at the center of mass, so there is no torque from a field: a plane turns because its wing blocks sit away from the center of gravity of the plane and are joined to it. Forces are explicit in the velocity at the start of the step, which is why drag and wing are limited. The oracle is `CpuForceFieldMath` (called by `CpuPredict.Run`). `ForceFieldTests` checks each kind against numbers worked out by hand, the groups (also static and kinematic bodies, and bodies in several groups), and the GPU kernel against the oracle on random bodies with all kinds, an update, a removal and explosions in the same step.

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

**Fused with the sweeps (`AvbdSolver.FuseDualUpdate`, default on; keyword `AVBD_FUSED_DUAL` in the primal kernel).** The update of one joint is a function in `Include/AvbdJointDual.hlsl` (`AvbdJointDual`, which updates a record held in registers, and `AvbdJointDualStore`, which writes back only the words that changed), used by the `Dual` kernel and by the primal kernel. With the keyword, the dual update after sweep *k* of a step is not a dispatch of its own but happens at the start of sweep *k + 1*, in the thread of the first body of the joint that is solved in that sweep. That body is the one of the lower color, and when its dispatch runs, neither body of the joint has moved in sweep *k + 1* (a body writes its pose at the very end of its own thread group; the other body's color has not run), so the joint is evaluated at exactly the poses of the end of sweep *k*, which is what the separate dispatch saw. The thread updates the record (the same code, so the result is the same), stores it, writes the id of the sweep into `reserved.w` of the record (the sweep ids are 1, 2, ... for the sweeps that run an update; a mark is the bits of the id as a float word) and stamps the updated joint. The other body of the joint, in a later dispatch of the same sweep, finds its id in the mark and only stamps; a broken joint is skipped by both. The first sweep of a step runs no update (the primal kernel's `_DualSweep` is 0). Two bodies of one color never share a joint, so there is no race; the rare coloring that gives two neighbors one color (`ColorForce`, see stage C) can update such a joint twice in one sweep, which is as inexact as solving the two bodies in parallel already is. After the last sweep one `Dual` dispatch (`DispatchFinalDual`) does the update of that sweep, clears the marks (the records leave a step with `reserved.w = 0`, so nothing outside the step sees them) and, for a joint that the last sweep did not mark, which means that no solved body is on it and its poses did not change, applies the updates of all the iterations one after the other, as the separate dispatches would have. That makes the fused step the same as the unfused one for every joint, to rounding, with 1 `Dual` dispatch instead of one per iteration: with 30 iterations, 29 fewer passes that each read every joint record and both of its poses and write four of its words. The cost is more code in the register-bound primal kernel; the switch is there so that a trace can compare both. Contacts are not fused: `DualContact` still runs after every sweep. A step is now: apply edits, rebuild constraints if needed, P, `ConstraintInit`, then per iteration the primal sweep (one dispatch per color) and, without the fusion, `Dual` (one dispatch); with it, the sweeps and one `Dual` at the end.

**Flags.** `Broken` is owned by `Dual`: when a fracturable joint's angular multiplier exceeds the threshold (`|lambdaAng|^2 > fracture^2`, tested for every fracturable joint, whatever its penalty), the kernel zeroes its penalties and multipliers and sets the flag. A broken joint has zero penalty, so the primal stage adds nothing for it, and both stage D kernels skip it. It stays in the lists and the coloring (see section 12). `Fracturable` is a flag plus a finite threshold; a joint without it never breaks.

**Encoding.** The stiffness caps follow the flags, not infinity: `HardLinear` or `HardAngular` selects "no stiffness cap" (the penalty maximum), anything else uses the stored finite stiffness. A non-hard angular stiffness of 0 disables the angle row, because the warm start caps its penalty at 0.

**Check.** The oracles are `CpuConstraintInit` and `CpuDual`. `DualTests` compares them with `ReferenceSolver` over a whole step (colored order, 1 and 3 iterations): body poses, C0, penalties, multipliers and the broken flag, on the rope, a fracture scene and random lattices (some joints fracture at once, some never). It then compares the GPU solver's `Step()` with the oracles on the same scenes, including every constraint record read back. Records go to the GPU without the warm start, so the kernel is tested. Comparing more than one step needs the velocity update; stage U adds it, and `VelocityTests` compares several steps.

### Stage U: velocity update

`AvbdVelocity.compute`, kernel `Velocity`: one thread per slot over `[0, HighWaterMark)`, once per step after the last iteration. A thread reads the pose and the step-start pose and writes only its own Motion record, so there is no ordering and no atomics.

| Body | Previous linear velocity | Velocity |
| --- | --- | --- |
| Free slot | untouched | untouched |
| Live, kinematic or mass 0 | set to the velocity | kept (the user or an edit set it; nothing moved the body) |
| Live, solved | set to the old velocity | `(x - x_start) / dt` and `QuatDifference(q, q_start) / dt` (BDF1, upstream's quaternion difference, no sign flip) |

The previous velocity is what stage P reads for the adaptive warm start, so a step's `prev` is the velocity before this update. This is the reference solver's loop, which snapshots every body; the only difference is that free slots are skipped.

**Speed limit not implemented.** The stage table lists a speed limit, but upstream has none and the reference solver has none, so adding one now would break the parity tests. It would be one more setting (a uniform) and a clamp after the division, off by default.

**Full step.** `AvbdSolver.Step()` is now complete for the stages that exist: apply edits, rebuild the constraint lists and colors if needed (lists on the CPU, then stage C), P, `ConstraintInit`, per iteration a primal sweep and `Dual`, then U. Two behaviors the driver needed:

- **Without constraint buffers** the solver creates an empty `AvbdConstraintBuffers` of its own at the first step. Before, it skipped the primal sweeps, so a body without constraints never reached its inertial target and could not fall. `AvbdSolver` is now `IDisposable`; `Dispose` releases only the buffers it created itself.
- **Bodies spawned, removed or changed later.** The color buckets and the lists depend on which bodies are solved. `AvbdBodyBuffers.SolvedSetVersion` increases when an applied batch contains a spawn, a remove, a change of shape (mass) or of the kinematic flag, and `Step()` calls `Rebuild()` when the version differs from the one the lists were built for. Before, a body spawned after the last `Rebuild()` was predicted but never solved. Pose, velocity, friction, group and appearance edits do not trigger a rebuild, because they do not change the lists.

**Check.** The oracles are `CpuVelocity` and `CpuStep` (P, D warm start, iterations, U, in the GPU's order). `VelocityTests`:

| Test | Needs a GPU | Checks |
| --- | --- | --- |
| Oracle, by hand | No | Velocity from a known displacement, angular velocity of a known rotation, static, kinematic and free slots |
| `Oracle_Step_MatchesTheReferenceSolver` | No | `CpuStep` against `ReferenceSolver` (colored) over 60 steps on the rope and the spring scene: poses, linear, angular and previous velocity at steps 1, 2, 10, 30 and 60 |
| `Oracle_Step_FollowsTheClosedFormOfAFreeFall` | No | A body without constraints: `v = n g dt`, `y = y0 + g dt^2 n (n + 1) / 2` |
| `Gpu_Velocity_MatchesTheOracle_OnRandomBodies` | Yes | The kernel against the oracle on random bodies with statics, kinematic bodies and free slots |
| `Gpu_Step_*` free fall, late spawn, static and kinematic | Yes | The same closed form through `Step()` without constraint buffers; a body spawned after the first step is solved; static and kinematic bodies keep pose and velocity |
| `Gpu_Step_MatchesTheReferenceSolver` | Yes | `Step()` against `ReferenceSolver` (colored) over 60 steps on the rope and the spring scene: poses, velocities and previous velocities at the same steps, and no failed solves |

**Tolerances.** Position 2e-3, rotation 2e-4, velocity 1e-2 (relative to max(1, |v|)). They were chosen from a sensitivity run of upstream's C++ solver on both scenes: with every initial position nudged by 1e-6, the largest drift after 60 steps was 3e-5 in position, 6e-6 in rotation and 1.4e-4 in velocity on the rope, and the spring scene did not amplify (1.4e-6 throughout). The bounds are 30 to 70 times that. Rounding differences between the GPU and the CPU are far below the nudge, but this has not been measured on a GPU.

**Limits.** The two scenes have no contacts (the reference creates empty manifolds for the far ground; they affect nothing). A joint that fractures stays in the lists, so the colored order can differ from the reference's after the step in which it breaks (section 12); the parity scenes have no fracture.

### Stage C: graph coloring on the GPU

`AvbdColoring.compute` colors the graph of solved bodies and buckets the bodies by color, replacing the CPU coloring of `ConstraintTopology`. The graph has one node per solved body and one edge per constraint between two solved bodies. Fixed bodies and world pins create no edge. The adjacency is read from the per-body lists (stage G).

**Algorithm: Jones-Plassmann.** Every body has a priority, a strict total order. A body colors itself once every neighbor of higher priority has a color, and takes the lowest color none of its neighbors has. All bodies that are ready color in the same round; one round is one dispatch. The result does not depend on the thread order inside a round: a body only sees final colors of its higher-priority neighbors, and a lower-priority neighbor cannot be colored before it. The result is therefore the sequential greedy first-fit coloring of the priority order. The color is written to `Meta.state.x`; a solved body without a color holds `AVBD_COLOR_PENDING`, a body that is not solved holds `AVBD_INVALID`.

**Priority order.** `GraphColoringOrder.SlotOrder`: a higher slot colors first, which is exactly the reference solver's coloring, so the colored sweep is the same sequence of updates as `ReferenceSolver` in `Colored` mode and the GPU solver can be compared with it. The cost is the number of rounds, the length of the longest chain with falling slots: a rope built in slot order needs one round per link. `GraphColoringOrder.Hashed` orders by a hash of the slot: a few rounds for any chain, a valid but different coloring (so different from the reference). `AvbdConstraintBuffers.ColoringOrder` selects it.

| Kernel | What |
| --- | --- |
| `ColorReset` | Zeroes the work buffer (200 words: 4 control, 64 counts, 64 cursors, 64 starts, 3 statistics, 1 unused) |
| `ColorInit` | Marks every slot: `PENDING` if solved (not free, not kinematic, mass above zero), else `INVALID` |
| `ColorBegin` | One thread; zeroes the counter of bodies still waiting, before the last round of a batch |
| `ColorRound` | One round, 8 threads per body (`AVBD_COLOR_LANES`, adjacent threads of a group). Lane `l` reads the colors of the `l`-th chunk of the body's neighbor entries once and reports (a) whether a neighbor that colors before the body is still pending and (b) which of the colors 0 to 31 the neighbors have; the reports are or-ed in groupshared memory and lane 0 decides. A pending body waits if any neighbor that colors before it is pending (and counts itself as waiting); otherwise it takes the lowest color no neighbor has (a body whose neighbors have all of the first 32 colors looks at the next 32 with a serial walk) and raises the color count (control word 2) to its color + 1 with an atomic max. More than 64 colors sets an overflow word |
| `ColorCount` | One atomic add per solved body into the count of its color |
| `ColorPlan` | One thread, after `ColorCount`: for each of the 64 colors, the start in the bucket array (stored as the start and as the bucket cursor) and the arguments of an indirect dispatch over its bodies |
| `ColorBucket` | Writes each solved slot to the next free entry of its color, at the cursors `ColorPlan` set |
| `ColorKeep` | Persistent coloring: like `ColorInit`, but a solved body keeps its color if it is below the cap (`_ColorCap`), and is `PENDING` otherwise |
| `ColorMark` | Persistent: flags a colored body that has the color of a neighbor that colors before it. Reads colors only, writes one flag per body to a scratch buffer, so the flags do not depend on the order of the threads |
| `ColorRelease` | Persistent: a flagged body becomes `PENDING` (counted in statistics word 196) |
| `ColorRoundPlan` / `ColorForce` | Persistent: `ColorRoundPlan` (one thread) gives the next extra round its groups if control word 0 says bodies are still waiting (none otherwise) and clears the counter; the extra round is `ColorRound` dispatched indirectly. `ColorForce` colors what is still without a color after all rounds, without waiting |

**Host side.** `AvbdGraphColoring` runs the rounds in batches (the first of 16, the later ones of 8: a round after the coloring is finished ends at once, so a long first batch is cheap and spares the second wait for the GPU in the usual case) and reads back the control words after each batch: whether bodies are still waiting, and, once none is, the number of colors. That is the only synchronization. The size and the start of every color are never on the CPU: `ColorCount` and `ColorPlan` leave them in the work buffer (counts at word 4, starts at word 132) and the dispatch arguments in a second buffer (`Args`, `IndirectArguments | Raw`, written through a `RWByteAddressBuffer`), and `ColorBucket` fills the buckets from the cursors `ColorPlan` set. The previous version read the counts back, computed the starts on the CPU and uploaded them as cursors.

**Indirect sweep.** The primal sweep is one dispatch per color and iteration, in order, because a body reads the poses the earlier colors just wrote. `AvbdSolver` sets `_Color` and calls `DispatchIndirect(args, color * 12)` for each of the `ColorCount` colors. The kernel reads the size and the start of its color from the work buffer (`_ColorWork`), so nothing about a color's size reaches the CPU. A color of at most 65,535 groups (4.19 million bodies) is a dispatch of `x` groups by one row; a bigger one is laid out in rows of 65,535 groups (`y` rows), and thread `(x, y)` handles entry `y * 65535 * 64 + x`, the last group possibly overshooting, which the size check absorbs. `AvbdGraphColoring.GroupsFor` is the C# mirror of the arithmetic. A color without bodies gets a `0 x 1 x 1` dispatch, which does nothing. A graph that needs more than 64 colors (for example 65 bodies that are all joined to each other) makes `Rebuild()` throw `InvalidOperationException`.

**Cooperative sweep.** A color is small next to the GPU (a few hundred bodies at 6500 bodies), and a body walks its whole constraint list (a body in a pile has 20 to 30 entries, contacts first) in one thread, so a sweep dispatch was bound by that serial walk, not by the number of bodies. The primal kernel therefore gives a body `AVBD_PRIMAL_LANES` (8) adjacent threads of a group. Lane `l` stamps the list entries `l, l + 8, ...` into its own partial `PrimalSystem` (lane 0 also adds the inertia term), every lane packs its system into 27 terms (the lower triangles of the linear and angular blocks, 6 each, the 3x3 cross block and two 3-vectors: the solver reads only the lower triangle, so nothing else is needed), the lanes of a body add them up in two steps so that groupshared memory holds half of the partial systems (lanes 4 to 7 write theirs, lanes 0 to 3 add them to their own and write the result), then lane `l` sums the 4 slots of the body for the terms `l, l + 8, ...` in a fixed order and writes them to a second groupshared array, `gsSums`, with the body as the fast index, and lane 0 solves the 6x6 system once, in place in `gsSums` (the LDL^T runs on groupshared memory, not on a local 36 + 6 float array that would stay in registers), and writes the pose (only its two words, not the start-of-step pose). A group of 64 threads solves 8 bodies; `groupshared` use is 27 x 32 x 4 + 27 x 8 x 4 = 4.3 KB (it was 9.5 KB with 33 terms for all 64 threads). **Loads in flight.** Everything a body needs (flags, mass, pose, list range, and the inertial target for lane 0) is requested together right after the slot is known, and a lane takes `AVBD_PRIMAL_BATCH` (2) list entries at a time: the entries, then the joint and spring records among them, are requested before the first of the batch is stamped, so their latencies overlap. A batch costs `AVBD_PRIMAL_BATCH` records of registers (160 B each); set the constant to 1 for the one-at-a-time loop if the registers cost more occupancy than the loads gain. A body with at most 8 entries gives each lane one entry, so the batch changes nothing for it. No thread returns before the three barriers (a skipped body only stamps nothing). `ColorPlan` sizes the dispatch as `count * 8` threads (item `i` is lane `i % 8` of the color's entry `i / 8`), `AvbdGraphColoring.GroupsFor` mirrors it, and the 2D row layout is unchanged (it now counts items, not bodies). The sum adds the entries in another order than `CpuPrimal`, so the GPU agrees with the oracle to rounding, within the existing `PrimalTests` tolerances; it does not change which constraints a body sees or which poses it reads.

**Order inside a color.** The bucket entries of one color are in the order the threads got their atomic cursor, so it is unspecified and can differ between runs. It does not change the result of a sweep: the bodies of one color share no constraint, each writes only its own pose and reads the poses of other colors. The sweep stays deterministic. (The CPU coloring puts the highest slot first; tests compare buckets as sets.)

**When it runs.** `Rebuild()` colors from scratch and waits (the loop above): when constraints were added, the set of solved bodies changed (`SolvedSetVersion`) or the order was changed. With contacts, `RefreshContactTopology` colors again every step, because the contacts change the graph; that is the persistent coloring below, which never waits for the GPU.

**Persistent coloring (`AvbdGraphColoring.RunPersistent`).** With contacts the old path stalled the host every step: it read back the control words after each batch of rounds, and the CPU could not queue the next stage until the answer came. The graph of one step is the graph of the step before with a few contacts added and removed, so most colors are still valid, as in three-avbd. A persistent step is:

1. `ColorReset`; then `ColorKeep` (colors below the cap stay, the rest become `PENDING`), `ColorMark`, `ColorRelease`. Of two neighbors with the same color the one that colors later (`ColorsBefore`) is released, so the bodies that keep their color are a proper coloring. Mark only reads and Release only writes, which makes the flags independent of the thread order.
2. A fixed number `K` of `ColorRound` dispatches (the last one preceded by `ColorBegin`). They color the `PENDING` bodies like a coloring from scratch would, with the kept bodies as colored neighbors.
3. Up to `AvbdGraphColoring.ExtraRounds` (8) more rounds, each preceded by `ColorRoundPlan` and dispatched indirectly with no groups when nobody waits, so a step that needs more rounds than the tuner expected still ends in a proper coloring at the price of a few empty dispatches. `ColorForce` then colors any body that is still without a color, without waiting; this happens only if a chain of falling priorities is longer than all the rounds (a hash of the slot makes that very unlikely), may give two neighbors one color, and the next step that keeps colors releases the lower one. An earlier version finished with one 256-thread group with barriers in a loop (`ColorFinish`); it was replaced because the editor froze on the first persistent step.
4. `ColorCount`, `ColorPlan` (which now also writes the number of colors, the highest color with a body plus one, into control word 2), `ColorBucket`.
5. An `AsyncGPUReadback` of the work buffer, issued at the end of the step (at most three in flight).

*Determinism and the number of rounds.* Steps 2 and 3 are Jones-Plassmann on the released bodies: a body takes the lowest color its colored neighbors do not have after every neighbor that colors before it is colored. Colored neighbors are the kept ones and the ones of higher priority. So the result is a function of the kept colors and the priority order only; it does not depend on `K`, on the thread order, or on when the statistics arrive. `K` only moves work between the first rounds and the extra ones (except in the `ColorForce` case above). `CpuGraphColoring.Persistent` is the oracle, and the tests compare the GPU with it for `K` = 1, 2 and the tuned value.

*The cap.* The primal sweep dispatches `ColorCount` indirect dispatches per iteration, and with kept colors the host does not know the number of colors without a readback. `AvbdConstraintBuffers.ColorCount` is therefore a cap: the number of colors the host dispatches, a few above the colors in use (`ColoringTuner.Headroom`, 3); the colors above the highest in use are empty dispatches (`0 x 1 x 1`). A body that finds no free color below the cap takes the last color below it and is counted in control word 1 (forced); the sweep then solves it in parallel with the bodies of that color, which is less accurate but stable, until the cap goes up. Colors at or above the cap are released by `ColorKeep` when the cap comes down.

*Async tuning (`ColoringTuner`, pure C#).* The statistics (released, left over after the rounds, forced, number of colors, last round in which a body took a color) arrive a few frames late. The tuner raises at once and lowers slowly: leftover bodies double the rounds of that kind of step; with no leftover the rounds follow the largest last round of the last 8 samples plus 2 (never below 2); the cap follows colors + 3, goes up at once, and a forced body raises it by 4 over the cap it ran with. The cap comes down only after a step from scratch, whose number of colors has no drift. Steps from scratch and steps that keep colors are tuned separately. The numbers are a starting point, not measured.

*Drift.* A kept coloring never lowers a color, so the colors creep up as contacts come and go. The first persistent step after a `Rebuild` and every `FreshInterval` (64) steps start from `ColorInit` in the same pipeline: a coloring from scratch, still without a readback. After `Rebuild()` the cap is the exact number of colors plus 3 (that run did wait).

*What is not the same.* The persistent coloring is a proper coloring, but not the coloring of the oracle `CpuGraphColoring.Color` and not the reference's, because a body keeps its color while it can. `GraphColoringOrder.SlotOrder` promises the reference's coloring, so it always colors from scratch and waits; so does `PersistentColoring = false`.

**Check.** `CpuGraphColoring` is the oracle: bodies visited in falling priority, each takes the lowest color not used by a colored neighbor. `GraphColoringTests`: the oracle in slot order equals `ConstraintTopology`'s coloring (rope and 12 random scenes); the hashed oracle is a valid coloring; the GPU equals the oracle exactly in both orders on the rope, the spring scene and random lattices (color of every body, number of colors, bucket contents); the hashed order needs under half the rounds of the slot order on a chain of 160 links; recoloring after a spawn, an added constraint and a body becoming kinematic; bodies without constraints; and the 65-body clique that exceeds 64 colors. `PrimalTests` and `VelocityTests` now run on the GPU coloring. For the persistent coloring: the oracle `Persistent` keeps a proper coloring, repairs a tampered one and forces the last color under a small cap; the GPU equals it after a rebuild, after the colors were set to 0 and 1 by slot (a great many clashes) with the tuned rounds, 2 rounds and 1 round (so the extra rounds have work), under a cap that is too small (5-body clique, cap 3), at the fresh interval, and with `PersistentColoring` off; the statistics come back from the readback; `ColoringTunerTests` drive the tuner with samples.

Not part of this stage: removing the last readback, and temporal coherence. The host still needs to know when the rounds are finished and how many colors there are. Replacing that means dispatching a bounded number of rounds without knowing the count (empty indirect rounds cost little) and finishing a rare overflow some other way; the sweep is already indirect, so the color count could then also be replaced by the 64 colors' arguments. A joint that has fractured is left out of the lists by stage G, so it is not in the graph either.

### Stage B: broadphase

`AvbdBroadphase.compute` finds the pairs of bodies whose bounding spheres overlap and whose boxes are not visibly apart (below). It is the producer of the pair list. The narrowphase (stage N) will be its only consumer; today it still tests all pairs itself (section 13), see the relation below. The sphere is centered on the body position and has the radius `|size| / 2` (half the box diagonal), like the reference solver: two bodies are a pair when `|dp|^2 <= (rA + rB)^2` and no joint or spring joins them. A body at a non-finite position makes no pairs.

**Pair record.** `PairRecord`, 16 B, `uint4`: x = body A, y = body B, z = pair kind, w = 0. Body A is the body with the higher slot (the reference's pair loop makes the newer body body A). Slots, not handles.

**Pair kinds (the bins).** The pair list is one buffer with two bins, each with its own capacity and counter:

| Bin | Kind | Pairs |
| --- | --- | --- |
| 0 | `DynamicDynamic` | both bodies solved (not free, not kinematic, mass above zero) |
| 1 | `DynamicStatic` | one body solved, the other live but not solved (mass 0 or kinematic) |

Two static or kinematic bodies never make a pair. A pair that does not fit its bin is counted but not stored, so a counter above the capacity tells the host the bin overflowed (`AvbdBroadphase.Overflowed`). Nothing else is lost.

**Deviation from the scope paragraph.** Section 1 says static colliders never enter the pair list and are tested directly in the narrowphase. This stage puts them in bin 1 instead, for two reasons: the reference solver's manifolds include pairs with static bodies, so the parity tests need them, and the narrowphase then has one input to read. The first version tested every solved body against every static body (a brute-force loop over a compacted list), which cost the product of the two counts and dominated the broadphase at 20000+ bodies with thousands of static bodies (the bottom rows of the walls, floors made of tiles). **Static bodies that fit a grid cell now share the grid** (next paragraph); only the static bodies larger than every solved body use the list.

**Grid (three-avbd's, PR #36).** A counting-sort grid in CSR form, with a fine and a coarse level and a list of large bodies. The grid is rebuilt every run; kinematic bodies that move need nothing special.

- *Who is in the grid.* Every live body with a finite radius: solved bodies and static or kinematic ones. A body is *large* when its radius is above the cut (below); large bodies are not in the grid but in the large list (`_LargeList`, at most `AVBD_BP_MAX_LARGE` = 64 of the largest bodies), and every body is tested against it.
- *Classification on the GPU, no readback.* three-avbd sorts the radii on the CPU: the median sets the cut (twice the median radius), the 64 largest bodies are the most that can be large, the largest radius below the cut sizes the cells. The host here keeps no copy of the radii, so `BpHistogram` counts the radii of all live bodies and of the solved bodies into log histograms (16 bins per octave, 640 bins from 2^-20 m to 2^20 m: `bin = floor(log2(r) * 16) + 320`), and one thread, `BpClassify`, reads the histograms and decides. The cut bin is the median bin plus 16 (a factor 2), raised to the bin of the 65th largest body when there are more than 64 bodies. The *small bin* is the highest non-empty bin at or below the cut. Bodies above it are large. Bin edges are rounded up by 0.1% (`BinUpper`), so a cell is at most 4.4% larger than the exact radius would give. The CPU model is `CpuBroadphase.Classification`.
- *Two levels.* The solved bodies are what probes, and the ones that move are often much smaller than the static pieces (a pile of small bricks on large tiles). The fine bin is the highest non-empty bin at or below `min(smallBin, smallBin(solved bodies) + 16)`. If `fineBin + 16 < smallBin` (the fine bodies are at least a factor 2 smaller than the coarse ones) there are two levels: bodies whose bin is at or below the fine bin are in the fine level, the other small bodies in the coarse level. Otherwise there is one level, the coarse one. A cell is `2 * BinUpper(bin) * AVBD_BP_CELL_MARGIN` (1.01) wide, floored at 1e-4, so two overlapping spheres of the level are in the same or in adjacent cells.
- *Table.* `tableSize` is a power of two between 1024 and 2^20, at least twice the body capacity. The cell of a body is `floor(position / cellSize)`; the table index is a hash of the cell coordinates *and the level*. Bodies of different cells that share an index are told apart by the cell stored beside the list (`_BodyCell.xyz`), so a probe drops them without touching the body.
- *Counting sort.* `BpCount` counts the bodies per table index (atomic), the scan turns the counts into starts, `BpScatter` places each body at start + cursor. `_GridStart` has `tableSize + 1` entries, so a cell's bodies are `_GridSorted[_GridStart[i] .. _GridStart[i+1])`. The scan is three kernels: `BpScanBlock` scans blocks of `AVBD_BP_SCAN_GROUP` = 1024 cells and stores the block sums, `BpScanTop` scans the block sums (at most 1024 blocks), `BpScanApply` adds the block offsets.
- *Who probes, and who owns a pair.* Only solved bodies probe. A solved body looks at the 27 cells around it on its own level and on every level above it (a fine body also looks in the coarse level; the coarse cell is large enough for the fine body's neighbors there). Two bodies of the same level that both probe: the pair belongs to the higher slot. A probing body and a body of the coarser level: the pair belongs to the finer one, which is the one that walks the coarser level. A static body does not probe, so `BpPairsFixed` finds the pairs that no solved body's own walk reaches: the small static bodies of the fine level against the solved bodies of the coarse level, and the large list against every solved body. Large solved bodies are tested against everybody in the grid by the large-list loop in `BpPairs`; a large body does not probe the grid.
- *Pair set.* Unchanged: sphere test, `BoxesSeparated`, `Constrained`, a pair that exists once, `(max slot, min slot)` as body A and B. `CpuBroadphase.Pairs` (brute force) is still the oracle.

Control words (`_BpControl`, `AVBD_BP_CONTROL_WORDS` = 1296): 0 upper bound of the largest small radius (float bits), 1 large count, 2 and 3 the pair counters of the two bins, 4 small bin, 5 fine bin, 6 two levels, 7 coarse cell (float bits), 8 fine cell, 9 live count, 10 solved count, 11 to 15 spare, 16 to 655 histogram of all live bodies, 656 to 1295 histogram of the solved bodies. The host reads the first 16 words only (`AvbdBroadphase.ReadClassification`), the narrowphase reads words 2 and 3 as before.

`_BodyCell` (`uint4`) per body: xyz = the cell coordinates of the body on its level (int bits), w = flags: `AVBD_BP_COARSE_FLAG` (1) the body is in the coarse level, `AVBD_BP_LARGE_FLAG` (2) it is in the large list, `AVBD_BP_STATIC_FLAG` (0x80000000) it is not solved; `AVBD_INVALID` when the body is in neither (free, non-finite).

| Kernel | What |
| --- | --- |
| `BpReset` | Zeroes the control words |
| `BpClearGrid` | Zeroes the grid starts and cursors |
| `BpHistogram` | Counts the radii of live bodies and of solved bodies into the two histograms |
| `BpClassify` | One thread: median, cut, small bin, fine bin, levels, cell sizes, large count; appends no pairs |
| `BpCount` | Cell and level of every grid body (stored in `_BodyCell`), atomic count per table index; puts large bodies into the large list |
| `BpScanBlock`, `BpScanTop`, `BpScanApply` | Exclusive prefix sum of the counts |
| `BpScatter` | Writes each grid body into its cell's range |
| `BpPairs` | The probe of every solved body (27 cells on its level and above) and the large list; appends to bin 0 and bin 1 |
| `BpPairsFixed` | The small static bodies of the fine level against the coarse solved bodies, and the large list against the solved large bodies; appends to bin 1 |

Not measured. Cost: the histogram and the classification are a few hundred bytes of atomics and one thread; the scan is `tableSize / 1024` blocks. A scene with only a few thousand bodies pays the 1024-cell minimum, which is below the cost of the dispatches.

**Box rejection.** The bounding sphere of a box overlaps another body's sphere far more often than the box overlaps the box (a long bar and its neighbor, a ground slab under every body), and every such candidate costs a manifold lookup and a full collision test in `NpAllocate`, each step, until it gets a manifold (which it never does if the boxes stay apart). So a candidate that passes the sphere test must also pass `BoxesSeparated`: the world bounding boxes of the two boxes overlap, and none of the six face axes (three of each box) separates them. The test separates only when the gap is above `AVBD_BP_SEPARATION_PAD` (1 mm). Why it drops only pairs the narrowphase drops: the narrowphase's SAT returns no contact as soon as any of its axes shows a gap above zero, and a face axis is one of those axes; a gap between the world bounding boxes means the boxes are disjoint, which the 15 axes find too. The pad covers the different rounding of the two computations and costs only a few candidates. Both use the poses of the step start (the broadphase runs before predict, the narrowphase reads the start pose). A NaN never separates. `BpPairs` and `BpPairsFixed` build the box of the searching body once and the box of a candidate only after its sphere test passed. The idea and the measurement (-15% on a brick ring, -2% to -5% on other scenes, in a different engine) are from `three-avbd`'s broadphase; see `Docs/design/three-avbd-comparison.md` (PR #31). Nothing here has been measured on this solver.

**Constraint exclusion.** A pair is skipped when a joint or a spring joins the two bodies. The test reads the constraint list (stage G) of the solved body of the pair, so it costs as many steps as that body has constraints. Pairs with a world pin are never excluded. This is the rule of the CPU oracle (`AvbdBody.ConstrainedTo`) and the only place the GPU applies it (the narrowphase of section 13 reads the list): a fractured joint (`Broken`) excludes its pair for as long as it is in the lists. Without contacts nothing removes it; with contacts the step drops it from the lists one step after it broke (section 13, fractured joints), which is when the reference removes it. Contact records (kind `Contact`) are in the lists too, since the narrowphase writes them, and are **not** an exclusion: a pair with contacts has to stay a candidate for as long as the boxes touch, which is how the reference's persistent manifolds behave. `IgnoreCollision` has no GPU record, in neither stage.

**Relation to the narrowphase of section 13.** The narrowphase reads the pair list: its `NpAllocate` kernel gives a manifold slot to every pair of the list that has none and whose boxes collide. The candidate rule (at least one solved body, overlapping bounding spheres, boxes not visibly apart, no joint or spring between the bodies) therefore lives only here. A manifold that already exists is collided every step whether the list has its pair or not, like the reference. Consequences:

- A pair with an existing manifold needs no special case in B: touching boxes overlap their bounding spheres, and a contact does not exclude, so the pair is in the list. `Solver_Broadphase_CoversEveryPairThatHasContacts` checks that every pair with contacts after a step was in that step's list.
- A pair that is excluded after it got a manifold (a joint added later) keeps the manifold. A pair is created only from the list.
- B must run before the narrowphase and after the constraint lists are rebuilt, and the solver does that: `AvbdSolver` creates the broadphase itself when the constraint buffers have contacts, with bins of 8 candidates per manifold slot and 1 static candidate per slot (plus 1024 each), and `DisableBroadphase` is then an error. `EnableBroadphase` replaces it with other capacities.

**Order.** The pairs inside a bin are in the order the threads got their atomic counter: unspecified, different between runs. The pair list is a set; tests compare sets. Stage S (sorting bodies by Morton key) does not exist yet, so the stage does not use it; when it does, the sorted order will make the lists of nearby bodies near each other in memory, and the order of the pairs will still be unspecified unless the narrowphase needs it.

**Host side.** `AvbdBroadphase` owns the grid, the static list, the pair buffer and the control words. `Run(bodies, constraints)` applies pending body edits and dispatches the six kernels; there is no readback. `ReadCounts`, `ReadPairs` and `ReadMaxRadius` read back for tests and tools. The solver runs it in `Step()` after the constraint rebuild and before predict, so it sees the poses of the end of the previous step, like the reference's broadphase, but only after `AvbdSolver.EnableBroadphase(dynamicPairCapacity, staticPairCapacity)`: until the narrowphase consumes the pairs, running it would cost time for nothing. The narrowphase reads the counts from the control buffer on the GPU and is dispatched indirectly (section 13).

**Limits.**
- No temporal coherence: the grid and the pairs are rebuilt each run. It is a few dispatches over the bodies, cheap next to the solver iterations.
- The static bodies are tested by brute force (above).
- A cell coordinate is clamped to +-1e9. Bodies farther from the origin than a billion cells all fall into the edge cells: the result stays correct (the exact sphere test still decides) but the search gets slow. With a cell of one unit that is far outside float precision anyway.
- A solved body of enormous radius next to many tiny ones makes the cell large and the cells crowded: the cell size follows the largest radius. Moving the few large bodies to the brute-force path is the fix, and is not done.
- A body with a non-finite size is not supported: it is left out of the cell size and its pairs are unspecified.
- The bin capacities are fixed when `AvbdBroadphase` is made. An overflow is reported, not repaired: the host decides whether to make a larger one.

**Check.** `CpuBroadphase` is the oracle: brute force over all pairs with the same rules (it is the narrowphase's candidate rule, which `CpuContacts` mirrors). `BroadphaseTests`: `BoxesSeparated` never separates a pair for which the reference narrow phase finds contacts (40,000 random pairs of turned boxes) and uses the pad; the oracle's pair set equals the reference solver's pair loop (minus static-static pairs) except for pairs of boxes that are apart, each of which must have no contact in the reference narrow phase, on the ground, stack, pyramid, rope and spring scenes and 10 dense random scenes (plus 10 with turned boxes); a scene of bars and slabs with known gaps (0.0005 kept, larger dropped); the GPU equals the oracle, on the state the GPU holds, on the same scenes and scenes far from the origin and with sizes from 0.05 to 8; pairs are well formed (body A above body B, kind matches the bodies, no duplicates, no static-static); kinematic, static and removed bodies; excluded constraints (a joint that broke after the lists were built still excludes until they are rebuilt, a contact does not); overflow counts pairs beyond the capacity and stores only real ones; repeated runs and moved bodies; empty and all-static scenes; the solver running the stage on the poses of the step start; and, with contacts, every pair the narrowphase holds contacts for being a pair of the list in 40 steps of a falling pyramid. `BodyLayoutTests` checks the constants and the record against the C# mirrors.

### Stage G: constraint lists on the GPU

`AvbdLists.compute` builds the constraint list of every solved body (CSR) from the joint and spring records and the manifold pool, replacing the CPU build of `ConstraintTopology` and the readbacks that fed it. `AvbdConstraintLists` (`Runtime/Constraints`) owns its two scratch buffers and dispatches it; `AvbdConstraintBuffers` calls it from `Rebuild()` and, with contacts, from `RefreshContactTopology()` after the narrow phase, followed by stage C.

**Output.** The `_ConstraintList` buffer (constraint ids grouped per solved body) and, per body, `Range = (offset, count, contacts, 0)` in the body Range array (owner: stage G). `contacts` is how many of the `count` entries are manifolds; they are the first ones in the list, so the joints and springs are the entries `[offset + contacts, offset + count)`. The broadphase uses it to skip the contacts when it asks whether a joint joins two bodies: that question is asked for every candidate pair, and a body at rest on a pile has dozens of contact entries and, usually, no joint at all. A body that is not solved gets count 0. Rules of section 12 are kept: only solved bodies get a list, a constraint to a body that is not solved (or to the world) is listed for its solved side only, and the list is sized `(capacity + manifold slots) * 2`, the most that can be needed (a joint, a spring or a manifold is listed for at most two bodies). A contact entry is a manifold slot with the top bit set (`AVBD_LIST_CONTACT_FLAG | slot`), one per manifold whatever its number of contact points; the other entries are constraint ids. The sort key of a contact is `(body A, body B)`.

**Fractured joints.** A record with the `Broken` flag (set by stage D) is not listed, and is counted. This replaces the per-step readback of the joint flags: the flag is read where it is, on the GPU. The timing is the one section 13 describes (the lists built in step k + 1, after the narrow phase, leave out a joint that broke in step k).

**Kernels**, in the order the host runs them:

| Kernel | Threads | Does |
| --- | --- | --- |
| `ListClear` | one per body | Zeroes the count and the contact count of the body (body 0 also zeroes the control words) |
| `ListCountRecords` | one per joint or spring | Adds 1 to the count of each solved body of the record; counts fractured records; flags records that name a body that is not in use |
| `ListCountContacts` | one per manifold slot below the pool's high-water mark (indirect, the narrow phase's retire arguments) | Adds the slot's contact count to the count and the contact count of each solved body of the pair (a free slot, an empty one or one with a bad body is skipped) |
| `ListScanBlock` | one group of 1024 per 1024 bodies | Exclusive scan of the counts inside the block; writes the offset inside the block (into the cursor word) and the block's sum |
| `ListScanTop` | one group of 1024 | Exclusive scan of the block sums; writes the total |
| `ListScanApply` | one per body | Offset = block offset + offset inside the block; writes Range `(offset, count, contacts, 0)` and the fill cursor |
| `ListFillRecords` | one per joint or spring | Appends the entry (id, other body, key) to the unsorted arrays of each solved body, with an atomic cursor |
| `ListFillContacts` | one per manifold slot below the high-water mark (indirect) | Appends one entry for the slot (`AVBD_LIST_CONTACT_FLAG | slot`) to the list of each solved body of its pair |
| `ListSort` | 8 per body (`AVBD_LIST_SORT_LANES`) | Rank sort of the body's list: lane `l` ranks the entries `l, l + 8, ...` and writes them, sorted, to `_ConstraintList` and `_ConstraintOther` |

**Side arrays.** The fill writes an unsorted copy of every entry into two arrays with one entry per list entry: `_ListKey` (uint2) is the sort key: a contact is `(A, B << 3 | k)`, a joint or spring `(0xFFFFFFFF, 0xFFFFFFFF - id)`; `_ListItem` (uint2) is the constraint id and the body on the other side (`AVBD_INVALID` for a world pin; both known where the entry is filled). The sort reads those two and writes the sorted `_ConstraintList` and `_ConstraintOther`, which stage C reads instead of dereferencing a constraint record per entry per round. The sort never reads a constraint record. `_ListWork` is `[counts | cursors | contacts per body | scan block sums]`, three body capacities plus one word per 1024 bodies.

**Order.** The pairs inside a bin are in the order the threads got their atomic counter: unspecified, different between runs. The pair list is a set; tests compare sets. Stage S (sorting bodies by Morton key) does not exist yet, so the stage does not use it; when it does, the sorted order will make the lists of nearby bodies near each other in memory, and the order of the pairs will still be unspecified unless the narrowphase needs it.

**Host side.** `AvbdBroadphase` owns the grid, the static list, the pair buffer and the control words. `Run(bodies, constraints)` applies pending body edits and dispatches the six kernels; there is no readback. `ReadCounts`, `ReadPairs` and `ReadMaxRadius` read back for tests and tools. The solver runs it in `Step()` after the constraint rebuild and before predict, so it sees the poses of the end of the previous step, like the reference's broadphase, but only after `AvbdSolver.EnableBroadphase(dynamicPairCapacity, staticPairCapacity)`: until the narrowphase consumes the pairs, running it would cost time for nothing. The narrowphase reads the counts from the control buffer on the GPU and is dispatched indirectly (section 13).

**Limits.**
- No temporal coherence: the grid and the pairs are rebuilt each run. It is a few dispatches over the bodies, cheap next to the solver iterations.
- The static bodies are tested by brute force (above).
- A cell coordinate is clamped to +-1e9. Bodies farther from the origin than a billion cells all fall into the edge cells: the result stays correct (the exact sphere test still decides) but the search gets slow. With a cell of one unit that is far outside float precision anyway.
- A solved body of enormous radius next to many tiny ones makes the cell large and the cells crowded: the cell size follows the largest radius. Moving the few large bodies to the brute-force path is the fix, and is not done.
- A body with a non-finite size is not supported: it is left out of the cell size and its pairs are unspecified.
- The bin capacities are fixed when `AvbdBroadphase` is made. An overflow is reported, not repaired: the host decides whether to make a larger one.

**Check.** `CpuBroadphase` is the oracle: brute force over all pairs with the same rules (it is the narrowphase's candidate rule, which `CpuContacts` mirrors). `BroadphaseTests`: `BoxesSeparated` never separates a pair for which the reference narrow phase finds contacts (40,000 random pairs of turned boxes) and uses the pad; the oracle's pair set equals the reference solver's pair loop (minus static-static pairs) except for pairs of boxes that are apart, each of which must have no contact in the reference narrow phase, on the ground, stack, pyramid, rope and spring scenes and 10 dense random scenes (plus 10 with turned boxes); a scene of bars and slabs with known gaps (0.0005 kept, larger dropped); the GPU equals the oracle, on the state the GPU holds, on the same scenes and scenes far from the origin and with sizes from 0.05 to 8; pairs are well formed (body A above body B, kind matches the bodies, no duplicates, no static-static); kinematic, static and removed bodies; excluded constraints (a joint that broke after the lists were built still excludes until they are rebuilt, a contact does not); overflow counts pairs beyond the capacity and stores only real ones; repeated runs and moved bodies; empty and all-static scenes; the solver running the stage on the poses of the step start; and, with contacts, every pair the narrowphase holds contacts for being a pair of the list in 40 steps of a falling pyramid. `BodyLayoutTests` checks the constants and the record against the C# mirrors.

### Stage G: constraint lists on the GPU

`AvbdLists.compute` builds the constraint list of every solved body (CSR) from the joint and spring records and the manifold pool, replacing the CPU build of `ConstraintTopology` and the readbacks that fed it. `AvbdConstraintLists` (`Runtime/Constraints`) owns its two scratch buffers and dispatches it; `AvbdConstraintBuffers` calls it from `Rebuild()` and, with contacts, from `RefreshContactTopology()` after the narrow phase, followed by stage C.

**Output.** The `_ConstraintList` buffer (constraint ids grouped per solved body) and, per body, `Range = (offset, count, contacts, 0)` in the body Range array (owner: stage G). `contacts` is how many of the `count` entries are manifolds; they are the first ones in the list, so the joints and springs are the entries `[offset + contacts, offset + count)`. The broadphase uses it to skip the contacts when it asks whether a joint joins two bodies: that question is asked for every candidate pair, and a body at rest on a pile has dozens of contact entries and, usually, no joint at all. A body that is not solved gets count 0. Rules of section 12 are kept: only solved bodies get a list, a constraint to a body that is not solved (or to the world) is listed for its solved side only, and the list is sized `(capacity + manifold slots) * 2`, the most that can be needed (a joint, a spring or a manifold is listed for at most two bodies). A contact entry is a manifold slot with the top bit set (`AVBD_LIST_CONTACT_FLAG | slot`), one per manifold whatever its number of contact points; the other entries are constraint ids. The sort key of a contact is `(body A, body B)`.

**Fractured joints.** A record with the `Broken` flag (set by stage D) is not listed, and is counted. This replaces the per-step readback of the joint flags: the flag is read where it is, on the GPU. The timing is the one section 13 describes (the lists built in step k + 1, after the narrow phase, leave out a joint that broke in step k).

**Kernels**, in the order the host runs them:

| Kernel | Threads | Does |
| --- | --- | --- |
| `ListClear` | one per body | Zeroes the count and the contact count of the body (body 0 also zeroes the control words) |
| `ListCountRecords` | one per joint or spring | Adds 1 to the count of each solved body of the record; counts fractured records; flags records that name a body that is not in use |
| `ListCountContacts` | one per manifold slot below the pool's high-water mark (indirect, the narrow phase's retire arguments) | Adds the slot's contact count to the count and the contact count of each solved body of the pair (a free slot, an empty one or one with a bad body is skipped) |
| `ListScanBlock` | one group of 1024 per 1024 bodies | Exclusive scan of the counts inside the block; writes the offset inside the block (into the cursor word) and the block's sum |
| `ListScanTop` | one group of 1024 | Exclusive scan of the block sums; writes the total |
| `ListScanApply` | one per body | Offset = block offset + offset inside the block; writes Range `(offset, count, contacts, 0)` and the fill cursor |
| `ListFillRecords` | one per joint or spring | Appends the entry (id, other body, key) to the unsorted arrays of each solved body, with an atomic cursor |
| `ListFillContacts` | one per manifold slot below the high-water mark (indirect) | Appends one entry for the slot (`AVBD_LIST_CONTACT_FLAG | slot`) to the list of each solved body of its pair |
| `ListSort` | 8 per body (`AVBD_LIST_SORT_LANES`) | Rank sort of the body's list: lane `l` ranks the entries `l, l + 8, ...` and writes them, sorted, to `_ConstraintList` and `_ConstraintOther` |

**Side arrays.** Next to the list the stage keeps two arrays with one entry per list entry. `_ListKey` (uint2) is the sort key, written when the entry is filled: a contact is `(A, B << 3 | k)`, a joint or spring `(0xFFFFFFFF, 0xFFFFFFFF - id)`. The sort compares these contiguous keys and never reads a constraint record (an earlier version read the 144-byte record of both entries at every comparison, which made the sort the main cost of the stage). `_ConstraintOther` is the body on the other side of the entry, written at the end of the sort; stage C reads it instead of dereferencing the record of every entry in every round, and `ConstraintTopology.ListOther` is its oracle.

**Order.** The fill order depends on thread timing and on the slots the pool handed out, so every list is sorted afterwards, with the key: contacts first, by (body A, body B, k) ascending, then joints and springs by falling id (newest first). That is the order of the CPU oracle, which is the reference's order of constraints in the first step, and it makes the floating-point sums reproducible from run to run. Insertion sort is fine because a list has a few dozen entries at most.

**Control buffer** (`AVBD_LIST_CONTROL_*`): the total number of entries, the number of fractured records left out (`AvbdConstraintBuffers.BrokenCount` reads it, once per build), and the number of records or manifolds that name a body slot that is not in use or report more than eight contacts (0 in a healthy scene; the CPU build threw an exception here).

**Cost.** No readback. What was slow at tens of thousands of bodies, and what replaced it: (1) one thread per body did an insertion sort in global memory (dependent loads and stores, a body in a pile has 20 to 30 entries), then read a 160 B record per entry for the other body; now 8 lanes per body rank the keys (independent, contiguous reads of keys that are cached) and write each entry once, out of place, so there are no hazards and no barriers, and a list of any length is correct (equal keys are ordered by their place in the unsorted list, so the ranks are always a permutation). The work per list is still quadratic in its length, divided over the lanes. (2) The prefix scan was one group of 1024 threads, each walking a run of `bodyCount / 1024` bodies with strided reads; it is now three kernels (block scan, scan of the block sums, apply) with coalesced reads, for up to 1024 x 1024 bodies (`AvbdConstraintLists` throws above that). (3) The contact kernels were dispatched over every manifold slot of the pool; they are now indirect over the slots below the high-water mark, with the narrow phase's retire arguments. (4) The number of contacts of a body (`Range.z`) is counted with the counts, not by a pass over the sorted list. (5) `ListBegin` is gone (body 0 of `ListClear` zeroes the control words).

**Check.** `ConstraintTopology` (`Build` and `BuildWithManifolds`) is the oracle. `ConstraintListTests`: the ranges and every list entry equal the oracle's for joints and springs with world pins, a body of mass 0 and a kinematic body as partners and two constraints between the same pair; for a fractured joint (left out, counted, `BrokenCount`); for a scene with contacts, after each of six steps; and for a spring to a massless body added after a first build. `BodyLayoutTests` compares the HLSL constants with the C# mirrors.

**Limits.** The coloring that follows is persistent with contacts and does not wait; a rebuild still waits (section 9, stage C). Constraints cannot be removed, and removing a body that has constraints is not supported.

### Ray cast

`AvbdRaycaster.Cast(origin, direction, maxDistance, callback)` (`AvbdRaycast.compute`) finds the nearest body a ray hits, without the CPU reading any pose. Two kernels: `Clear` sets a one-word result to the bits of `maxDistance`, and `Cast`, one thread per slot below the high-water mark, skips free slots, takes the ray into the box's frame (state record pose, size from the shape record) and runs the slab test. A ray that starts inside a box hits it at distance 0. A hit nearer than the result does `InterlockedMin` on the float's bits, which sort like the floats for values from 0 up, so no sort or float atomic is needed. The 4-byte result comes back with `AsyncGPUReadback`, a few frames later; the callback gets whether anything was hit and the distance. It is not part of `Step()`; the poses it tests are those the last step left (edits are applied first). The tests are in `RaycastTests`.

## 12. Constraints (step 4: joints and springs)

Contacts without a broadphase are section 13. Joints and springs share one record, a constraint, with the kind in the high 8 bits of the flags word (rule P7).

### Record

`ConstraintRecord` is 160 B, ten words (`Runtime/Layout/ConstraintRecords.cs`, `Shaders/Include/AvbdConstraintRecords.hlsl`). It holds joints and springs only; contacts have their own, smaller records (section 13, "Contact storage"). It is gathered by id, so it is a multiple of 32 B (rule P3). V and D read nearly every field of it, so there is no hot/cold split (rule P5).

| Word | Type | Content |
| --- | --- | --- |
| 0 | `uint4` | body A (`AVBD_INVALID` = the world), body B, flags and kind, plastic slot (only with the `Plastic` flag) |
| 1 | `float4` | anchor on A (a world point if A is the world); joint: linear stiffness, spring: stiffness |
| 2 | `float4` | anchor on B; joint: angular stiffness, spring: rest length |
| 3 | `float4` | linear constraint at the step start (C0); torque arm |
| 4 | `float4` | angular constraint at the step start; fracture threshold |
| 5, 6 | `float4` | penalty of the linear and the angular rows |
| 7, 8 | `float4` | multiplier of the linear and the angular rows |
| 9 | `float4` | x: linear fracture threshold (joints with `FracturableLinear`), else zero |

The spare lanes follow rule P4: stiffness shares a word with the anchor and torque arm with C0 (read by the same stages). A spring uses words 0 to 2 only, so a spring wastes 112 B; a separate smaller spring record is an option for later.

Flags: `HardLinear`, `HardAngular`, `Fracturable`, `Plastic`, `BendBreak`, `FracturableLinear`, `PlasticLinear` and `StretchBreak` (set when the constraint is created; `Fracturable` and `Plastic` are about the angular row, the other three about the linear row) and `Broken` (owner: stage D). **Infinite stiffness is a flag.** The reference solver uses `+infinity` for a hard constraint and tests it with `isinf`; GPU fast-math is not guaranteed to keep infinity, so the record holds a flag and stores 0 as the stiffness. The behaviors stay the same: a hard constraint has no stiffness cap in the warm start (its penalty is limited by the maximum penalty), a hard constraint uses stabilization and stores its multiplier, and the dual step caps the penalty at the maximum penalty. Fracture works the same way (a flag and a finite threshold, instead of an infinite one).

### Plastic joints

A joint can be **plastic**: when its angular multiplier is above a yield value, the angle lock gives way and the joint keeps the bend it ends up with, instead of springing back; a bend limit breaks a joint that has bent too far. This is three-avbd's joint plasticity, applied to the angular row of a joint with a hard angle lock (a ball joint has nothing to yield on its angular row). The linear row has its own yield, stretch limit and fracture value, described under *Linear row* below. It is **off unless the constraint buffers are made with a plastic capacity**, and then costs nothing for the joints that are not plastic.

*Rest rotation.* The angular row holds body B at `rotA * rest` instead of at `rotA`: its value is `AvbdQuatDifference(rotA * rest, rotB) * torqueArm`, and `rest` is the identity for a rigid joint (the product with the identity is exact, so a joint that never yields computes what it did before). `ConstraintInit`, `Dual` and the primal stage's `StampJoint` all take the angular value from one helper (`AvbdJointAngularError`, `AvbdPlasticRecords.hlsl`; `AvbdQuat.JointAngularError` in C#). The Jacobian does not change.

*Yield and bend break*, in the angular block of `Dual` (the reference solver's `AvbdJoint.UpdateDual`, `CpuDual` and the kernel agree):

1. After the multiplier update, if `|lambdaAng| > yield` and the joint is not above its fracture threshold (that one breaks instead), the multiplier is scaled back to `yield` and the joint has *yielded* in this iteration.
2. A yielding row does not ramp its penalty.
3. A joint that yielded sets `rest = normalize(conj(rotA) * rotB)`, so the current pose is the new rest pose, and zeroes `C0` of the angular row (the stabilization reference of this step would pull the bend back otherwise). The next step's `ConstraintInit` measures `C0` from the new `rest`.
4. With a bend limit, the joint then breaks (penalties and multipliers zeroed, `Broken` set, as for fracture) if the angle from `restStart` to `rest`, `2 atan2(|d.xyz|, |d.w|)` with `d = conj(restStart) * rest`, is above the limit.

A fracture value above the yield value only catches a load that jumps past it within one iteration. A load that arrives suddenly overshoots: the joint yields at about twice the static load it is meant to carry (three-avbd documents the same).

*Storage.* The state lives in a separate array, `PlasticRecord` (64 B, `Runtime/Layout/PlasticRecords.cs`, `Shaders/Include/AvbdPlasticRecords.hlsl`), not in the 160 B constraint record, so a joint that is not plastic keeps its memory traffic. The joint finds its record at `ident.w` (the word was unused for a joint) and the `Plastic` flag says it has one; `BendBreak` says the limit is set (rather than an infinite value, rule on infinity). Fields: `rest` and `restStart` (quaternions, identity at creation), `limits` = (yield, bend limit in radians, 0, 0), and a reserved word, which pads the record to a multiple of 32 B (rule P3). Owners (rule P7): creation writes the flags, `restStart` and `limits`; stage D writes `rest`; `ConstraintInit` and stage V only read it.

*Switch.* `new AvbdConstraintBuffers(bodies, capacity, shaderSet, contactCapacity, plasticCapacity)`: with `plasticCapacity` 0 (the default) no plastic array exists and `AvbdSolver` runs the joint kernels as they were; above 0 it allocates the array and runs the variants of `ConstraintInit`, `Dual` and `Primal` compiled with `AVBD_PLASTIC_JOINTS` (`multi_compile_local`; the solver sets the keyword on the shader before every dispatch). `AddJoint(..., fracture, yield, bendLimit)` makes a joint plastic with a finite `yield`; it needs a hard angle lock and a free plastic slot (`AddJoint` throws without a plastic capacity, and returns -1 and counts an overflow when the slots are used up). The switch is a constructor argument, like `contactCapacity`, because the array is allocated with the buffers; a setting that could change while the solver runs would still pay for the array and could not remove the variant's extra registers.

*Limits.* The rest rotation always starts as the identity (there is no authored rest rotation); a plastic joint is added through `AddJoint`, not edited later; `Broken` does not return. Nothing reads back how many joints have yielded.

*Linear row.* The linear row can fracture (`fractureLinear`, a threshold on `|lambdaLin|`), yield (`yieldLinear`) and break at a stretch limit (`stretchLimit`, in meters). The same rules as for the angular row apply (a multiplier above the fracture value breaks instead of yielding; a yielding row does not ramp its penalty). What is kept is the *anchor*, not a rest rotation: the row is `C = anchorA - anchorB`, so a yielding joint moves the anchor on body A (in A's body space; a world point for a joint to the world) to where the anchor on B is now. `C` becomes 0, the multiplier stays clamped at the yield value, and `C0` of the linear row is zeroed like the angular one. The stretch is `|anchorA_new - anchorA_start|`; above `stretchLimit` the joint breaks. The state is the `anchorA` field that the primal stage and `ConstraintInit` already read, so those stages needed no change and no variant. `PlasticRecord.limits` = (angular yield, bend limit, linear yield, stretch limit) and `reserved.xyz` is the starting anchor on A; the linear fracture value is in word 9 of the constraint record.

*Hard rows only.* Fracture and yield test the multiplier, and only a hard row has one: a soft (finite stiffness) row has `lambda` = 0 and a penalty capped at its stiffness, so a test on the multiplier could never fire. `AddJoint` therefore throws for a finite `fractureLinear` or `yieldLinear` on a joint with a finite linear stiffness, as the angular row already does. The alternatives would have been to test the spring force `k * |C|` in the soft row (a second code path in the dual kernel, extra branches for every joint and a new meaning of the threshold) or to give soft joints a multiplier (more memory traffic for all joints). Neither is free, and the hard-row version costs nothing for joints without the flags. A joint that needs both a spring-like response and a limit can use a rigid joint plus a spring constraint next to it.

### Lists and colors

`AvbdConstraintBuffers` (`Runtime/Constraints`) owns four buffers: the records, the per-body lists, the bodies bucketed by color, and the status words. Per-body ranges (offset, count) go into the body arrays' Range array, whose owner is stage G.

- The list is a scalar `uint` buffer (the P1 exception) of constraint ids, grouped by body. Bit 31 is reserved to tag contact entries later (rule P6).
- Only solved bodies get a list. A constraint between a solved body and a fixed one is listed for the solved body only.
- Entries are newest constraint first, so the floating-point sums run in the reference's order.
- Colors come from a greedy first-fit coloring: bodies from the highest slot down, lowest color not used by a colored neighbor. This is the reference's `GraphColoring`, and `ConstraintTopologyTests` checks the two against each other. Fixed bodies and world pins create no edge. `ConstraintTopology` still computes this coloring on the CPU, as the oracle of stage C; the solver uses the GPU coloring (stage C above), which gives the same colors in the default order.
- Bodies are bucketed by color in one list; the start and size of each color are on the GPU (stage C) and the host only knows the number of colors, so a color is one indirect dispatch.

**The lists are built on the GPU** (stage G, section 11), and so are the coloring and the buckets (stage C). `ConstraintTopology` is the CPU oracle for both. `Rebuild()` applies the pending body edits and reads the sizes and poses of the bodies back only when new constraints must be completed (the torque arm, the spring's rest length), so it is for when the constraint set changes, not for every frame. `AvbdSolver.Step()` calls it when constraints are pending or when the set of solved bodies changed (section 11, stage U). Known limits: constraints cannot be removed, removing a body that has constraints is not supported, and without contacts the lists are rebuilt only by `Rebuild()`, so a joint that fractures (stage D) stays in the lists and the coloring until then (with contacts the lists are rebuilt every step and drop it; section 13). The graph color is written to the Meta record by stage C (`Meta.state.x`).

### Tests

| Test | Needs a GPU | Checks |
| --- | --- | --- |
| `BodyLayoutTests` (constraint cases) | No | Record size and rule P3, flag bits, HLSL constants and struct members against the C# mirror |
| `ConstraintTopologyTests` | No | Lists and coloring against `ReferenceSolver` on the rope, the spring scene and 12 random scenes; fixed bodies; stiffness encoding |
| `PrimalTests` (oracle) | No | `CpuPrimal` against `ReferenceSolver` after one colored iteration, including C0 of every joint |
| `PrimalTests` (GPU) | Yes | The kernel against the oracle on the same scenes; `AddJoint` and `AddSpring` completing the records; a joint with a known closed-form result; static bodies never moving |
| `DualTests` (oracle) | No | `CpuConstraintInit` and `CpuDual` against `ReferenceSolver` after one step (1 and 3 iterations), a hand-computed dual update, fracture; plastic joints (yield, bend break, fracture first) against the reference solver and by hand, and a plastic joint that never yields against a rigid one; the same for the linear row (yield with anchor shift, stretch limit, linear fracture, an angular row that is untouched) |
| `DualTests` (GPU) | Yes | `Step()` against the oracles on the rope, the fracture scene and random lattices, including all constraint records; springs untouched by stage D; the plastic scene (records and `PlasticRecord`s), the plastic kernel variants on scenes without plastic joints, `AddJoint` validation and plastic slots, and a hanging box that bends and breaks |
| `GraphColoringTests` (oracle) | No | `CpuGraphColoring` in slot order against `ConstraintTopology`; the hashed order is a valid coloring and a strict order |
| `IndirectSweepTests` | Yes (all but the first) | The argument arithmetic covers every entry of a color once, including colors of more than 65,535 groups (no GPU); the GPU's start, size and arguments of every color against the oracle's sizes and the mirror; a color of five groups where every body is solved by a step |
| `GraphColoringTests` (GPU) | Yes | The kernels against the oracle in both orders (colors, color count, buckets); rounds on a long chain; recoloring after edits; no constraints; more than 64 colors; the persistent steps against `CpuGraphColoring.Persistent` (clashes, 1/2/tuned rounds, small cap, fresh interval, off) |
| `ColoringTunerTests` | No | The tuner: rounds up at once and down after a window, cap up at once, down only after a step from scratch, forced bodies |
| `ConstraintListTests` | Yes | Stage G: the GPU lists and ranges against `ConstraintTopology` entry by entry (pins, fixed and kinematic partners, a fractured joint, contacts over six steps, a massless partner added later) |
| `VelocityTests` | Both | Stage U and whole steps against the reference solver (section 11, stage U) |

None of the stage V, D, U, C or G code or tests has been compiled or run: there was no Unity editor, compiler or GPU in the environment where it was written.

## 13. Contacts (stage N)

Box-box contacts for the pairs the broadphase (stage B) finds. The cost is proportional to the number of candidate pairs and touching pairs, not to the square of the number of bodies. Manifolds live in a pool of slots (below); the number of slots is a constructor argument (`contactCapacity`) and the only limit on how many bodies can collide is the memory of the contact records: 1.3 kB per slot.

The first version of this stage tested every pair of bodies and stored each pair's manifold at a place fixed by its two slots. The CPU oracle (`CpuContacts`) still does that, and the GPU must produce the same manifolds: the oracle is the all-pairs semantics, and the pool is a storage change that must not change them.

### Reference semantics that are kept

- Body A of a manifold is the higher slot and B the lower. A manifold lives while the collision test finds contacts and is forgotten (history included) the step it finds none.
- Contacts are matched to the previous step's by feature key. A matched contact keeps its penalty, multiplier and stick state (penalty and multiplier decayed by `gamma`, `alpha * gamma`); if the old contact did not stick, the new `rA`, `rB` are taken. A new contact starts at the minimum penalty with a zero multiplier.
- `C0 = Basis * (xA - xB) + (margin, 0, 0)` with margin 0.01. The normal points from B to A. Friction is `sqrt(fA * fB)`.
- The reference's broadphase test (a sphere of radius `|size / 2|` per body) and `ConstrainedTo` (a joint or spring joins the pair) decide only whether a manifold is **created**. A pair that already has contacts is tested every step regardless. On the GPU stage B makes that decision (see the stage B section) and stage N tests every slot that is in use, listed by the broadphase or not.

### Records

A contact is a `ConstraintRecord` of kind `Contact` (2), so the primal kernel reads it like a joint. Mapping: A = higher slot, B = lower slot; anchor A = (rA, friction); anchor B = (rB, 0); `c0Lin` = C0; `c0Ang` = normal (B to A); `penaltyLin` and `lambdaLin` = the three rows (normal, tangent 1, tangent 2); `ident.w` = feature key; flag `Stick` (bit 4). The tangent basis is rebuilt from the normal in the kernels (`AvbdContactBasis`), as in the reference.

### Manifold pool

**Contact storage.** A manifold slot `s` has a 32 B pair record (`_ContactPairs[s]`, `ContactPairRecord`, `Shaders/Include/AvbdContactRecords.hlsl`) and room for eight 64 B points (`_ContactPoints[s * 8 + k]`, `ContactPointRecord`), so a slot takes 544 B (the generic 160 B record, eight times, took 1.3 kB). The pair record holds what the points of a pair share: `ident` = (body A, body B, both `AVBD_INVALID` for a free slot; number of points; stick bits, bit k for point k) and `normalFriction` = (normal pointing from B to A, friction). A point is 16 floats: `anchorA` = (rA, feature key as the bits of an int), `anchorB` = (rB, C0.x), `c0Penalty` = (C0.y, C0.z, penalty.x, penalty.y), `penaltyLambda` = (penalty.z, lambda.xyz); the tangent basis is derived from the normal. The stages that walk a pair load the two bodies and make the contact basis once, then loop over its points. The CPU oracle (`CpuContacts`) still stores one `ConstraintRecord` per contact, and `ContactCodec.ToRecord` / `AvbdConstraintBuffers.ReadContactRecords` convert a GPU point to that form for the tests. The stick bit moved from the record flags to the pair record so that the dual of a manifold writes it without atomics. A pair keeps its slot for as long as it has contacts, so the thread that owns a slot owns the old and the new contacts of its pair and needs no atomics for them. Free slots are on a stack (`_FreeList`, with the number of free slots in `_NpControl[AVBD_NP_FREE_TOP]`; the stack starts so that slot 0 is popped first).

Finding the manifold of a pair that the broadphase lists needs a map from the pair to its slot. It is a hash table rebuilt every step from the used slots (`_PairTable`, open addressing with linear probing, a power of two of at least twice the slots, entries are slot indices, key compared through `_SlotBodies`). It is only built and read inside one step, so it never needs deletions.

Other buffers: `_NpControl` (free count, overflow count and the high-water mark, below), `_NpArgs` (indirect dispatch arguments, below), `_DirtySlots` (a bit per body slot spawned or removed since the last narrow phase; a used slot of a dirty body is freed first, so a recycled body slot does not inherit the manifolds of the body that had it), and the broadphase's `_Pairs` and `_BpControl`.

**High-water mark.** `_NpControl[AVBD_NP_HIGH_WATER]` is the highest slot ever taken + 1; `NpAllocate` raises it with an atomic max. Every used slot is below it, so a kernel that visits slots needs no more threads. It never falls. Because a freed slot is the next one popped, a never-used slot is only taken when every slot below it is in use, so the mark is the peak number of manifolds that were ever in use at once, not the capacity (a pool of 500,000 slots for a scene that touches in 3,000 places is a mark near 3,000).

### Kernels

Run after stage B and stage P, in this order (`AvbdNarrowphase.compute`):

- `NpClearTable`: empties the hash table. The only kernel here whose size is a constant (the table).
- `NpPlan`, one thread: from the high-water mark and the broadphase's pair counters (each capped at its capacity), writes the indirect dispatch arguments of the kernels below and of the dual update of the contacts (`_NpArgs`, three words per entry, `AVBD_NP_ARGS_*`: retire, allocate over the dynamic bin, allocate over the static bin, narrowphase, dual; the dual's count is `8 * slots`). It runs twice, before `NpRetire` and again after `NpAllocate`, whose new slots can raise the mark.
- `NpRetire`, indirect, one thread per slot below the high-water mark: frees the slot of a pair with a dirty body, enters every other used slot in the table.
- `NpAllocate`, indirect, twice (`_Bin` = 0 and 1), one thread per pair the broadphase stored in that bin (a counter above the bin's capacity is capped): looks the pair up in the table; a pair without a slot whose boxes collide pops a free slot and stores its pair there with a zero count (so no history). The collision test is repeated in the next kernel for such a pair; the point of doing it first is that a candidate is a pair of overlapping *spheres*, and spending a slot on every candidate would make the pool several times larger than the number of manifolds and starve real pairs. The pairs of the broadphase are unique, so a new slot needs no table entry. With no free slot the pair is dropped, `_NpControl[AVBD_NP_OVERFLOW]` counts it, and the pair is a candidate again next step. Popping the stack is an atomic decrement; the decrement that finds it empty is undone by the thread that made it, which keeps the counter exact when many threads fail at once.
- `Narrowphase`, indirect, one thread per slot below the high-water mark: skips free slots; a slot of a free body or of two bodies that are not solved is freed; collides at the **start poses** (stage P has run, as in the reference where the broadphase sees the poses before the prediction) with the 15-axis SAT and the face clipping (`AvbdBoxCollision.hlsl`, one-to-one with the reference's `BoxCollision`); merges with the old contacts of the slot by key and writes the records and the count (`AvbdContactWarmStart.hlsl`). A slot whose boxes no longer collide is freed, which removes the manifold and its history, like the reference.
- `AvbdPrimal.compute`: the contact branch of the per-body loop stamps the contact's three rows into the 6x6 system (`AvbdContactSolve.hlsl`); the force is clamped to `F.x <= 0` and the friction cone.
- `AvbdDual.compute`, kernel `DualContact`, indirect with the arguments of the last `NpPlan`, one thread per slot below the high-water mark (a free slot returns at once), which walks the points of its pair: multiplier and penalty update, and the stick bits. Only the penalty and multiplier words of a point (32 of its 64 B) and the pair's `ident` are written back.

Slots are pushed only by `NpRetire` and `Narrowphase` and popped only by `NpAllocate`, and the kernels run one after the other, so a push never races a pop.

**Indirect dispatch.** The host used to size these dispatches from the capacities: every slot of the pool, every entry of both bins and, for the dual update, eight records per slot, which for a big pool was millions of threads that almost all returned at once, every iteration. Now no count is on the CPU at all: the kernels run over the slots that have been in use and the pairs that were found, through `DispatchIndirect` with arguments the GPU wrote. A count of more than 65,535 groups is laid out in rows of that many groups, like the coloring's (`AvbdIndirect.hlsl`, mirrored by `AvbdIndirect.GroupsFor`); a thread compares its item with the counts again because the last group overshoots. The arguments buffer is created as `IndirectArguments | Raw` and written through a `RWByteAddressBuffer`. The dispatches are still in the same order as before, so nothing about the results changes: a slot above the high-water mark has never been used and would have returned at once.

### Step order

`ApplyEdits`, `Rebuild` if needed, B, P, `PrepareContacts` (dirty bits), N (`NpClearTable`, `NpPlan`, `NpRetire`, `NpAllocate` twice, `NpPlan`, `Narrowphase`), `RefreshContactTopology`, `ConstraintInit` (joints and springs), then per iteration the primal sweep per color, `Dual`, `DualContact` (indirect); then U.

`RefreshContactTopology` builds the per-body lists on the GPU (stage G) from the slot table and the records, with no readback: contacts first in each list, **ordered by pair** (body A, then body B) whichever slots the pairs hold, then joints and springs newest first, which is the reference's order of constraints in the first step. It then runs the GPU coloring (stage C) with the contacts in the graph: persistent, so it keeps the colors of the step before, recolors only the bodies that clash and reads nothing back (stage C, persistent coloring); the sweep dispatches the cap. With contacts the coloring changes every step; `SlotOrder` (and `PersistentColoring = false`) color from scratch, wait for the GPU and give the exact coloring.

### Fractured joints

Stage D marks a joint that fractured with the `Broken` flag. Stage G reads the flag on the GPU and leaves the joint out of the lists it builds, and so out of the coloring. With contacts the lists are rebuilt right after the narrow phase of every step, so the joint that broke in step k is dropped from the lists of step k + 1. The broadphase reads those lists to exclude joined pairs (section 11, stage B), and it runs before that rebuild, so it still finds the pair joined in step k + 1 and a pair stops being excluded in step k + 2. That is the reference solver's timing: a joint that broke in step k is still attached in the broadphase of step k + 1, where its `initialize()` removes it, and the pair is a candidate from step k + 2. `AvbdConstraintBuffers.BrokenCount` is the number of broken joints and springs counted by the last list build (one readback of the control buffer when it is read after a build). Without contacts the lists are not rebuilt after a fracture (section 12).

### Limits

- The pool has a fixed number of slots (`contactCapacity`, at most `MaxContactManifolds` = 500,000, which is 272 MB of pair records and points). A new touching pair that finds no free slot has no contacts until one is free; `AvbdConstraintBuffers.ReadContacts()` reports it (`Overflow`). The broadphase's bins have the same kind of limit (`AvbdBroadphase.Overflowed`): the solver gives them 8 candidates per slot (+1024) and 1 static candidate per slot (+1024).
- The high-water mark never falls, so after a burst of contacts (a pile that later spreads out) the dispatches keep the burst's size. They are no bigger than the capacity, which is what they were before. Resetting the mark would need a pass that finds the highest slot in use.
- A manifold that exists is kept even if a joint or spring is added between its bodies later. A pair that the broadphase does not list is not created.
- No ignore-collision pairs yet (the reference's `IgnoreCollision`). Pairs joined by a joint or spring are excluded.
- Only box-box. Contacts of one pair are limited to eight (the reference's limit).
- The lists are rebuilt on the GPU every step and the coloring is persistent (kept colors, a fixed number of rounds, extra rounds and `ColorForce` for the rest, tuned from a readback that arrives late), so a step with contacts has no wait for the GPU in the solver; a `Rebuild()` still waits. The sweep is dispatched indirectly, `ColorCount` times (the cap).
- After the first step the summation order of a body's contacts may differ from the reference's (the reference appends new manifolds to the lists); the results differ by rounding only.

### How it is checked

- `Tools/HlslParity` compiles the contact HLSL as C++ and compares it with upstream's unmodified solver: `collide_parity` (narrow phase, about 800 000 cases: demo scenes, random pairs, ground, grids, crossing bars) and `contact_parity` (warm start, primal terms, dual). Both are bit-identical, apart from upstream's rounding order of one quaternion product (see its README). This checks the arithmetic, not the Unity glue.
- `ContactTests` (needs Unity): the topology with contacts; `CpuContacts` against the reference solver's manifolds at the first step (stack, and a scene of overlapping boxes) and over 90 (ground) and 60 (stack) steps; a box coming to rest; the GPU narrow phase against the oracle; the GPU step against the reference solver; joined pairs; recycled slots; the body-count cap.
- Tolerances for the multi-step comparisons come from upstream's own sensitivity: with every position nudged by 1e-6, the dropped box differs by 7e-4 (position) and 3e-4 (velocity) after 60 steps; the stack by up to 7e-3 and 0.1 (a boundary flip at step 1) and 8e-3 later. The pyramid is chaotic (1e-2 after 30 steps), so it is not compared.

None of the Unity code, shaders and tests of this stage has been compiled or run here (no editor, compiler or GPU in the environment).

The pool is checked by `ManifoldPoolTests`: a slot is given back when its pair separates; a full pool drops pairs, stores only touching ones and recovers; 700 bodies collide (the dense layout stopped at 512) with every face-neighbor pair present and nothing but neighbors; and over 40 steps of a falling pyramid every slot is used or free, no pair has two slots, no used slot is empty and body A is always the higher slot. The high-water mark is above every used slot and never falls, and for a pool of 20,000 slots under a small pyramid the planned arguments (retire, both allocate bins, narrowphase, dual) equal the sizes derived from the mark and from the broadphase's counters, and are a small fraction of what the capacity would need. `ContactTests` compares the manifolds of the GPU, wherever the pool put them, with the oracle's, which also runs the indirect kernels.
