# GPU data layout and buffer lifecycle (task 3)

Design spec for the package's GPU buffers. It is written step by step; each section is added when its step is done.

| Step | Section | Status |
| --- | --- | --- |
| 1 | Read/write matrix | Done |
| 2 | Packing rules | Done (this version) |
| 3 | Body arrays | Not started |
| 4 | Other buffers | Not started |
| 5 | Capacity calculator | Not started |
| 6 | Slot allocation and edits | Not started |
| 7 | Growth and overflow policy | Not started |
| 8 | Buffer owner | Not started |
| 9 | HLSL/C# layout contract | Not started |
| 10 | Tests and review | Not started |

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
| Step-start pose | | | | | | | W | R | R | R | | R | Hot |
| Linear and angular velocity | W | | R | | | | R | | | W | R | | Cold |
| Previous linear velocity (adaptive warm start) | | | | | | | R | | | W | | | Cold |
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
