import assert from 'node:assert/strict';
import { ARGS_WORDS, C_CLASHES, C_OVERFLOW, IA_COLOR, makeArgsWGSL } from '../src/avbd2d/gpu/layout.ts';
import { ARGS_ITEMS_3D, BODY_FLOATS, B_POS, C_CLASHING_COLORS, PRELUDE_3D } from '../src/avbd3d/gpu/layout.ts';
import { GpuSolver3D } from '../src/avbd3d/gpu/solver.ts';
import { solveWGSL } from '../src/avbd3d/gpu/wgsl-solve.ts';
import { Rigid } from '../src/avbd3d/ref/body.ts';
import { Joint } from '../src/avbd3d/ref/forces.ts';
import { Solver } from '../src/avbd3d/ref/solver.ts';
import { gpuTest } from './device.ts';

const PENDING = 256;
interface Buffers { colorBuffer: GPUBuffer; argsBuffer: GPUBuffer; adjBuffer: GPUBuffer; counterBuffer: GPUBuffer }
interface Fixture { count: number; rounds: number; cap: number; edges: [number, number][]; pending: number[] }
const bits = (a: Float32Array) => new Uint32Array(a.buffer, a.byteOffset, a.length);

function priority(i: number): number {
  let x = (i + 0x9e3779b9) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
}

// One local maximum settles per round along this monotone-priority path.
function chain(count: number, rounds: number, cap = 64): Fixture {
  const order = Array.from({ length: count }, (_, i) => i).sort((a, b) => priority(b) - priority(a) || b - a);
  return { count, rounds, cap, edges: order.slice(1).map((b, i) => [order[i], b]), pending: order.slice(rounds).sort((a, b) => a - b) };
}
function clique(): Fixture {
  const edges: [number, number][] = [];
  for (let a = 0; a < 65; a++) for (let b = a + 1; b < 65; b++) edges.push([a, b]);
  const last = Array.from({ length: 65 }, (_, i) => i).sort((a, b) => priority(a) - priority(b) || a - b)[0];
  return { count: 65, rounds: 128, cap: 64, edges, pending: [last] };
}
function independentChains(): Fixture {
  const edges: [number, number][] = [], pending: number[] = [];
  for (const offset of [0, 12]) {
    const order = Array.from({ length: 12 }, (_, i) => offset + i).sort((a, b) => priority(b) - priority(a) || b - a);
    for (let i = 1; i < order.length; i++) edges.push([order[i - 1], order[i]]);
    pending.push(...order.slice(4));
  }
  return { count: 24, rounds: 4, cap: 64, edges, pending: pending.sort((a, b) => a - b) };
}
async function words(device: GPUDevice, buffer: GPUBuffer): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    return new Uint32Array(staging.getMappedRange().slice(0));
  } finally { staging.destroy(); }
}
function make(device: GPUDevice, f: Fixture, options: { fuse?: boolean; solve?: string; oldSchedule?: boolean } = {}): GpuSolver3D {
  const ref = new Solver(); ref.gravity = 0; ref.dt = 1 / 120; ref.iterations = 7;
  const bodies = Array.from({ length: f.count }, (_, i) => new Rigid(ref, [0.5, 0.6, 0.7], 1, 0.4, [i * 2, (i % 3) / 32, (i % 5) / 64]));
  for (const [a, b] of f.edges) new Joint(ref, bodies[a], bodies[b], [0, 0, 0], [0.125, 0.0625, 0], 1000, 0);
  const gpu = new GpuSolver3D(device, ref, { bodyCapacity: 256, spatialSort: false, colorRounds: f.rounds,
    ...(options.solve === undefined ? {} : { shaders: { solve: options.solve } }) });
  assert.equal(gpu.fuseSparseColors, false, 'fusion remains opt in');
  gpu.colorCap = f.cap; gpu.fuseSparseColors = options.fuse ?? false;
  gpu.setFilters(Array.from({ length: f.count }, (_, i) => i), Array(f.count).fill(1), Array(f.count).fill(0));
  if (options.oldSchedule) {
    // The shared original argument writer dispatches every ordinary chunk concurrently.
    // Together with oldPrimal below this is the pre-fix physical schedule, not a new oracle.
    const internal = gpu as unknown as { layouts: { args: GPUBindGroupLayout }; pipes: { argsColors: GPUComputePipeline } };
    internal.pipes.argsColors = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [internal.layouts.args] }),
      compute: { module: device.createShaderModule({ code: makeArgsWGSL(PRELUDE_3D, ARGS_ITEMS_3D) }), entryPoint: 'argsColors' },
    });
  }
  return gpu;
}
async function snapshot(device: GPUDevice, gpu: GpuSolver3D) {
  const internal = gpu as unknown as Buffers;
  return { body: bits(await gpu.readBodies()), joint: bits(await gpu.readJoints()), color: await words(device, internal.colorBuffer),
    args: await words(device, internal.argsBuffer), adj: await words(device, internal.adjBuffer), counters: await words(device, internal.counterBuffer) };
}
function sameState(a: Awaited<ReturnType<typeof snapshot>>, b: Awaited<ReturnType<typeof snapshot>>) {
  for (const key of ['body', 'joint', 'color', 'adj', 'counters'] as const) assert.deepEqual(a[key], b[key], key);
  for (let c = 0; c < 64; c++) {
    if ((a.counters[C_CLASHING_COLORS + (c >>> 5)] & (1 << (c & 31))) === 0) {
      assert.deepEqual(a.args.slice(IA_COLOR + 3 * c, IA_COLOR + 3 * c + 3), b.args.slice(IA_COLOR + 3 * c, IA_COLOR + 3 * c + 3));
    }
  }
}
function schedule(gpu: GpuSolver3D, f: Fixture, a: Awaited<ReturnType<typeof snapshot>>) {
  const rank = new Map<number, number>(), starts = 2 * gpu.bodyCapacity, buckets = starts + 65;
  const pending = Array.from({ length: f.count }, (_, i) => i).filter(i => (a.color[i] & PENDING) !== 0);
  assert.deepEqual(pending, f.pending); assert.equal(a.counters[C_CLASHES], f.pending.length); assert.equal(a.counters[C_OVERFLOW], 0);
  assert.equal(a.counters[0], 0, 'joint-only fixture must not add contacts');
  const marked = (c: number) => (a.counters[C_CLASHING_COLORS + (c >>> 5)] & (1 << (c & 31))) !== 0;
  const pendingColors = new Set(pending.map(i => a.color[i] & 255));
  const lanes = (n: number) => n >= gpu.primalLanes[0] ? 1 : n >= gpu.primalLanes[1] ? 2 : n >= gpu.primalLanes[2] ? 4 : 8;
  for (let c = 0; c < gpu.colorCap; c++) {
    const lo = a.color[starts + c], count = a.color[starts + c + 1] - lo;
    assert.equal(marked(c), pendingColors.has(c), 'only pending colours use the serial mode');
    assert.equal(a.args[IA_COLOR + 3 * c], marked(c) ? 1 : Math.ceil(count * lanes(count) / 64));
    for (let k = 0; k < count; k++) { const i = a.color[buckets + lo + k]; assert.ok(!rank.has(i)); rank.set(i, k); }
  }
  assert.equal(rank.size, f.count, 'every body occurs once in its original bucket');
  for (const [x, y] of f.edges) {
    const c = a.color[x] & 255;
    if (c !== (a.color[y] & 255)) continue;
    const count = a.color[starts + c + 1] - a.color[starts + c], perChunk = 64 / lanes(count);
    if (Math.floor(rank.get(x)! / perChunk) !== Math.floor(rank.get(y)! / perChunk)) {
      assert.ok(marked(c), 'connected chunks must not run in concurrent workgroups');
    }
  }
  assert.ok(new Float32Array(a.body.buffer).every(Number.isFinite));
  assert.equal(a.args.length, ARGS_WORDS + 3, 'retain the existing sparse argument triple only');
}

for (const [name, f] of [['round budget', chain(96, 4)], ['64-color exhaustion', clique()]] as const) {
  gpuTest(`clashing colour chunks repeat exactly: ${name}`, async device => {
    const normal = make(device, f), repeat = make(device, f), fused = make(device, f, { fuse: true });
    try {
      for (const gpu of [normal, repeat, fused]) gpu.step();
      const a = await snapshot(device, normal), b = await snapshot(device, repeat), c = await snapshot(device, fused);
      schedule(normal, f, a); schedule(fused, f, c);
      assert.deepEqual(a, b);
      for (const key of ['body', 'joint', 'color', 'adj', 'counters'] as const) assert.deepEqual(a[key], c[key], key);
      assert.deepEqual([...c.args.slice(ARGS_WORDS)], [1, 1, 1], 'exercise sparse chunks too');
      const body = new Float32Array(a.body.buffer);
      for (const i of f.pending) assert.notEqual(body[i * BODY_FLOATS + B_POS], i * 2, 'pending body must advance');
    } finally { normal.destroy(); repeat.destroy(); fused.destroy(); }
  });
}

gpuTest('clashing colour masks clear on the next quiet step and restore ordinary dispatches', async device => {
  // Exercise each mask word without changing production coloring parameters.
  for (const cap of [8, 64]) {
    const f = chain(96, 4, cap), gpu = make(device, f);
    try {
      gpu.step(); const clashing = await snapshot(device, gpu); schedule(gpu, f, clashing);
      const word = (cap - 1) >>> 5, bit = (1 << ((cap - 1) & 31)) >>> 0;
      assert.equal(clashing.counters[C_CLASHING_COLORS + word], bit);
      assert.equal(clashing.counters[C_CLASHING_COLORS + (1 - word)], 0);
      assert.equal(clashing.args[IA_COLOR + 3 * (cap - 1)], 1);

      // This fixture's priority path now has enough rounds to finish. Keep its bodies,
      // joints and cap in place, so the previous step's mask must actually be cleared.
      gpu.colorRounds = 128; gpu.step(); const quiet = await snapshot(device, gpu);
      schedule(gpu, { ...f, rounds: 128, pending: [] }, quiet);
      assert.deepEqual([...quiet.counters.slice(C_CLASHING_COLORS, C_CLASHING_COLORS + 2)], [0, 0]);
      assert.equal(quiet.counters[C_CLASHES], 0);
      assert.ok(Array.from({ length: cap }, (_, c) => quiet.args[IA_COLOR + 3 * c]).some(n => n > 1),
        'a quiet bucket must regain its ordinary multi-workgroup dispatch');
    } finally { gpu.destroy(); }
  }
});

// Restore only the old ordinary primal call, retaining the migration helper. The math
// remains the original shader math; this is not an independent implementation of it.
const oldPrimal = solveWGSL.replace('  if (serial != 0u) { primalChunks(pc.color, lid.x); }\n  else { primalColor(pc.color, lid.x, wid.x, 0u); }',
  '  primalColor(pc.color, lid.x, wid.x, 0u);');
assert.notEqual(oldPrimal, solveWGSL);

for (const [name, f] of [['one workgroup', chain(10, 4, 8)], ['two independent workgroups', independentChains()]] as const) {
  gpuTest(`pending colours preserve the old within-chunk Jacobi: ${name}`, async device => {
    const normal = make(device, f), old = make(device, f, { solve: oldPrimal, oldSchedule: true });
    try {
      normal.step(); old.step(); const a = await snapshot(device, normal);
      schedule(normal, f, a); sameState(a, await snapshot(device, old));
    } finally { normal.destroy(); old.destroy(); }
  });
}

gpuTest('proper colouring keeps old math and current custom modules bit-exact', async device => {
  for (const f of [chain(12, 16), { count: 24, rounds: 4, cap: 64, edges: [] as [number, number][], pending: [] }]) {
    const normal = make(device, f), old = make(device, f, { solve: oldPrimal, oldSchedule: true }), custom = make(device, f, { solve: solveWGSL });
    try {
      for (const gpu of [normal, old, custom]) gpu.step(); const a = await snapshot(device, normal); schedule(normal, f, a);
      sameState(a, await snapshot(device, old)); assert.deepEqual(a, await snapshot(device, custom));
    } finally { normal.destroy(); old.destroy(); custom.destroy(); }
  }
});

gpuTest('custom chunk solves retain their own body math', async device => {
  const marked = solveWGSL.replace('bodies[i].pos = vec4f(pos.xyz - vec3f(v1, v2, v3), pos.w);',
    'bodies[i].pos = vec4f(123.0 + f32(i), pos.yzw);');
  assert.notEqual(marked, solveWGSL);
  const f = chain(10, 4, 8), gpu = make(device, f, { solve: marked });
  try { gpu.step(); const body = await gpu.readBodies(); for (const i of f.pending) assert.equal(body[i * BODY_FLOATS + B_POS], 123 + i); }
  finally { gpu.destroy(); }
});
