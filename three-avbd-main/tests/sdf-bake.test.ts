import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { BoxGeometry, TorusGeometry } from 'three';
import { bakeVoxelSdf, bakeVoxelSdfAsync, finalizeSdf, AnalyticField } from '../src/lib/sdf.ts';
const near = (a: number, b: number, epsilon = 1e-5) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);

test('CPU bake preserves units, welds geometry seams and matches box distances', () => {
  const geometry = new BoxGeometry(2, 2, 2).translate(3, 0, 0);
  const before = geometry.getAttribute('position').array.slice();
  const baked = bakeVoxelSdf(geometry, { resolution: 16 });
  assert.equal(baked.watertight, true); assert.deepEqual(baked.warnings, []);
  assert.deepEqual(geometry.getAttribute('position').array, before);
  const oracle = new AnalyticField({ kind: 'box', halfExtents: [1, 1, 1] });
  const { dimensions: dims, origin, cell, grid } = baked.field;
  for (let z = 0; z < dims[2]; z++) for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
    const expected = oracle.sample([origin[0] + x * cell - 3, origin[1] + y * cell, origin[2] + z * cell]);
    near(grid[4 * (x + dims[0] * (y + dims[1] * z))], expected.phi);
  }
  near(baked.field.sample([3, 0, 0]).phi, -1);
  near(baked.field.sample([4.25, 0, 0]).phi, .25);
  near(baked.field.sample([4.25, 0, 0]).gradient[0], 1);
  const normalized = bakeVoxelSdf(geometry, { resolution: 16, normalize: true });
  assert.deepEqual(normalized.transform, { center: [3, 0, 0], scale: .5 });
  near(normalized.field.sample([0, 0, 0]).phi, -.5);
  const props = finalizeSdf(baked.field, { resolution: 16, candidateCount: 200 });
  near(props.volume, 8); near(props.centerOfMass[0], 3); near(props.inertia[0], 16 / 3, 2e-5);
});

test('CPU bake retains a nonconvex torus hole and handles reversed winding', () => {
  const g = new TorusGeometry(1, .3, 8, 20);
  const baked = bakeVoxelSdf(g, { resolution: 16 });
  assert.ok(baked.field.sample([0, 0, 0]).phi > .6);
  assert.ok(baked.field.sample([1, 0, 0]).phi < -.15);
  const index = g.getIndex()!;
  for (let i = 0; i < index.count; i += 3) { const a = index.getX(i); index.setX(i, index.getX(i + 1)); index.setX(i + 1, a); }
  const flipped = bakeVoxelSdf(g, { resolution: 16 });
  assert.deepEqual(flipped.field.grid, baked.field.grid);
});

test('CPU bake validates topology, indices, allocation budget and aborts', async () => {
  const g = new BoxGeometry();
  assert.throws(() => bakeVoxelSdf(g, { resolution: 256, maxGridBytes: 1024 }), /maxGridBytes/);
  assert.throws(() => bakeVoxelSdf(g, { resolution: 1 }), /resolution/);
  assert.throws(() => bakeVoxelSdf({ positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], indices: [0, 1, 3] }), /index/);
  const open = bakeVoxelSdf({ positions: [0, 0, 0, 1, 0, 0, 0, 1, 0] }, { resolution: 4 });
  assert.equal(open.watertight, false); assert.equal(open.warnings.length, 1);
  const controller = new AbortController(); let progressed = false;
  await assert.rejects(bakeVoxelSdfAsync(g, { signal: controller.signal, onProgress: () => { progressed = true; controller.abort(); } }), { name: 'AbortError' });
  assert.equal(progressed, true);
});

test('async CPU bake agrees with sync and yields monotonic progress', async () => {
  const g = new BoxGeometry(), progress: number[] = [];
  const asyncResult = await bakeVoxelSdfAsync(g, { resolution: 4, onProgress: p => progress.push(p) });
  assert.deepEqual(asyncResult.field.grid, bakeVoxelSdf(g, { resolution: 4 }).field.grid);
  assert.equal(progress.at(-1), 1); assert.ok(progress.every((v, i) => i === 0 || v >= progress[i - 1]));
});

test('analytic finalization matches pinned upstream box and torus mass and ordered seeds', () => {
  for (const name of ['box', 'torus'] as const) {
    const field = new AnalyticField(name === 'box' ? { kind: 'box', halfExtents: [.5, .5, .5] } : { kind: 'torus', majorRadius: .5, minorRadius: .2 });
    const expected = JSON.parse(readFileSync(new URL(`./fixtures/oracle-sdf/seeds-${name}.json`, import.meta.url), 'utf8'));
    const props = finalizeSdf(field);
    near(props.volume, expected.volume); props.centerOfMass.forEach((v, i) => near(v, expected.centerOfMass[i]));
    props.inertia.forEach((v, i) => near(v, expected.inertia[Math.floor(i / 3)][i % 3]));
    assert.equal(field.seeds.length, expected.seeds.length);
    field.seeds.forEach((p, i) => p.forEach((v, j) => near(v, expected.seeds[i].p[j])));
  }
});

test('real CPU worker transfers results, preserves input and supports cancellation/reuse', async () => {
  const { Worker } = await import('node:worker_threads');
  const { bakeVoxelSdfInWorker } = await import('../src/lib/sdf.ts');
  const thread = new Worker(new URL('./helpers/sdf-worker.ts', import.meta.url));
  const listeners = new Map<unknown, (...args: unknown[]) => void>();
  const worker = {
    postMessage: (data: unknown, transfers?: ArrayBuffer[]) => thread.postMessage(data, transfers),
    addEventListener(type: string, fn: (event: unknown) => void) {
      const wrapper = (data: unknown) => fn(type === 'message' ? { data } : data); listeners.set(fn, wrapper); thread.on(type, wrapper);
    },
    removeEventListener(type: string, fn: (event: unknown) => void) { const wrapper = listeners.get(fn); if (wrapper) thread.off(type, wrapper); },
  } as unknown as globalThis.Worker;
  try {
    const geometry = new BoxGeometry(), positions = geometry.getAttribute('position').array, indices = geometry.getIndex()!.array;
    const byteLength = positions.byteLength;
    const result = await bakeVoxelSdfInWorker(worker, { positions, indices }, { resolution: 4 });
    near(result.field.sample([0, 0, 0]).phi, -.5); assert.equal(positions.byteLength, byteLength);
    const cancel = new AbortController();
    await assert.rejects(bakeVoxelSdfInWorker(worker, { positions, indices }, { resolution: 16, signal: cancel.signal, onProgress: () => cancel.abort() }), { name: 'AbortError' });
    await bakeVoxelSdfInWorker(worker, { positions, indices }, { resolution: 4, onProgress: () => { throw undefined; } }).then(
      () => assert.fail('Thrown undefined must reject the worker promise'), error => assert.equal(error, undefined));
    const again = await bakeVoxelSdfInWorker(worker, { positions, indices }, { resolution: 4 });
    assert.deepEqual(again.field.grid, result.field.grid);
    const direct = bakeVoxelSdf(geometry, { resolution: 4 });
    assert.deepEqual(result.field.surfaceTriangles, direct.field.surfaceTriangles);
    finalizeSdf(result.field, { resolution: 8, candidateCount: 300 });
    finalizeSdf(direct.field, { resolution: 8, candidateCount: 300 });
    assert.deepEqual(result.field.seeds, direct.field.seeds);
    await assert.rejects(bakeVoxelSdfInWorker(worker, { positions, indices: [0, 1, .5] }, { resolution: 4 }), /index/);
  } finally { await thread.terminate(); }
});

test('public fields reject invalid dimensions and nonfinite stored values', async () => {
  const { VoxelField } = await import('../src/lib/sdf.ts');
  for (const radius of [-1, 0, NaN, Infinity, 1e-60, 1e40]) assert.throws(() => new AnalyticField({ kind: 'sphere', radius }));
  assert.throws(() => new AnalyticField({ kind: 'box', halfExtents: [1, -1, 1] }));
  assert.throws(() => new AnalyticField({ kind: 'spiky-wheel', hubRadius: 1, hubHalfWidth: 1, spikeRadius: .1, spikeHalfHeight: .3, spikes: 1.5 }));
  const bounds = { min: [0,0,0] as [number,number,number], max: [1,1,1] as [number,number,number] };
  const grid = new Float32Array(32);
  for (const cell of [1e-60, 1e40]) assert.throws(() => new VoxelField([2,2,2], [0,0,0], cell, grid, bounds));
  grid[7] = NaN; assert.throws(() => new VoxelField([2,2,2], [0,0,0], 1, grid, bounds), /finite/);
  grid[7] = 0;
  assert.throws(() => new VoxelField([2,2,2], [0,0,0], 1, grid, { min: [2,0,0], max: [1,1,1] }), /bounds/);
});

test('mesh bake preserves nested cavity and disconnected solid signs; diagnoses nonmanifold edges', () => {
  const outer = new BoxGeometry(2,2,2).toNonIndexed(), inner = new BoxGeometry(1,1,1).toNonIndexed();
  const a = Array.from(outer.getAttribute('position').array), b = Array.from(inner.getAttribute('position').array);
  for (let i = 0; i < b.length; i += 9) for (let j = 0; j < 3; j++) [b[i+3+j], b[i+6+j]] = [b[i+6+j], b[i+3+j]];
  const cavity = bakeVoxelSdf({ positions: [...a, ...b] }, { resolution: 16 });
  assert.equal(cavity.watertight, true);
  assert.ok(cavity.field.sample([0,0,0]).phi > .45);
  assert.ok(cavity.field.sample([.75,0,0]).phi < -.2);
  const other = a.map((v, i) => i % 3 === 0 ? v + 4 : v);
  const disjoint = bakeVoxelSdf({ positions: [...a, ...other] }, { resolution: 24 });
  assert.ok(disjoint.field.sample([0,0,0]).phi < -.9);
  assert.ok(disjoint.field.sample([4,0,0]).phi < -.9);
  assert.ok(disjoint.field.sample([2,0,0]).phi > .9);
  const duplicate = bakeVoxelSdf({ positions: [...a, ...a.slice(0,9)] }, { resolution: 4 });
  assert.equal(duplicate.watertight, false);
});
