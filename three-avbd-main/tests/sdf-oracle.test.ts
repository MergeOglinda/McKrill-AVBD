import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';

const directory = new URL('./fixtures/oracle-sdf/', import.meta.url);
const load = (name: string) => JSON.parse(readFileSync(new URL(name, directory), 'utf8'));
interface Sample { p: number[]; phi: number; gradient: number[] }
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 2e-6, `${a} != ${b}`);

test('SDF oracle provenance stays pinned to the upstream driver and fixture bytes', () => {
  const manifest = load('manifest.json');
  assert.equal(manifest.revision, '1dcdf1e7fa07d16fda15826ad8c2e6697075a76e');
  const driver = readFileSync(new URL('../tools/cpp-oracle-sdf/oracle.cpp', import.meta.url));
  assert.equal(createHash('sha256').update(driver).digest('hex'), manifest.driverSha256);
  for (const [name, hash] of Object.entries(manifest.adapters)) {
    assert.equal(createHash('sha256').update(readFileSync(new URL(`../tools/cpp-oracle-sdf/${name}`, import.meta.url))).digest('hex'), hash, name);
  }
  for (const name of ['solver.cpp', 'scenes.cpp']) assert.match(manifest.inputs[`source/${name}`], /^[a-f0-9]{64}$/);
  assert.deepEqual(readdirSync(directory).filter(n => n.endsWith('.json') && n !== 'manifest.json').sort(), Object.keys(manifest.fixtures).sort());
  for (const [name, hash] of Object.entries(manifest.fixtures)) {
    assert.equal(createHash('sha256').update(readFileSync(new URL(name, directory))).digest('hex'), hash, name);
  }
  for (const name of ['bunny', 'gear', 'cow', 'chair']) assert.match(manifest.inputs[`assets/${name}.obj`], /^[a-f0-9]{64}$/);
});

test('SDF level 1 covers primitives, all Table 5 warps and four finite voxel fields', () => {
  const names = readdirSync(directory).filter(n => n.endsWith('.json') && !['manifest.json', 'optimizer.json', 'optimizers.json', 'manifolds.json', 'manifolds-optimizers.json'].includes(n) && !n.startsWith('seeds-') && !n.startsWith('scene-'));
  assert.equal(names.length, 24);
  for (const name of names) {
    const field = load(name);
    assert.equal(field.samples.length, 175);
    for (const s of field.samples as Sample[]) {
      assert.equal(s.p.length, 3); assert.equal(s.gradient.length, 3);
      assert.ok([s.phi, ...s.p, ...s.gradient].every(Number.isFinite), name);
    }
    if (field.grid) {
      close(field.grid.cell, 0.01);
      assert.equal(field.grid.bytes, field.grid.dimensions.reduce((a: number, b: number) => a * b, 16));
    }
  }
  for (const [type, values] of [['bend', [0.15, 0.55, 0.95]], ['displace', [0.05, 0.10, 0.20]], ['twist', [0.50, 2.50, 5.00]]] as const) {
    for (const k of values) close(load(`${type}-${k.toFixed(2)}.json`).warp.k, k);
  }
});

test('upstream box and sphere samples agree with independent analytic distances', () => {
  for (const s of load('sphere.json').samples as Sample[]) {
    const r = Math.hypot(...s.p);
    close(s.phi, r - 0.5);
    if (r > 0) s.gradient.forEach((g, i) => close(g, s.p[i] / r));
  }
  for (const s of load('box.json').samples as Sample[]) {
    const q = s.p.map(x => Math.abs(x) - 0.5);
    close(s.phi, Math.hypot(...q.map(x => Math.max(x, 0))) + Math.min(Math.max(...q), 0));
  }
});

test('optimizer fixtures retain domains, clamped seeds, repulsion and iteration-limit cases', () => {
  const cases = load('optimizer.json').cases;
  assert.equal(cases.length, 16);
  assert.ok(cases.some((c: { repulsionPoints: number[][] }) => c.repulsionPoints.length > 0));
  for (const c of cases) {
    assert.ok(Number.isFinite(c.objective));
    assert.ok(c.iterations >= 0 && c.iterations <= c.maxIterations);
    assert.ok(c.queries >= 4 && Number.isInteger(c.queries));
    c.x.forEach((x: number, i: number) => assert.ok(Number.isFinite(x) && x >= c.domain.min[i] - 1e-7 && x <= c.domain.max[i] + 1e-7));
    if (c.maxIterations === 0) { assert.equal(c.iterations, 0); assert.equal(c.queries, 4); }
  }
});

interface Contact {
  x: number[]; normal: number[]; depth: number; separation: number; softmax: number; rank: number;
  active: boolean; cached: boolean; normalImpulse: number; frictionImpulse: number[];
}
const distance = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]));

test('SDF curvature seeds retain upstream order, scaled thresholds and Poisson spacing', () => {
  const names = readdirSync(directory).filter(n => n.startsWith('seeds-'));
  assert.equal(names.length, 24);
  for (const name of names) {
    const f = load(name), seeds = f.seeds as { p: number[]; phi: number; laplacian: number }[];
    const extent = Math.max(...f.bounds.max.map((x: number, i: number) => x - f.bounds.min[i]));
    close(f.kappa, 4 / extent); close(f.delta, 0.001 * extent); close(f.spacing, 0.08 * extent);
    assert.ok(f.volume > 0 && Number.isFinite(f.volume));
    assert.ok(seeds.length > 0, name);
    for (let i = 0; i < seeds.length; i++) {
      const s = seeds[i];
      assert.ok([s.phi, s.laplacian, ...s.p].every(Number.isFinite), name);
      assert.ok(Math.abs(s.laplacian) > f.kappa - 1e-5, name);
      for (let j = 0; j < i; j++) assert.ok(distance(s.p, seeds[j].p) >= f.spacing - 1e-6, name);
    }
  }
});

test('SDF manifold random draws replay the exact PCG state with no patched RNG', () => {
  const mask64 = (1n << 64n) - 1n, mask32 = (1n << 32n) - 1n;
  for (const c of load('manifolds.json').cases) {
    let state = BigInt(c.rng.before);
    for (const draw of c.rng.draws as number[]) {
      const old = state;
      state = (old * 6364136223846793005n + 1442695040888963407n) & mask64;
      const shifted = (((old >> 18n) ^ old) >> 27n) & mask32, rot = old >> 59n;
      assert.equal(Number(((shifted >> rot) | (shifted << ((-rot) & 31n))) & mask32), draw);
    }
    assert.equal(state.toString(), c.rng.after, c.name);
  }
});

test('SDF manifolds preserve cache semantics, classification, separation, dedup and final rank', () => {
  const cases = load('manifolds.json').cases;
  assert.equal(cases.length, 21);
  const find = (name: string) => cases.find((c: { name: string }) => c.name === name);
  for (const c of cases) {
    const ps = c.points as Contact[];
    assert.equal(c.stats.pairs, 1);
    assert.equal(c.stats.contacts, ps.filter(p => p.active).length);
    assert.ok(c.stats.queries >= ps.length);
    assert.ok(ps.length <= c.params.cacheSize + c.params.randomSamples);
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      assert.ok([...p.x, ...p.normal, p.depth, p.separation, p.rank, p.softmax].every(Number.isFinite));
      close(Math.hypot(...p.normal), 1);
      assert.equal(p.active, p.separation < c.params.margin);
      p.x.forEach((x, j) => assert.ok(x >= c.domain.min[j] - 1e-6 && x <= c.domain.max[j] + 1e-6));
      const rank = p.softmax + (p.cached ? 0 : c.params.replacementThreshold) - c.params.alpha * ps.reduce((sum, q) => sum + distance(p.x, q.x), 0);
      close(rank, p.rank);
      if (i) assert.ok(ps[i - 1].rank <= p.rank);
      for (let j = 0; j < i; j++) assert.ok(distance(p.x, ps[j].x) >= c.params.tolerance - 1e-6);
      if (p.cached) assert.ok(c.initialCache.some((q: Contact) => q.normalImpulse === p.normalImpulse && distance(q.frictionImpulse, p.frictionImpulse) === 0));
    }
  }
  assert.equal(find('convex-skip').rng.draws.length, 0);
  assert.ok(find('convex-no-skip').rng.draws.length > 0);
  assert.equal(find('duplicate-cache').points.length, 1);
  assert.ok(find('warm-disabled').points.every((p: Contact) => !p.cached && p.normalImpulse === 0));
  assert.equal(find('no-cache').points.length, 1);
  assert.ok(find('margin-gap').points.every((p: Contact) => p.active && p.separation > 0));
  assert.ok(find('separated-spheres').points.every((p: Contact) => !p.active && p.separation > 0));
  for (const name of ['replacement-zero', 'replacement-high', 'torus-seeds', 'bend-seeds', 'voxel-seeds']) assert.ok(find(name).rng.draws.length > 0, name);
  const warm = cases.filter((c: { name: string }) => c.name === 'warm-box');
  for (let i = 1; i < warm.length; i++) {
    assert.deepEqual(warm[i].initialCache, warm[i - 1].points);
    assert.equal(warm[i].rng.before, warm[i - 1].rng.after);
  }
  const a = find('static-a-advection'), b = find('static-b-advection'), d = find('dynamic-advection');
  assert.notDeepEqual(a.points, b.points); assert.notDeepEqual(a.points, d.points);
});

test('SDF released scenes capture six seconds of finite CPU behavior, including all Table 5 strengths', () => {
  const names = readdirSync(directory).filter(n => n.startsWith('scene-'));
  assert.equal(names.length, 20);
  for (const name of names) {
    const f = load(name);
    assert.equal(f.solver.type, 'upstream sequential impulses');
    assert.equal(f.solver.substeps, 4); close(f.solver.dt, 1 / 60);
    assert.equal(f.frames, 360); assert.equal(f.samples.length, f.frames);
    assert.ok(f.initialBodies.some((b: { static: boolean }) => b.static));
    assert.ok(f.initialBodies.some((b: { static: boolean }) => !b.static));
    for (const [i, s] of f.samples.entries()) {
      assert.equal(s.frame, i + 1);
      assert.ok([s.kineticEnergy, s.maxContactPenetration, s.minCenterY].every(Number.isFinite));
      assert.ok(s.kineticEnergy >= 0 && s.maxContactPenetration >= 0);
      for (const v of Object.values(s.stats)) assert.ok(Number.isInteger(v) && (v as number) >= 0);
      assert.ok(s.stats.contacts <= s.stats.pairs * (f.params.cacheSize + f.params.randomSamples));
      if (s.frame % 30 === 0) {
        assert.ok(Number.isFinite(s.maxVertexPenetration) && s.maxVertexPenetration >= 0);
        assert.ok(Number.isInteger(s.vertexQueries) && s.vertexQueries >= 0);
      }
    }
  }
  // Qualitative checks, not fitted paper performance numbers or trajectory parity.
  for (const name of ['scene-0.json', 'scene-1.json', 'scene-6.json']) {
    const f = load(name), last = f.samples.at(-1);
    assert.ok(last.kineticEnergy < 0.01, name);
    assert.ok(last.minCenterY > 0, name);
    assert.ok(last.stats.contacts > 0, name);
  }
  for (const [index, ks] of [[3, ['0.15', '0.55', '0.95']], [4, ['0.50', '2.50', '5.00']], [5, ['0.05', '0.10', '0.20']]] as const) {
    for (const k of ks) {
      const f = load(`scene-${index}-k${k}.json`);
      assert.equal(f.warpOverride, k);
      for (const b of f.initialBodies) if (!b.static) close(b.shape.k, Number(k));
    }
  }
});
