import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Solver } from '../src/avbd3d/ref/solver.ts';
import { Rigid } from '../src/avbd3d/ref/body.ts';
import { AnalyticField } from '../src/avbd3d/ref/sdf/shapes.ts';
import { SdfCollisionAdapter } from '../src/avbd3d/ref/sdf/adapter.ts';
import { SdfBody } from '../src/avbd3d/ref/sdf/body.ts';
import { vec, length } from '../src/avbd3d/ref/sdf/math.ts';

function setup() {
  const solver = new Solver(); solver.iterations = 15; solver.alpha = .95;
  const floor = new Rigid(solver, [12, 12, 1], 0, .5, [0, 0, -.5]);
  const adapter = new SdfCollisionAdapter(solver);
  adapter.attach(floor, new AnalyticField({ kind: 'box', halfExtents: [6, 6, .5] }));
  return { solver, adapter, floor };
}

test('SDF CPU contacts hold a falling three-box stack without changing the default backend', () => {
  const { solver, adapter } = setup();
  const boxes = [0, 1, 2].map(i => new Rigid(solver, [1, 1, 1], 1, .5, [0, 0, 1 + i * 1.05]));
  for (const b of boxes) adapter.attach(b, new AnalyticField({ kind: 'box', halfExtents: [.5, .5, .5] }));
  let contacts = 0;
  for (let i = 0; i < 360; i++) { solver.step(); contacts = Math.max(contacts, adapter.stats.contacts); }
  assert.ok(contacts >= 3);
  boxes.forEach((b, i) => {
    assert.ok(b.positionLin[2] > .4 + i * .8, `stack height ${i}: ${b.positionLin}`);
    assert.ok(length(vec(b.velocityLin)) < .2, `stack velocity ${i}: ${b.velocityLin}`);
    assert.ok(Array.from(b.positionAng).every(Number.isFinite));
  });
  adapter.dispose(); assert.equal(solver.collisionDetection, null);
  assert.equal(solver.forces.length, 0);
});

test('SDF CPU nonconvex torus falls and rests on the plane using per-contact normals', () => {
  const { solver, adapter } = setup();
  const body = new Rigid(solver, [1.4, .4, 1.4], 1, .5, [0, 0, 2]);
  adapter.attach(body, new AnalyticField({ kind: 'torus', majorRadius: .5, minorRadius: .2 }));
  for (let i = 0; i < 360; i++) solver.step();
  assert.ok(body.positionLin[2] > .15 && body.positionLin[2] < 1, `torus height ${body.positionLin[2]}`);
  assert.ok(length(vec(body.velocityLin)) < .2);
  assert.ok(adapter.stats.contacts > 0);
});

test('SDF adapter removes stale pairs, supports reattachment and refuses another backend', () => {
  const { solver, adapter, floor } = setup();
  const body = new Rigid(solver, [1, 1, 1], 1, .5, [0, 0, .49]);
  const field = new AnalyticField({ kind: 'sphere', radius: .5 }); adapter.attach(body, field);
  solver.step(); assert.ok(solver.forces.length > 0);
  body.positionLin[0] = 100; solver.step(); assert.equal(solver.forces.length, 0);
  body.positionLin.set([0, 0, .49]); adapter.attach(body, field); solver.step(); assert.ok(solver.forces.length > 0);
  assert.throws(() => new SdfCollisionAdapter(solver));
  assert.throws(() => adapter.attach(new Rigid(new Solver(), [1, 1, 1], 1, .5, [0, 0, 0]), field));
  solver.clear(); solver.step(); assert.equal(solver.forces.length, 0); assert.equal(body.forces.length, 0); assert.equal(floor.forces.length, 0);
  adapter.dispose(); solver.step();
});

test('SDF body transforms use center of mass and uniform scale, not shape origin', () => {
  const field = new AnalyticField({ kind: 'box', halfExtents: [.5, .5, .5] }); field.centerOfMass = [.1, .2, .3];
  const body = new SdfBody(field, [2, 3, 4], [0, 0, 0, 1], 2);
  const x = body.toWorld([.4, .3, .2]), local = body.toLocal(x);
  local.forEach((v, i) => assert.ok(Math.abs(v - [.4, .3, .2][i]) < 1e-6));
  for (const scale of [0, 1e-60, 1e40]) assert.throws(() => new SdfBody(field, [0, 0, 0], [0, 0, 0, 1], scale));
  assert.throws(() => new SdfBody(field, [1e40, 0, 0]));
  assert.throws(() => new SdfBody(field, [0, 0, 0], [1e30, 0, 0, 1]));
});

test('SDF collision preserves a torus hole instead of colliding against its bounding box', () => {
  const solver = new Solver(), adapter = new SdfCollisionAdapter(solver);
  const torus = new Rigid(solver, [1.4, .4, 1.4], 0, .5, [0, 0, 0]);
  torus.positionAng.set([Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
  adapter.attach(torus, new AnalyticField({ kind: 'torus', majorRadius: .5, minorRadius: .2 }));
  const sphere = new Rigid(solver, [.2, .2, .2], 1, .5, [0, 0, 2]);
  adapter.attach(sphere, new AnalyticField({ kind: 'sphere', radius: .1 }));
  let contacts = 0;
  for (let i = 0; i < 120; i++) { solver.step(); contacts += adapter.stats.contacts; }
  assert.equal(contacts, 0); assert.ok(sphere.positionLin[2] < -3);
});

test('mixed-size CPU bodies remain bounded over a thirty-second run', () => {
  const { solver, adapter } = setup();
  const sizes = [.25, 1, 2];
  const bodies = sizes.map((s, i) => {
    const b = new Rigid(solver, [s,s,s], s ** 3, .5, [(i-1)*3,0,3]);
    adapter.attach(b, new AnalyticField({ kind: 'box', halfExtents: [s/2,s/2,s/2] })); return b;
  });
  for (let frame = 0; frame < 1800; frame++) {
    solver.step();
    assert.ok(solver.forces.length <= 15, `unbounded contact forces: ${solver.forces.length}`);
    for (const body of bodies) assert.ok([...body.positionLin, ...body.velocityLin, ...body.positionAng].every(Number.isFinite));
  }
  bodies.forEach((b, i) => {
    assert.ok(Math.abs(b.positionLin[2] - sizes[i]/2) < .08, `settled height: ${b.positionLin[2]}`);
    assert.ok(length(vec(b.velocityLin)) < .2);
  });
  adapter.dispose(); assert.equal(solver.forces.length, 0);
});
