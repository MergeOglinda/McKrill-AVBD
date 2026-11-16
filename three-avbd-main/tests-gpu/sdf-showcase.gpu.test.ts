import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { World, type SdfBodyBinding } from '../src/lib/world.ts';
import { showcaseShapes } from '../src/sdf-showcase/shapes.ts';

gpuTest('small showcase pile settles with all three fields, spawning and a drag joint', async device => {
  const shapes = await showcaseShapes();
  const world = await World.create({ device, maxBodies: 10, iterations: 12 });
  try {
    world.addBox({ size: [20, .5, 20], position: [0, -.25, 0], fixed: true });
    const handle = world.addSphere({ radius: .05, fixed: true, group: 0, collidesWith: 0 })!;
    const bindings: SdfBodyBinding[] = Object.values(shapes).map((shape, i) => ({
      body: world.addBox({ size: [1, 1, 1], position: [(i - 1) * 3, 1, 0], density: 2 })!, field: shape.field,
    }));
    await world.enableSdfCollisions(bindings, {}, { maxPairs: 45 });
    for (let i = 0; i < 90; i++) world.step();
    await world.read();
    for (const { body } of bindings) {
      assert.ok(body.position.every(Number.isFinite));
      assert.ok(body.position[1] > .05 && body.position[1] < 1, 'body remains above ground');
    }
    assert.ok((await world.solver.readContactList()).length >= 3);
    const spawned = world.addBox({ size: [1, 1, 1], position: [0, 4, 0] })!;
    await world.updateSdfCollisions([{ body: spawned, field: shapes.gears.field }]);
    handle.set({ position: [0, 4, 0] });
    const joint = world.addJoint(handle, spawned, { type: 'ball' });
    for (let i = 0; i < 20; i++) { handle.moveTo([i * .02, 4, 0]); world.step(); }
    await world.read(); assert.ok(spawned.position[0] > .2, 'joint pulls spawned body');
    joint.remove();
    for (const { body } of bindings) body.remove(); spawned.remove();
    await world.updateSdfCollisions(); world.step();
    assert.equal((await world.solver.readCounters()).overflow, 0);
  } finally { world.destroy(); for (const shape of Object.values(shapes)) shape.geometry.dispose(); }
});
