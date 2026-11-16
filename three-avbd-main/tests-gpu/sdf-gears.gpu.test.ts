import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { World } from '../src/lib/world.ts';
import { prepareGears, createGearTrain, GearDrive, angleZ, angleDelta, CENTERS, TEETH } from '../src/sdf-gears/model.ts';

gpuTest('SDF gear teeth drive passive spindle followers, reverse and disconnect', async device => {
  const gears = await prepareGears();
  const world = await World.create({ device, maxBodies: 4, gravity: [0, 0, 0], iterations: 20, dt: 1 / 120 });
  try {
    const { bodies } = await createGearTrain(world, gears);
    const drive = new GearDrive(bodies[0]);
    let previous = bodies.map(angleZ);
    let loaded = 0, worstDrift = 0;
    async function run(steps: number) {
      const total = [0, 0, 0];
      for (let i = 0; i < steps; i++) {
        drive.step(world.dt); world.step();
        if (i % 12 !== 11) continue;
        await world.read();
        const angles = bodies.map(angleZ);
        for (let j = 0; j < bodies.length; j++) {
          const body = bodies[j];
          assert.ok([...body.position, ...body.rotation, ...body.angularVelocity].every(Number.isFinite));
          total[j] += angleDelta(angles[j], previous[j]);
          worstDrift = Math.max(worstDrift, Math.hypot(...body.position.map((v, k) => v - CENTERS[j][k])));
          assert.ok(Math.hypot(body.rotation[0], body.rotation[1]) < .025, 'axles remain aligned');
        }
        previous = angles;
        if (i % 60 === 59) {
          assert.equal((await world.solver.readCounters()).overflow, 0);
          loaded += (await world.solver.readContactList()).filter(c => Math.abs(c.lam[0]) > .01).length;
        }
      }
      return total;
    }
    const assertRatio = (angles: number[]) => {
      assert.ok(Math.abs(angles[0]) > 1, 'input turns far enough to cross several teeth');
      for (let i = 1; i < angles.length; i++) {
        const expected = angles[0] * TEETH[0] / TEETH[i] * (i % 2 ? -1 : 1);
        assert.ok(Math.abs(angles[i] - expected) < .14, `gear ${i}: actual ${angles[i]}, expected ${expected}`);
      }
    };
    assertRatio(await run(480));
    drive.speed = -.5;
    assertRatio(await run(480));
    assert.ok(loaded > 10, 'solver normal loads are present at the meshing teeth');
    assert.ok(worstDrift < .035, `spindle drift ${worstDrift}`);
    drive.automatic = false; drive.target = drive.angle + .8;
    const manual = await run(360);
    assert.ok(Math.abs(drive.angle - drive.target) < .01, 'manual input reaches its requested angle');
    assert.ok(manual[1] < -.8 && manual[2] > .3, 'manual handwheel drives both followers');
    drive.automatic = true; drive.speed = 8 * 2 * Math.PI / 60;
    assertRatio(await run(480));
    assert.ok(worstDrift < .035, `spindle drift at maximum speed ${worstDrift}`);
    // A negative control: motor still turns, but disabling tooth collisions cannot move followers.
    bodies.forEach(body => body.setCollisionGroups(1, 0));
    for (let i = 1; i < bodies.length; i++) bodies[i].set({ velocity: [0, 0, 0], angularVelocity: [0, 0, 0] });
    drive.automatic = true; drive.speed = .5;
    await run(60); // relax residual constraint forces before measuring the stationary control
    const free = await run(240);
    assert.ok(free[0] > .7);
    assert.ok(Math.abs(free[1]) < .04 && Math.abs(free[2]) < .04, `disconnected followers: ${free}`);
    assert.equal((await world.solver.readContactList()).length, 0);
  } finally { world.destroy(); gears.forEach(gear => gear.geometry.dispose()); }
});
