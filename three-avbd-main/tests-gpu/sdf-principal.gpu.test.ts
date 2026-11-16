import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { World } from '../src/lib/world.ts';
import { AnalyticField } from '../src/lib/sdf.ts';
import { B_SIZE, B_MOMENT } from '../src/avbd3d/gpu/layout.ts';

gpuTest('SDF rotated field keeps ray hits, contact anchors and warm starts in the principal body frame', async device => {
  const world = await World.create({device, maxBodies: 4});
  try {
    const body = world.addBox({size: [1,1,1], position: [0, .48, 0], massProperties: {mass: 3, moments: [1,2,3]}})!;
    world.addBox({size: [8,.5,8], position: [0,-.25,0], fixed: true});
    const q: [number,number,number,number] = [0,0,Math.SQRT1_2,Math.SQRT1_2];
    const field = new AnalyticField({kind: 'box', halfExtents: [.5,1,.3]});
    await world.enableSdfCollisions([{body,field,rotation:q}]);
    q[2] = 0; q[3] = 1; // binding owns its transform
    const hit = await world.raycast([3,.48,0],[-1,0,0]);
    assert.equal(hit?.body,body); assert.ok(Math.abs(hit!.distance - 2)<.002);
    for (let i=0;i<20;i++) world.step();
    await world.read();
    const data=await world.solver.readBodies();
    assert.equal(data[B_SIZE+3],3); assert.deepEqual(Array.from(data.slice(B_MOMENT,B_MOMENT+3)),[1,2,3]);
    const contacts=await world.solver.readContactList(); assert.ok(contacts.length>0);
    for (const p of contacts) {
      const a=p.a===body.index?p.rA:p.rB;
      assert.ok(Math.abs(a[1])<.53 && Math.abs(a[0])<1.03, 'anchors use rotated body frame');
    }
    assert.ok(contacts.some(p=>Math.abs(p.lam[0])>0),'solved warm force');
    await world.updateSdfCollisions(); world.step();
    assert.ok((await world.solver.readContactList()).some(p=>Math.abs(p.lam[0])>0));
    body.remove();
    assert.throws(()=>world.addBox({size:[1,1,1],massProperties:{mass:1,moments:[0,1,1]}}),/positive/);
    assert.equal(world.addBox({size:[1,1,1]})!.index,body.index,'invalid mass does not consume a free slot');
  } finally {world.destroy();}
});
