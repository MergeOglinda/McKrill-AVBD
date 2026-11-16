import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { World } from '../src/lib/world.ts';
import { AnalyticField } from '../src/lib/sdf.ts';
import { CONTACT_WORDS, K_LAM } from '../src/avbd3d/gpu/layout.ts';

const params={cacheSize:1,randomSamples:0,alpha:0};
gpuTest('SDF capacity growth preserves GPU warm starts and repairs overflow',async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  for(const x of [0,.6,4,4.6])world.addSphere({radius:.4,position:[x,0,0]});
  try {
    await world.enableSdfCollisions([],params,{maxPairs:1,autoGrow:false});
    world.solver.params.massPenalty=false;
    world.step();const first=await world.solver.readContactList();
    assert.equal(first.length,1);assert.equal((await world.solver.readCounters()).overflow&1,1);
    device.queue.writeBuffer(world.solver.contactStorage.contacts,K_LAM*4,new Float32Array([-7,0,0]));
    assert.equal(await world.adaptSdfCollisions(),true);
    assert.ok(world.sdfCollisionCapacity>=2);
    world.step();const next=await world.solver.readContactList();
    assert.equal(next.length,2);assert.equal((await world.solver.readCounters()).overflow,0);
    const kept=next.find(c=>c.a===first[0].a&&c.b===first[0].b)!;
    assert.ok(Math.abs(kept.lam[0]+7*world.solver.params.alpha*world.solver.params.gamma)<1e-4);
    assert.deepEqual(next.find(c=>c!==kept)!.lam,[0,0,0]);
    assert.equal(await world.adaptSdfCollisions(),false);
  } finally {world.destroy();}
});

gpuTest('World periodically grows SDF capacity, and opt-out leaves overflow visible',async device=>{
  for(const autoGrow of [true,false]) {
    const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
    for(const x of [0,.2,.4,.6])world.addSphere({radius:.4,position:[x,0,0]});
    try {
      await world.enableSdfCollisions([],params,{maxPairs:1,autoGrow});
      for(let i=0;i<10;i++)world.step();
      // Await queue progress and the existing periodic readback's promise continuation.
      await device.queue.onSubmittedWorkDone();
      for(let i=0;i<20&&autoGrow&&world.sdfCollisionCapacity===1;i++)
        await new Promise(resolve=>setTimeout(resolve,5));
      assert.equal(world.sdfCollisionError,null);
      assert.equal(world.sdfCollisionCapacity>=6,autoGrow);
      world.step();assert.equal((await world.solver.readCounters()).overflow&1,autoGrow?0:1);
    } finally {world.destroy();}
  }
});

gpuTest('SDF refresh handles append, sparse removal, slot reuse and unchanged warm starts',async device=>{
  const world=await World.create({device,maxBodies:8,gravity:[0,0,0],iterations:0});
  world.addSphere({radius:.4,position:[0,0,0]});
  const removed=world.addSphere({radius:.4,position:[.6,0,0]})!;
  world.addSphere({radius:.4,position:[4,0,0]});
  world.addSphere({radius:.4,position:[4.6,0,0]});
  try {
    await world.enableSdfCollisions([],params);
    world.solver.params.massPenalty=false;world.step();
    const first=await world.solver.readContactList();assert.equal(first.length,2);
    first.forEach((_,i)=>device.queue.writeBuffer(world.solver.contactStorage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-9,0,0])));
    removed.remove();
    assert.equal(world.sdfCollisionsNeedUpdate,true);
    assert.throws(()=>world.step(),/updateSdfCollisions/);
    await assert.rejects(()=>world.raycast([0,0,2],[0,0,-1]),/updateSdfCollisions/);
    await world.updateSdfCollisions();
    assert.equal(world.sdfCollisionsNeedUpdate,false);
    world.step();let contacts=await world.solver.readContactList();assert.equal(contacts.length,1);
    assert.equal(contacts[0].a,3);
    assert.ok(Math.abs(contacts[0].lam[0]+9*world.solver.params.alpha*world.solver.params.gamma)<1e-4);
    const replacement=world.addSphere({radius:.4,position:[.6,0,0]})!;
    assert.equal(replacement.index,removed.index);
    world.addSphere({radius:.4,position:[8,0,0]});
    world.addSphere({radius:.4,position:[8.6,0,0]});
    await world.updateSdfCollisions();assert.ok(world.sdfCollisionCapacity>=15);
    world.step();contacts=await world.solver.readContactList();
    assert.equal(contacts.length,3);assert.deepEqual(contacts.find(c=>c.a===replacement.index)!.lam,[0,0,0]);
    for(const body of world.bodies)body.remove();
    await world.updateSdfCollisions();world.step();
    assert.equal((await world.solver.readCounters()).pairs,0);
    assert.equal(await world.raycast([0,0,2],[0,0,-1]),null);
    assert.equal((await world.solver.readCounters()).overflow,0);
  } finally {world.destroy();}
});

gpuTest('SDF field updates snapshot inputs, invalidate changed pairs and retain unrelated history',async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  const a=world.addSphere({radius:.4,position:[0,0,0]})!;
  world.addSphere({radius:.4,position:[.6,0,0]});
  world.addSphere({radius:.4,position:[4,0,0]});
  world.addSphere({radius:.4,position:[4.6,0,0]});
  const field=new AnalyticField({kind:'sphere',radius:.4});
  try {
    await world.enableSdfCollisions([{body:a,field}],params);
    world.solver.params.massPenalty=false;world.step();
    const first=await world.solver.readContactList();
    first.forEach((_,i)=>device.queue.writeBuffer(world.solver.contactStorage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-5,0,0])));
    field.bounds.min=[-20,-20,-20];field.bounds.max=[20,20,20];
    if(field.spec.kind==='sphere')field.spec.radius=20;
    await world.updateSdfCollisions();world.step();
    assert.equal((await world.solver.readCounters()).pairs,2,'omitted binding keeps snapshot, not mutated caller field');
    const smaller=new AnalyticField({kind:'sphere',radius:.1});
    const updating=world.updateSdfCollisions([{body:a,field:smaller}]);
    if(smaller.spec.kind==='sphere')smaller.spec.radius=20;
    assert.throws(()=>world.step(),/updateSdfCollisions/);
    assert.throws(()=>a.remove(),/Await SDF/);
    await updating;world.step();
    const next=await world.solver.readContactList();assert.equal(next.length,1);assert.equal(next[0].a,3);
    assert.ok(next[0].lam[0]<0,'unrelated pair retains history');
    await world.updateSdfCollisions([{body:a,field:new AnalyticField({kind:'sphere',radius:.4})}]);
    world.step();assert.deepEqual((await world.solver.readContactList()).find(c=>c.a===1)!.lam,[0,0,0]);
  } finally {world.destroy();}
});

gpuTest('SDF failed updates preserve the old backend and async teardown is safe',async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  const a=world.addSphere({radius:.4})!;
  world.addSphere({radius:.4,position:[.6,0,0]});
  let destroyed=false;
  try {
    await world.enableSdfCollisions([],params);
    await assert.rejects(()=>world.updateSdfCollisions([{body:a,field:new AnalyticField({kind:'sphere',radius:.4}),scale:-1}]),/scale/);
    world.step();assert.equal((await world.solver.readContactList()).length,1);
    const adapting=world.adaptSdfCollisions();world.disableSdfCollisions();
    assert.equal(await adapting,false);
    await world.enableSdfCollisions([],params);
    const updating=world.updateSdfCollisions();
    world.destroy();destroyed=true;
    await assert.rejects(()=>updating,/destroyed/);
  } finally {if(!destroyed)world.destroy();}
});

gpuTest('SDF growth migrates history when production contact storage also reallocates',async device=>{
  const world=await World.create({device,maxBodies:65,gravity:[0,0,0],iterations:0});
  for(let i=0;i<65;i++)world.addSphere({radius:.4,position:[i*.001,0,0]});
  try {
    await world.enableSdfCollisions([],params,{maxPairs:1,autoGrow:false});
    world.solver.params.massPenalty=false;world.step();
    const [first]=await world.solver.readContactList();assert.ok(first);
    const oldBuffer=world.solver.contactStorage.contacts;
    device.queue.writeBuffer(oldBuffer,K_LAM*4,new Float32Array([-6,0,0]));
    await world.adaptSdfCollisions();
    assert.notEqual(world.solver.contactStorage.contacts,oldBuffer);
    world.step();
    const contacts=await world.solver.readContactList();assert.equal(contacts.length,65*64/2);
    const kept=contacts.find(c=>c.a===first.a&&c.b===first.b)!;
    // Deeply overlapping spheres can rotate the optimizer's normal between steps.
    // Compare the complete world-space force, rather than assuming an unchanged basis.
    const n=kept.normal,t=Math.abs(n[0])>Math.abs(n[2])?[-n[1],n[0],0]:[0,-n[2],n[1]];
    const len=Math.hypot(...t),u=t.map(v=>v/len),v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];
    const gain=world.solver.params.alpha*world.solver.params.gamma;
    for(let k=0;k<3;k++)assert.ok(Math.abs(n[k]*kept.lam[0]+u[k]*kept.lam[1]+v[k]*kept.lam[2]+6*gain*first.normal[k])<1e-4);

    assert.equal((await world.solver.readCounters()).overflow,0);
  } finally {world.destroy();}
});

gpuTest('SDF same-step recycled slots and replaced overlapping fields never inherit old forces',async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  const a=world.addSphere({radius:.4})!;
  let b=world.addSphere({radius:.4,position:[.6,0,0]})!;
  try {
    await world.enableSdfCollisions([],params);
    world.solver.params.massPenalty=false;
    world.step();assert.equal((await world.solver.readContactList()).length,1);
    device.queue.writeBuffer(world.solver.contactStorage.contacts,K_LAM*4,new Float32Array([-8,0,0]));
    const slot=b.index;b.remove();b=world.addSphere({radius:.4,position:[.6,0,0]})!;assert.equal(b.index,slot);
    await world.updateSdfCollisions();world.step();
    assert.deepEqual((await world.solver.readContactList())[0].lam,[0,0,0]);
    device.queue.writeBuffer(world.solver.contactStorage.contacts,K_LAM*4,new Float32Array([-8,0,0]));
    await world.updateSdfCollisions([{body:a,field:new AnalyticField({kind:'sphere',radius:.41})}]);
    world.step();assert.deepEqual((await world.solver.readContactList())[0].lam,[0,0,0]);
    for(const maxPairs of [0,-1,1.5,NaN,1e12])
      await assert.rejects(()=>world.updateSdfCollisions([],{maxPairs}),/capacity/);
    world.step();assert.equal((await world.solver.readContactList()).length,1);
  } finally {world.destroy();}
});

gpuTest('SDF refresh replaces a newly staged body in its pending slot',async device=>{
  const world=await World.create({device,maxBodies:5,gravity:[0,0,0],iterations:0});
  world.addSphere({radius:.4,fixed:true});
  try {
    await world.enableSdfCollisions([],params);
    const staged=world.addSphere({radius:.4,position:[10,0,0]})!;
    staged.remove();
    const replacement=world.addSphere({radius:.4,position:[.6,0,0]})!;
    assert.equal(replacement.index,staged.index);
    await world.updateSdfCollisions();
    assert.equal(world.solver.bodyCount,2,'reused pending slot must not append an extra GPU body');
    world.step();
    assert.equal((await world.solver.readContactList()).length,1);
    const hit=await world.raycast([.6,2,0],[0,-1,0]);
    assert.equal(hit?.body,replacement);
  } finally {world.destroy();}
});

gpuTest('SDF pending rays retain submitted geometry across refresh and destruction',async device=>{
  const world=await World.create({device,maxBodies:2,gravity:[0,0,0],iterations:0});
  const body=world.addSphere({radius:1,fixed:true})!;
  let destroyed=false;
  try {
    await world.enableSdfCollisions([],params);
    const pending=world.raycast([0,3,0],[0,-1,0]);
    await world.updateSdfCollisions([{body,field:new AnalyticField({kind:'sphere',radius:.2})}]);
    const hit=await pending;
    assert.equal(hit?.body,body);assert.ok(Math.abs(hit!.distance-2)<1e-4);
    const next=await world.raycast([0,3,0],[0,-1,0]);
    assert.ok(Math.abs(next!.distance-2.8)<1e-4);
    const last=world.raycast([0,3,0],[0,-1,0]);
    world.destroy();destroyed=true;
    assert.ok(Math.abs((await last)!.distance-2.8)<1e-4);
  } finally {if(!destroyed)world.destroy();}
});

gpuTest('SDF rejected growth leaves the installed backend usable',async device=>{
  const world=await World.create({device,maxBodies:2,gravity:[0,0,0],iterations:0});
  world.addSphere({radius:.4});world.addSphere({radius:.4,position:[.6,0,0]});
  const read=world.solver.readCounters.bind(world.solver);
  try {
    await world.enableSdfCollisions([],params,{maxPairs:1,autoGrow:false});world.step();
    const counters=await read();
    // Simulate an observed candidate count beyond binding limits without allocating it.
    world.solver.readCounters=async()=>({...counters,pairs:0xffffffff,overflow:1});
    await assert.rejects(()=>world.adaptSdfCollisions(),/device limits/);
    world.solver.readCounters=read;
    assert.equal(world.sdfCollisionCapacity,1);
    world.step();assert.equal((await world.solver.readContactList()).length,1);
    assert.equal((await read()).overflow,0);
  } finally {world.solver.readCounters=read;world.destroy();}
});

gpuTest('SDF growth cleans temporary buffers when a host allocation throws',async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  for(const x of [0,.6,4,4.6])world.addSphere({radius:.4,position:[x,0,0]});
  const create=device.createBuffer.bind(device);
  const allocated=new Set<GPUBuffer>(),destroyed=new Set<GPUBuffer>();
  try {
    await world.enableSdfCollisions([],params,{maxPairs:1,autoGrow:false});world.step();
    device.createBuffer=(descriptor)=>{
      if(descriptor.label==='sdf results 1')throw new Error('Injected host allocation failure');
      const buffer=create(descriptor),destroy=buffer.destroy.bind(buffer);
      allocated.add(buffer);buffer.destroy=()=>{destroyed.add(buffer);destroy();};
      return buffer;
    };
    // Includes the counter staging read, then spatial buffers and part of the new pass.
    await assert.rejects(()=>world.adaptSdfCollisions(),/Injected host allocation failure/);
    device.createBuffer=create;
    assert.ok(allocated.size>3);assert.equal(destroyed.size,allocated.size);
    assert.equal(world.sdfCollisionCapacity,1);
    world.step();assert.equal((await world.solver.readContactList()).length,1);
    assert.equal(await world.adaptSdfCollisions(),true);
    world.step();assert.equal((await world.solver.readCounters()).overflow,0);
  } finally {device.createBuffer=create;world.destroy();}
});

gpuTest('SDF worlds keep adapting the colouring, with or without capacity growth',async device=>{
  for(const autoGrow of [true,false]) {
    const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
    for(const x of [0,.2,.4,.6])world.addSphere({radius:.4,position:[x,0,0]});
    try {
      await world.enableSdfCollisions([],params,{maxPairs:6,autoGrow});
      const seen:number[]=[];const adaptColors=world.solver.adaptColors.bind(world.solver);
      world.solver.adaptColors=counters=>{seen.push(counters.colors);adaptColors(counters);};
      for(let i=0;i<10;i++)world.step();
      await device.queue.onSubmittedWorkDone();
      for(let i=0;i<20&&!seen.length;i++)await new Promise(resolve=>setTimeout(resolve,5));
      assert.equal(seen.length,1,`autoGrow ${autoGrow}: one periodic colour adaptation`);
      assert.ok(seen[0]>0,'contacting spheres need colours');
    } finally {world.destroy();}
  }
});

gpuTest('detaching SDF collisions makes the next reusing step detect contacts afresh',async device=>{
  const world=await World.create({device,maxBodies:2,gravity:[0,0,0],iterations:0});
  world.addSphere({radius:.4,position:[0,0,0]});world.addSphere({radius:.4,position:[.6,0,0]});
  try {
    await world.enableSdfCollisions([],params,{maxPairs:1});
    world.step();assert.equal((await world.solver.readContactList()).length,1);
    world.disableSdfCollisions();
    // SDF contacts are not built-in manifolds; reuse must not replay them (or nothing).
    world.step({reuseManifolds:true});
    assert.ok((await world.solver.readContactList()).length>0,'built-in narrowphase found the overlap');
  } finally {world.destroy();}
});
