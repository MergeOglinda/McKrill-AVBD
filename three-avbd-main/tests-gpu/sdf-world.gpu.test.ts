import assert from 'node:assert/strict';
import { BoxGeometry, Quaternion, Vector3 } from 'three';
import { gpuTest } from './device.ts';
import { World, type ContactEvent } from '../src/lib/world.ts';
import { AnalyticField, VoxelField, bakeVoxelSdf } from '../src/lib/sdf.ts';
import { CONTACT_WORDS, K_LAM } from '../src/avbd3d/gpu/layout.ts';

gpuTest('World automatic SDF candidates honor masks, joint release and GPU-broken joints', async device=>{
  const world=await World.create({device,maxBodies:8,gravity:[0,0,0],iterations:0});
  const floor=world.addBox({size:[8,.4,8],position:[0,-.2,0],fixed:true})!;
  const a=world.addSphere({radius:.4,position:[-1,.25,0]})!;
  world.addSphere({radius:.4,position:[1,.25,0]});
  try {
    await world.enableSdfCollisions([],{cacheSize:1,randomSamples:0,alpha:0});
    const count=async()=>{world.step();const c=await world.solver.readCounters();assert.equal(c.overflow,0);return c.pairs;};
    assert.equal(await count(),2,'two sphere/ground AABB pairs, spheres are separated');
    assert.equal((await world.solver.readContactList()).length,2,'both overlapping pairs have active contacts');
    for(let i=0;i<2;i++)device.queue.writeBuffer(world.solver.contactStorage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-3,0,0]));
    a.setCollisionGroups(1,0);assert.equal(await count(),1,'mask excludes pair');
    a.setCollisionGroups(1,0xffffffff);assert.equal(await count(),2);
    assert.deepEqual((await world.solver.readContactList()).find(c=>c.a===a.index)!.lam,[0,0,0],'filtered pair discards stale force cache');
    const joint=world.addJoint(floor,a,{type:'ball'})!;
    assert.equal(await count(),1,'live joint excludes connected bodies');
    world.solver.disableConstraint(joint.slot);
    assert.equal(await count(),2,'GPU-disabled/broken joint no longer excludes pair');
    joint.remove();assert.equal(await count(),2,'released slot removed from exclusions');
    a.set({position:[20,.35,0]});assert.equal(await count(),1,'GPU bounds reject moved body');
    a.set({position:[-1,.25,0]});assert.equal(await count(),2,'re-entry automatically generates pair');
    const extra=world.addSphere({radius:.1,position:[30,0,0]})!;
    assert.throws(()=>world.step(),/updateSdfCollisions/);
    await world.updateSdfCollisions();extra.remove();await world.updateSdfCollisions();
    a.reportContacts=true;assert.equal(a.reportContacts,true);
    await assert.rejects(()=>world.enableSdfCollisions(),/already enabled/);
    world.disableSdfCollisions();assert.equal(world.sdfCollisionsEnabled,false);
    a.remove();world.step();assert.equal(a.alive,false);
  } finally {world.destroy();}
});

gpuTest('World SDF voxel binding settles with automatic candidates and no pose readback during stepping', async device=>{
  const world=await World.create({device,maxBodies:4});
  world.addBox({size:[8,.4,8],position:[0,-.2,0],fixed:true});
  // Original collision box is 1m; the SDF is a uniformly scaled 0.8m cube.
  const body=world.addBox({size:[1,1,1],position:[0,.9,0],density:1})!;
  const geometry=new BoxGeometry();const field=bakeVoxelSdf(geometry,{resolution:12}).field;geometry.dispose();
  try {
    await world.enableSdfCollisions([{body,field,scale:.8}]);
    assert.equal(world.sdfCollisionsEnabled,true);
    for(let i=0;i<120;i++) {world.step();if(i%20===19)await device.queue.onSubmittedWorkDone();}
    await world.read();
    assert.ok(body.position.every(Number.isFinite));
    assert.ok(body.position[1]>.35&&body.position[1]<.46,`SDF cube height ${body.position[1]}`);
    assert.ok((await world.solver.readContactList()).length>0);
    assert.equal((await world.solver.readCounters()).overflow,0);
  } finally {world.destroy();}
});

gpuTest('World SDF setup validates bindings and leaves topology usable after failure or cancellation', async device=>{
  const world=await World.create({device,maxBodies:4});
  const body=world.addBox({size:[1,1,1],reportContacts:true})!;
  let destroyed=false;
  try {
    const geometry=new BoxGeometry();const field=bakeVoxelSdf(geometry,{resolution:4}).field;geometry.dispose();
    await assert.rejects(()=>world.enableSdfCollisions([{body,field},{body,field}]),/duplicate/);
    const extra=world.addSphere({radius:.1})!;assert.ok(extra);
    const enabling=world.enableSdfCollisions();
    assert.throws(()=>body.remove(),/Await SDF/);assert.equal(body.alive,true);
    world.destroy();destroyed=true;
    await assert.rejects(()=>enabling,/destroyed/);
  } finally {if(!destroyed)world.destroy();}
});

gpuTest('World automatic SDF reserves spatial capacity and supports more than 64 bodies', async device=>{
  const world=await World.create({device,maxBodies:65,gravity:[0,0,0],iterations:0});
  for(let i=0;i<64;i++) world.addBox({size:[.1,.1,.1],position:[i,0,0],fixed:true});
  try {
    await world.enableSdfCollisions();
    assert.ok(world.solver.manifoldCapacity>=64*8*5,'capacity reserved before dispatch');
    world.step();const counters=await world.solver.readCounters();
    assert.equal(counters.pairs,0,'static/static pairs rejected');assert.equal(counters.contacts,0);assert.equal(counters.overflow,0);
    world.disableSdfCollisions();world.addBox({size:[.1,.1,.1],position:[65,0,0]});
    await world.enableSdfCollisions();
    world.step();assert.equal((await world.solver.readCounters()).overflow,0);
    assert.equal(world.sdfCollisionsEnabled,true);
  } finally {world.destroy();}
});


gpuTest('World SDF emits one pair event for independent points and ends on filtering', async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0],iterations:0});
  world.addBox({size:[5,1,5],position:[0,-.5,0],fixed:true});
  const body=world.addBox({size:[1,1,1],position:[0,.4,0],reportContacts:true})!;
  const events:ContactEvent[]=[];world.onContact(e=>events.push(e));
  try {
    await world.enableSdfCollisions([],{skipConvexSamples:false,cacheSize:4,randomSamples:1});
    world.step();await world.read();
    assert.ok((await world.solver.readContactList()).length>1,'multiple independent solver contacts');
    assert.equal(events.length,1);assert.equal(events[0].type,'begin');
    assert.ok(events[0].normal[1]>.99);assert.ok(events[0].point.every(Number.isFinite));
    world.step();await world.read();assert.equal(events.length,1,'no duplicate begins');
    body.setCollisionGroups(1,0);world.step();await world.read();
    assert.deepEqual(events.map(e=>e.type),['begin','end']);
    body.setCollisionGroups(1,0xffffffff);world.step();await world.read();
    assert.deepEqual(events.map(e=>e.type),['begin','end','begin']);
    body.reportContacts=false;world.step();await world.read();assert.equal(events.at(-1)!.type,'end');
    const contacts=await world.solver.readContactList();
    for(let i=0;i<contacts.length;i++)device.queue.writeBuffer(world.solver.contactStorage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-3,0,0]));
    body.reportContacts=true;world.step();await world.read();
    const solved=await world.solver.readContactList();
    const expected=solved.reduce((sum,c)=>sum+Math.abs(c.lam[0]),0)*world.dt;
    assert.ok(expected>0,'cached normal forces survived');
    assert.ok(Math.abs(events.at(-1)!.impulse-expected)<1e-5,'pair event sums all normal-force magnitudes');
    body.set({position:[0,3,0]});world.step();await world.read();assert.equal(events.at(-1)!.type,'end');
    body.set({position:[0,.4,0]});world.step();await world.read();assert.equal(events.at(-1)!.type,'begin');
    const before=events.length;world.disableSdfCollisions();world.step();await world.read();
    assert.equal(events.length,before,'backend switch preserves pair identity');
    assert.equal(world.droppedContactEvents,0);
  } finally {world.destroy();}
});

gpuTest('World SDF rays see torus holes, scale, current poses, masks, ignores and nearest ties', async device=>{
  const world=await World.create({device,maxBodies:4,gravity:[0,0,0]});
  const a=world.addBox({size:[8,8,8],fixed:true})!;
  const b=world.addBox({size:[8,8,8],fixed:true})!;
  const field=new AnalyticField({kind:'torus',majorRadius:1,minorRadius:.25});
  try {
    await world.enableSdfCollisions([{body:a,field,scale:2},{body:b,field,scale:2}]);
    const rays=[{origin:[0,4,0],direction:[0,-2,0]},{origin:[2,4,0],direction:[0,-2,0]}];
    const [hole,hit]=await world.raycasts(rays);
    assert.equal(hole,null,'ray passes through non-convex hole, not original box');
    assert.equal(hit!.body,a,'equal distance picks lower slot');assert.ok(Math.abs(hit!.distance-3.5)<.002);
    assert.ok(hit!.normal[1]>.999);
    assert.equal((await world.raycasts([rays[1]],{ignore:[a]}))[0]!.body,b);
    a.setCollisionGroups(2,0xffffffff);b.setCollisionGroups(4,0xffffffff);
    assert.equal((await world.raycasts([rays[1]],{collidesWith:4}))[0]!.body,b);
    assert.equal(await world.raycast([2,4,0],[0,-1,0],{maxDistance:3}),null);
    assert.equal((await world.raycast([2,0,0],[0,1,0]))!.distance,0);
    a.set({position:[10,0,0]});b.set({position:[20,0,0]});
    assert.equal(await world.raycast([2,4,0],[0,-1,0]),null,'writes visible without stepping');
    assert.equal((await world.raycast([12,4,0],[0,-1,0]))!.body,a);
    await assert.rejects(()=>world.raycast([0,0,0],[0,0,0]),/nonzero/);
    world.disableSdfCollisions();
    assert.ok(await world.raycast([10,5,0],[0,-1,0]),'standard shape restored');
  } finally {world.destroy();}
});

gpuTest('World SDF rays sample baked fields with COM offset and rotated uniform scale', async device=>{
  const world=await World.create({device,maxBodies:2,gravity:[0,0,0]});
  const geometry=new BoxGeometry(2,1,1);geometry.translate(3,0,0);
  const field=bakeVoxelSdf(geometry,{resolution:12}).field;field.centerOfMass=[3,0,0];geometry.dispose();
  const body=world.addBox({size:[.1,.1,.1],position:[5,0,0],rotation:[0,0,Math.SQRT1_2,Math.SQRT1_2],fixed:true})!;
  try {
    await world.enableSdfCollisions([{body,field,scale:2}]);
    const hit=await world.raycast([5,5,0],[0,-1,0]);
    assert.equal(hit!.body,body);assert.ok(Math.abs(hit!.distance-3)<.03,`distance ${hit!.distance}`);
    assert.ok(hit!.normal[1]>.99);
  } finally {world.destroy();}
});


gpuTest('World SDF ray handles a steep voxel field without stepping through its surface', async device=>{
  const world=await World.create({device,maxBodies:1,gravity:[0,0,0]});
  const body=world.addBox({size:[.1,.1,.1],fixed:true})!;
  const grid=new Float32Array(9*2*2*4);
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<9;x++) {
    const px=-1+x*.25;grid.set([20*(Math.abs(px)-.2),Math.sign(px),0,0],4*(x+9*(y+2*z)));
  }
  const field=new VoxelField([9,2,2],[-1,0,0],.25,grid,{min:[-1,0,0],max:[1,.25,.25]});
  try {
    await world.enableSdfCollisions([{body,field}]);
    const hit=await world.raycast([2,.1,.1],[-1,0,0]);
    assert.ok(hit);assert.ok(Math.abs(hit.distance-1.8)<.001);assert.ok(hit.normal[0]>.99);
    await assert.rejects(()=>world.raycasts(Array.from({length:257},()=>({origin:[0,0,0],direction:[1,0,0]}))),/at most 256/);
  } finally {world.destroy();}
});

gpuTest('World SDF pending rays retain submitted inputs and body identity across disabling', async device=>{
  const world=await World.create({device,maxBodies:2,gravity:[0,0,0]});
  const body=world.addSphere({radius:1,fixed:true})!;
  try {
    await world.enableSdfCollisions();
    const origin=[0,3,0];const rays=[{origin,direction:[0,-1,0]}];
    const pending=world.raycasts(rays);
    origin[0]=100;rays.length=0;
    world.disableSdfCollisions();body.remove();
    const replacement=world.addSphere({radius:.1,position:[100,0,0],fixed:true})!;
    assert.equal(replacement.index,body.index);
    const hits=await pending;
    assert.equal(hits.length,1,'input array mutations cannot change submitted batch size');
    assert.equal(hits[0]!.body,body,'result belongs to the body present when submitted');
    assert.ok(Math.abs(hits[0]!.point[0])<1e-6);
    assert.ok(Math.abs(hits[0]!.point[1]-1)<.002);
  } finally {world.destroy();}
});

gpuTest('World SDF ray slab clipping preserves small nonzero local directions', async device=>{
  const world=await World.create({device,maxBodies:1,gravity:[0,0,0]});
  const body=world.addSphere({radius:1,fixed:true})!;
  try {
    await world.enableSdfCollisions([{body,field:new AnalyticField({kind:'sphere',radius:1}),scale:1e13}]);
    const scaled=await world.raycast([0,3e13,0],[0,-1,0]);
    assert.ok(scaled,'a large uniform scale must not turn the ray into a parallel ray');
    assert.ok(Math.abs(scaled.distance/2e13-1)<.001);
    world.disableSdfCollisions();
    await world.enableSdfCollisions([{body,field:new AnalyticField({kind:'box',halfExtents:[1,1,1e14]})}]);
    const shallow=await world.raycast([2,0,0],[-1e-13,0,1],{maxDistance:2e13});
    assert.ok(shallow,'shallow direction reaches the long field slab');
    assert.ok(Math.abs(shallow.distance/1e13-1)<.001);
    assert.equal(await world.raycast([2,0,0],[0,0,1],{maxDistance:2e13}),null,'truly parallel ray misses');
  } finally {world.destroy();}
});

// Analytic rolling bodies refresh contact anchors; flat-resting ones must still hold by static friction.
const cylinderWorld=async(device:GPUDevice,floorAngle:number,pose:{position:[number,number,number];rotation:[number,number,number,number];velocity?:[number,number,number];angularVelocity?:[number,number,number]})=>{
  const world=await World.create({device,maxBodies:2});
  const s=Math.sin(floorAngle/2),c=Math.cos(floorAngle/2);
  world.addBox({size:[20,.4,20],position:[0,-.2,0],rotation:[0,0,s,c],fixed:true,friction:.6});
  const r=.3,h=.25,mass=1,moments:[number,number,number]=[mass*(3*r*r+4*h*h)/12,mass*r*r/2,mass*(3*r*r+4*h*h)/12];
  const body=world.addBox({size:[1,1,1],...pose,friction:.6,massProperties:{mass,moments}})!;
  await world.enableSdfCollisions([{body,field:new AnalyticField({kind:'cylinder',radius:r,halfHeight:h})}]);
  return {world,body,r,h};
};

gpuTest('World SDF analytic cylinder rolls along the floor without sinking', async device=>{
  const q=Math.SQRT1_2,{world,body,r}=await cylinderWorld(device,0,{position:[0,.3005,0],rotation:[q,0,0,q],velocity:[1,0,0],angularVelocity:[0,0,-1/.3]});
  try {
    let lowest=Infinity;
    for(let frame=0;frame<240;frame++){world.step();if(frame%10===9){await world.read();lowest=Math.min(lowest,body.position[1]);}}
    await world.read();
    assert.ok(body.position[0]>.5,`it rolled (${body.position[0]} m)`);
    assert.ok(lowest>r-.005,`axis height stayed at the radius: lowest ${lowest}`);
  } finally {world.destroy();}
});

gpuTest('World SDF analytic cylinder standing on its cap holds on a 10 degree incline', async device=>{
  const angle=10*Math.PI/180,s=Math.sin(angle/2),c=Math.cos(angle/2);
  // Cap centre on the incline surface (which passes through the origin), axis along its normal.
  const normal:[number,number,number]=[-Math.sin(angle),Math.cos(angle),0],h=.25+.0005;
  const {world,body}=await cylinderWorld(device,angle,{position:normal.map(v=>v*h) as [number,number,number],rotation:[0,0,s,c]});
  try {
    for(let frame=0;frame<120;frame++)world.step();
    await world.read();
    const drift=Math.hypot(body.position[0]-normal[0]*h,body.position[1]-normal[1]*h,body.position[2]);
    assert.ok(drift<.01,`static friction holds the cylinder (drift ${drift} m)`);
  } finally {world.destroy();}
});

gpuTest('World SDF tilted cylinder and capsule tumble onto an edge and stay at rest above the floor', async device=>{
  for(const spec of [{kind:'cylinder',radius:.3,halfHeight:.25},{kind:'capsule',radius:.3,halfLength:.25}] as const){
    const world=await World.create({device,maxBodies:2});
    world.addBox({size:[20,.4,20],position:[0,-.2,0],fixed:true,friction:.6});
    const t=Math.sin(Math.PI/12),c=Math.cos(Math.PI/12),mass=1,moments:[number,number,number]=[.04,.045,.04];
    // 30 degrees off vertical, so it lands on an edge and tips over.
    const body=world.addBox({size:[1,1,1],position:[0,.9,0],rotation:[0,0,t,c],friction:.6,massProperties:{mass,moments}})!;
    await world.enableSdfCollisions([{body,field:new AnalyticField(spec)}]);
    try {
      const energy=()=>{
        // Body-frame angular velocity: moments are principal, ω is world-space.
        const w=new Vector3(...body.angularVelocity).applyQuaternion(new Quaternion(...body.rotation).invert());
        return .5*mass*body.velocity.reduce((s,v)=>s+v*v,0)+.5*(moments[0]*w.x*w.x+moments[1]*w.y*w.y+moments[2]*w.z*w.z);
      };
      let lowest=Infinity,after=0;
      for(let frame=0;frame<300;frame++){
        world.step();
        if(frame%10===9){await world.read();lowest=Math.min(lowest,body.position[1]);
          if(frame>=60)after=Math.max(after,energy());}
      }
      await world.read();
      // Lowest resting heights: cylinder on its side (r=.3) or cap (h=.25); capsule on its side (r=.3).
      assert.ok(lowest>.245,`${spec.kind}: centre stays above the floor (lowest ${lowest})`);
      // The drop releases ~6 J; pumping (the old rolling defect) would keep it moving, a jitter would not.
      assert.ok(after<.02,`${spec.kind}: stays at rest after landing (max ${after} J)`);
      assert.ok(energy()<1e-2,`${spec.kind}: settles (energy ${energy()})`);
    } finally {world.destroy();}
  }
});

gpuTest('World SDF point capacity runs the paper widest cache sweep (16 cached + 4 samples)', async device=>{
  const world=await World.create({device,maxBodies:2});
  world.addBox({size:[20,.4,20],position:[0,-.2,0],fixed:true});
  // A tilted torus offers many separate contact patches against the floor.
  const body=world.addBox({size:[1,1,1],position:[0,.45,0],rotation:[.2,0,0,Math.sqrt(1-.04)],massProperties:{mass:1,moments:[.1,.2,.1]}})!;
  await world.enableSdfCollisions([{body,field:new AnalyticField({kind:'torus',majorRadius:.4,minorRadius:.15})}],
    {cacheSize:16,randomSamples:4},{maxPointsPerPair:20});
  try {
    let most=0;
    for(let frame=0;frame<120;frame++){world.step();if(frame%20===19){const [r]=await world.readSdfResults();most=Math.max(most,r?.points.length??0);}}
    await world.read();
    assert.ok(most>5,`results exceed the default 5-point capacity (${most})`);
    assert.ok(most<=20);
    assert.ok(body.position.every(Number.isFinite)&&body.position[1]>.1,`torus rests on the floor (${body.position[1]})`);
    await assert.rejects(async()=>{const w=await World.create({device,maxBodies:1});try{w.addBox({size:[1,1,1],position:[0,0,0]});await w.enableSdfCollisions([],{cacheSize:16,randomSamples:4});}finally{w.destroy();}},/at most 4 cached\/5 total/);
  } finally {world.destroy();}
});
