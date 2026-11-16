import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { Solver } from '../src/avbd3d/ref/solver.ts';
import { Rigid } from '../src/avbd3d/ref/body.ts';
import { AnalyticField } from '../src/avbd3d/ref/sdf/shapes.ts';
import { SdfBody } from '../src/avbd3d/ref/sdf/body.ts';
import { GpuSolver3D } from '../src/avbd3d/gpu/solver.ts';
import { SdfCollisionBridge } from '../src/avbd3d/gpu/sdf-bridge.ts';
import { sdfBroadphaseOptions } from '../src/avbd3d/gpu/sdf-broadphase.ts';
import { BODY_FLOATS, B_POS, CONTACT_WORDS, K_LAM } from '../src/avbd3d/gpu/layout.ts';

const sphere=new AnalyticField({kind:'sphere',radius:.4});
gpuTest('SDF spatial candidates match brute force with rotated/scaled bounds and oversized bodies',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  const bodies:SdfBody[]=[];
  for(let i=0;i<96;i++) {
    const field=i%13===0?new AnalyticField({kind:'box',halfExtents:[2,.2,.3]}):sphere;
    const p:[number,number,number]=[(i%8)*.75-3,Math.floor(i/8)%4*.65-1,Math.floor(i/32)*.7-1];
    const scale=.6+(i%5)*.2,angle=i*.17;
    const q:[number,number,number,number]=[0,Math.sin(angle/2),0,Math.cos(angle/2)];
    const fixed=i%7===0;
    const rigid=new Rigid(ref,[.1,.1,.1],fixed?0:1,.3,p);
    rigid.positionAng.set(q);
    bodies.push(new SdfBody(field,p,q,scale,fixed));
  }
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const bridge=await SdfCollisionBridge.createAutomatic(solver,bodies.map(b=>({field:b.field,scale:b.scale})),
    {margin:.02,cacheSize:1,randomSamples:0},{cellSize:1.2,maxPairs:2048});
  try {
    solver.setCollisionBackend(bridge);solver.step();
    const expected:string[]=[];
    for(let a=1;a<bodies.length;a++)for(let b=0;b<a;b++) {
      if(bodies[a].isStatic&&bodies[b].isStatic)continue;
      const aa=bodies[a].bounds,bb=bodies[b].bounds;
      if(aa.min.every((v,k)=>v<=bb.max[k]+.02)&&bb.min.every((v,k)=>v<=aa.max[k]+.02))expected.push(a+':'+b);
    }
    const results=await bridge.readResults(),actual=results.map(p=>p.a+':'+p.b);
    assert.deepEqual(actual.sort(),expected.sort());
    assert.equal(new Set(actual).size,actual.length);
    assert.equal((await solver.readCounters()).overflow,0);
    assert.ok((await bridge.readBroadphaseStats()).largeBodies>0);
  } finally {solver.destroy();}
});

gpuTest('SDF spatial sparse scene uses linear capacity and avoids all-pairs tests',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  for(let i=0;i<256;i++)new Rigid(ref,[.8,.8,.8],1,.3,[Math.floor(i/2)*4+(i%2)*.6,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const bridge=await SdfCollisionBridge.createAutomatic(solver,Array.from({length:256},()=>({field:sphere})),{cacheSize:1,randomSamples:0});
  try {
    solver.setCollisionBackend(bridge);solver.step();
    const stats=await bridge.readBroadphaseStats();
    assert.equal(stats.pairs,128);assert.equal((await bridge.readResults()).length,128);
    assert.equal(stats.capacity,256*8);assert.ok(stats.testedPairs<256*255/20);
    assert.equal((await solver.readCounters()).overflow,0);
  } finally {solver.destroy();}
});

gpuTest('SDF oversized list supports more than 64 bodies without losing pairs',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  for(let i=0;i<67;i++)new Rigid(ref,[1,1,1],i===66?1:0,.3,[0,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const bridge=await SdfCollisionBridge.createAutomatic(solver,Array.from({length:67},()=>({field:sphere})),
    {cacheSize:1,randomSamples:0},{cellSize:.1});
  try {
    solver.setCollisionBackend(bridge);solver.step();
    assert.equal((await bridge.readBroadphaseStats()).largeBodies,67);
    assert.equal((await bridge.readResults()).length,66);
    assert.equal((await solver.readCounters()).overflow,0);
  } finally {solver.destroy();}
});

gpuTest('SDF spatial overflow is explicit and increasing capacity recovers every pair',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  for(let i=0;i<4;i++)new Rigid(ref,[1,1,1],1,.3,[i*.1,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  try {
    for(const capacity of [1,6]) {
      const bridge=await SdfCollisionBridge.createAutomatic(solver,Array.from({length:4},()=>({field:sphere})),
        {cacheSize:1,randomSamples:0},{maxPairs:capacity});
      solver.setCollisionBackend(bridge);solver.step();
      const stats=await bridge.readBroadphaseStats(),counters=await solver.readCounters();
      assert.equal(stats.pairs,6);assert.equal(counters.pairs,6);
      assert.equal(counters.overflow&1,capacity===1?1:0);
      assert.equal((await bridge.readResults()).length,capacity);
      solver.setCollisionBackend(null);
    }
    for(const options of [{cellSize:0},{cellSize:NaN},{maxPairs:0},{maxPairs:1.5},{maxPairs:1e12}])
      assert.throws(()=>sdfBroadphaseOptions(device,[{field:sphere}],options),/Invalid/);
  } finally {solver.destroy();}
});

gpuTest('SDF spatial pair churn retains warm forces by body identity and clears stale history',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  for(const x of [0,.6,4,4.6])new Rigid(ref,[.8,.8,.8],1,.3,[x,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});solver.params.massPenalty=false;
  const bridge=await SdfCollisionBridge.createAutomatic(solver,Array.from({length:4},()=>({field:sphere})),{cacheSize:1,randomSamples:0,alpha:0});
  try {
    solver.setCollisionBackend(bridge);solver.step();
    const first=await solver.readContactList();assert.equal(first.length,2);
    first.forEach((_,i)=>device.queue.writeBuffer(solver.contactStorage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-10-i,0,0])));
    // Remove the first pair; the second changes compact slot or stays valid in place.
    device.queue.writeBuffer(solver.bodyBuffer,BODY_FLOATS*4+B_POS*4,new Float32Array([20,0,0]));
    solver.step();
    const next=await solver.readContactList();assert.equal(next.length,1);
    const priorIndex=first.findIndex(p=>p.a===next[0].a&&p.b===next[0].b);
    assert.ok(priorIndex>=0);
    assert.ok(Math.abs(next[0].lam[0]-(-10-priorIndex)*solver.params.alpha*solver.params.gamma)<1e-4);
    device.queue.writeBuffer(solver.bodyBuffer,BODY_FLOATS*4+B_POS*4,new Float32Array([.6,0,0]));
    solver.step();
    const reentered=await solver.readContactList();assert.equal(reentered.length,2);
    const fresh=reentered.find(p=>p.a===1&&p.b===0)!;
    assert.ok(fresh);assert.deepEqual(fresh.lam,[0,0,0]);
    assert.equal((await solver.readCounters()).overflow,0);
  } finally {solver.destroy();}
});
