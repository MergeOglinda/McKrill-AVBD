import assert from 'node:assert/strict';
import { gpuTest } from './device.ts';
import { Solver } from '../src/avbd3d/ref/solver.ts';
import { Rigid } from '../src/avbd3d/ref/body.ts';
import { AnalyticField } from '../src/avbd3d/ref/sdf/shapes.ts';
import { GpuSolver3D } from '../src/avbd3d/gpu/solver.ts';
import { SdfCollisionBridge } from '../src/avbd3d/gpu/sdf-bridge.ts';
import { CONTACT_WORDS, K_LAM, K_PEN, K_RA, K_RB, STICK_BIT } from '../src/avbd3d/gpu/layout.ts';

gpuTest('SDF production bridge preserves independent normals and point-identity warm starts', async device=>{
  const ref=new Solver(); ref.gravity=0; ref.iterations=0;
  new Rigid(ref,[1,1,1],1,.6,[.3,0,0]);
  new Rigid(ref,[1,1,1],1,.6,[0,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  solver.params.massPenalty=false; // Isolate retained penalty decay from the optional mass floor.
  const sphere=new AnalyticField({kind:'sphere',radius:.4});
  const ring=new AnalyticField({kind:'torus',majorRadius:.4,minorRadius:.2});
  ring.seeds=[[.6,0,0],[-.6,0,0],[0,0,.6],[0,0,-.6]];
  const bridge=await SdfCollisionBridge.create(solver,[{field:sphere},{field:ring}],[{a:1,b:0,key:42}]);
  try {
    solver.setCollisionBackend(bridge); solver.step();
    const first=await solver.readContactList(), [initial]=await bridge.readResults();
    assert.ok(first.length>1,'multiple active points');
    assert.ok(first.some(p=>first.some(q=>p.normal.reduce((s,v,i)=>s+v*q.normal[i],0)<.99)),'independent normals differ');
    first.forEach(c=>{
      const p=initial.points.find(p=>p.id===c.feature)!; assert.ok(p?.active);
      assert.ok(Math.hypot(...c.normal.map((v,i)=>v+p.normal[i]))<1e-5);
    });
    // Inject distinct solved state per point. Rank changes must not swap these forces.
    const storage=solver.contactStorage;
    for(let i=0;i<first.length;i++) {
      device.queue.writeBuffer(storage.contacts,i*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-(i+1),.1*(i+1),-.2*(i+1)]));
      device.queue.writeBuffer(storage.contacts,i*CONTACT_WORDS*4+K_PEN*4,new Float32Array([10+i,20+i,30+i]));
    }
    solver.step();
    const second=await solver.readContactList(), [result]=await bridge.readResults();
    let matched=0;
    for(const c of second) {
      const p=result.points.find(p=>p.id===c.feature)!;
      const oldIndex=first.findIndex(q=>q.feature===c.feature);
      if(p.cached&&oldIndex>=0) {
        matched++;
        const previous=first[oldIndex], gain=solver.params.alpha*solver.params.gamma;
        const basis=(n:number[])=>{
          const t=Math.abs(n[0])>Math.abs(n[2])?[-n[1],n[0],0]:[0,-n[2],n[1]];
          const l=Math.hypot(...t); const u=t.map(v=>v/l);
          return [n,u,[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]]];
        };
        const oldBasis=basis(previous.normal), newBasis=basis(c.normal);
        const lambda=[-(oldIndex+1),.1*(oldIndex+1),-.2*(oldIndex+1)];
        const world=[0,1,2].map(j=>oldBasis.reduce((sum,row,k)=>sum+row[j]*lambda[k],0));
        const expected=newBasis.map(row=>gain*row.reduce((sum,v,j)=>sum+v*world[j],0));
        c.lam.forEach((v,j)=>assert.ok(Math.abs(v-expected[j])<1e-4,`force ${c.feature}, axis ${j}`));
        assert.ok(Math.abs(c.pen[0]-(10+oldIndex)*solver.params.gamma)<1e-4,`pen ${c.pen[0]} expected ${(10+oldIndex)*solver.params.gamma}`);
      }
    }
    assert.ok(matched>0,'persistent points warmed');
    bridge.setPairs([{a:1,b:0,key:43}]);solver.step();
    const reset=await solver.readContactList();assert.ok(reset.length>0);
    reset.forEach(c=>assert.deepEqual(c.lam,[0,0,0]));
    bridge.setPairs([]);solver.step();assert.equal((await solver.readContactList()).length,0);
    solver.setCollisionBackend(null);solver.step();
    assert.equal((await solver.readCounters()).overflow,0);
    assert.ok((await solver.readContactList()).length>0,'standard collisions resume after detach');
  } finally {solver.destroy();}
});

gpuTest('SDF production contacts support a falling body with GPU-owned poses', async device=>{
  const ref=new Solver(); ref.iterations=10;
  new Rigid(ref,[8,8,.4],0,.6,[0,0,-.2]);
  new Rigid(ref,[.8,.8,.8],1,.6,[0,0,.8]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const floor=new AnalyticField({kind:'box',halfExtents:[4,4,.2]});
  const sphere=new AnalyticField({kind:'sphere',radius:.4});
  const bridge=await SdfCollisionBridge.create(solver,[{field:floor},{field:sphere}],[{a:1,b:0,key:1}],{cacheSize:1,randomSamples:0,alpha:0});
  try {
    solver.setCollisionBackend(bridge);
    for(let i=0;i<120;i++) {solver.step();if(i%20===19)await device.queue.onSubmittedWorkDone();}
    const bodies=await solver.readBodies(), z=bodies[40+2];
    assert.ok(bodies.every(Number.isFinite));assert.ok(z>.35&&z<.47,`settled sphere center ${z}`);
    const contacts=await solver.readContactList();assert.ok(contacts.length>0);
    assert.ok(contacts.some(c=>Math.abs(c.lam[0])>0),'solver generated support force');
    assert.equal((await solver.readCounters()).overflow,0);
  } finally {solver.destroy();}
});

gpuTest('SDF production warm starts survive pair reorder and reject a changed pair identity', async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  new Rigid(ref,[8,8,.4],0,.6,[0,0,-.2]);
  new Rigid(ref,[.8,.8,.8],1,.6,[-1,0,.35]);
  new Rigid(ref,[.8,.8,.8],1,.6,[1,0,.35]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});solver.params.massPenalty=false;
  const floor=new AnalyticField({kind:'box',halfExtents:[4,4,.2]});
  const sphere=new AnalyticField({kind:'sphere',radius:.4});
  const bridge=await SdfCollisionBridge.create(solver,[{field:floor},{field:sphere},{field:sphere}],
    [{a:1,b:0,key:10},{a:2,b:0,key:20}],{cacheSize:1,randomSamples:0,alpha:0});
  try {
    assert.throws(()=>bridge.setPairs([{a:1,b:0,key:10},{a:1,b:0,key:20}]),/unique body pairs/);
    assert.throws(()=>bridge.setPairs([{a:0,b:1,key:10}]),/a > b/);
    solver.setCollisionBackend(bridge);solver.step();
    const first=await solver.readContactList();assert.equal(first.length,2);
    // Map readback order to actual contact indices via each manifold, not atomic ordering.
    const read=async(buffer:GPUBuffer)=>{
      const staging=device.createBuffer({size:64,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      try {const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,0,staging,0,64);device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);return new Uint32Array(staging.getMappedRange().slice(0));} finally {staging.destroy();}
    };
    const storage=solver.contactStorage, words=await read(storage.manifolds);
    for(let i=0;i<2;i++) device.queue.writeBuffer(storage.contacts,words[i*8+2]*CONTACT_WORDS*4+K_LAM*4,new Float32Array([-words[i*8]*3,0,0]));
    bridge.setPairs([{a:2,b:0,key:20},{a:1,b:0,key:10}]);solver.step();
    const reordered=await solver.readContactList();assert.equal(reordered.length,2);
    reordered.forEach(c=>{
      const old=first.find(q=>q.a===c.a)!;
      const projection=old.normal.reduce((s,v,i)=>s+v*c.normal[i],0);
      assert.ok(Math.abs(c.lam[0]+c.a*3*solver.params.alpha*solver.params.gamma*projection)<1e-4,`pair ${c.a}: ${c.lam[0]}, projection ${projection}`);
    });
    bridge.setPairs([{a:2,b:0,key:21},{a:1,b:0,key:10}]);solver.step();
    const changed=await solver.readContactList();
    assert.deepEqual(changed.find(c=>c.a===2)!.lam,[0,0,0]);
    assert.ok(changed.find(c=>c.a===1)!.lam[0]<0);
  } finally {solver.destroy();}
});

gpuTest('SDF bridge retains sticking anchors on nonspherical surfaces and validates mutable parameters', async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  new Rigid(ref,[8,8,.4],0,.6,[0,0,-.2]);
  new Rigid(ref,[.8,.8,.8],1,.6,[0,0,.35]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const floor=new AnalyticField({kind:'box',halfExtents:[4,4,.2]});
  const box=new AnalyticField({kind:'box',halfExtents:[.4,.4,.4]});
  const bridge=await SdfCollisionBridge.create(solver,[{field:floor},{field:box}],[{a:1,b:0,key:1}],{cacheSize:1,randomSamples:0,alpha:0});
  try {
    solver.setCollisionBackend(bridge);solver.step();
    const [initial]=await solver.readContactList();assert.ok(initial);
    const anchor=initial.rA.map((v,i)=>i===2?Math.fround(v+.01):v);
    const words=new Uint32Array(4), floats=new Float32Array(words.buffer);
    floats.set(anchor);words[3]=initial.feature|STICK_BIT;
    device.queue.writeBuffer(solver.contactStorage.contacts,K_RA*4,words);
    solver.step();const [next]=await solver.readContactList();
    assert.equal(next.feature,initial.feature);assert.equal(next.stick,true);
    assert.deepEqual(next.rA,anchor);assert.deepEqual(next.rB,initial.rB);
    const read=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      const encoder=device.createCommandEncoder();
      encoder.copyBufferToBuffer(solver.contactStorage.contacts,(K_RB+3)*4,read,0,4);
      device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const normalError=new Float32Array(read.getMappedRange())[0];
      const delta=next.rA.map((v,i)=>v-next.rB[i]+(i===2?.55:0));
      const expected=next.normal.reduce((sum,v,i)=>sum+v*delta[i],0)-bridge.params.margin+bridge.params.allowedPenetration;
      assert.ok(Math.abs(normalError-expected)<1e-5,'normal error uses the retained anchors, not the relocated search point');
    } finally {read.destroy();}
    bridge.params.allowedPenetration=NaN;
    assert.throws(()=>solver.step(),/allowed penetration/);
    bridge.params.allowedPenetration=.01;
    solver.step();assert.equal((await solver.readCounters()).overflow,0);
  } finally {solver.destroy();}
});

gpuTest('SDF bridge setup rejects teardown during asynchronous shader compilation', async device=>{
  const ref=new Solver();new Rigid(ref,[1,1,1],1,.6,[0,0,0]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const field=new AnalyticField({kind:'box',halfExtents:[.5,.5,.5]});
  const creating=SdfCollisionBridge.createAutomatic(solver,[{field}]);
  solver.destroy();
  await assert.rejects(()=>creating,/destroyed during setup/);
  await assert.rejects(()=>SdfCollisionBridge.createAutomatic(solver,[{field}]),/destroyed/);
});

gpuTest('SDF earlier activation adds separated contacts without changing the resting gap',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  new Rigid(ref,[8,8,.4],0,.6,[0,0,-.2]);
  new Rigid(ref,[.8,.8,.8],1,.6,[0,0,.43]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const bridge=await SdfCollisionBridge.create(solver,[
    {field:new AnalyticField({kind:'box',halfExtents:[4,4,.2]})},
    {field:new AnalyticField({kind:'box',halfExtents:[.4,.4,.4]})},
  ],[{a:1,b:0,key:1}],{cacheSize:1,randomSamples:0,alpha:0,tolerance:.001});
  try{
    solver.setCollisionBackend(bridge);solver.step();
    assert.equal((await solver.readContactList()).length,0,'3 cm gap is outside released activation');
    bridge.params.margin=.04;bridge.params.allowedPenetration=.03;
    solver.step();assert.ok((await solver.readContactList()).length>0,'earlier activation covers the gap');
    const read=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(solver.contactStorage.contacts,(K_RB+3)*4,read,0,4);
      device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      assert.ok(Math.abs(new Float32Array(read.getMappedRange())[0]-.02)<.001,'3 cm separation minus unchanged 1 cm resting gap');
    }finally{read.destroy();}
  }finally{solver.destroy();}
});

gpuTest('SDF relocated deeper contacts refresh anchors while retaining identity and normal warm force',async device=>{
  const ref=new Solver();ref.gravity=0;ref.iterations=0;
  new Rigid(ref,[8,8,.4],0,.6,[0,0,-.2]);
  new Rigid(ref,[.8,.8,.8],1,.6,[0,0,.35]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});
  const bridge=await SdfCollisionBridge.create(solver,[
    {field:new AnalyticField({kind:'box',halfExtents:[4,4,.2]})},
    {field:new AnalyticField({kind:'box',halfExtents:[.4,.4,.4]})},
  ],[{a:1,b:0,key:1}],{cacheSize:1,randomSamples:0,alpha:0,refreshDeeperContacts:true});
  try{
    solver.setCollisionBackend(bridge);solver.step();
    const [initial]=await solver.readContactList();assert.ok(initial);
    const words=new Uint32Array(4),floats=new Float32Array(words.buffer);
    floats.set(initial.rA.map((v,i)=>i===2?v+.08:v));words[3]=initial.feature|STICK_BIT;
    device.queue.writeBuffer(solver.contactStorage.contacts,K_RA*4,words);
    device.queue.writeBuffer(solver.contactStorage.contacts,K_LAM*4,new Float32Array([-2,0,0]));
    solver.step();const [next]=await solver.readContactList();
    const [result]=await bridge.readResults();const point=result.points.find(p=>p.id===initial.feature)!;
    assert.equal(next.feature,initial.feature);assert.equal(next.stick,false);
    assert.deepEqual(next.rA,point.anchorA);assert.deepEqual(next.rB,point.anchorB);
    const projection=initial.normal.reduce((sum,v,i)=>sum+v*next.normal[i],0);
    assert.ok(Math.abs(next.lam[0]+2*solver.params.alpha*solver.params.gamma*projection)<1e-5);
  }finally{solver.destroy();}
});
