// Bounded non-convex/contact-friction validation: at most two bodies per scene.
import assert from 'node:assert/strict';
import { World, type ContactEvent } from '../src/lib/world.ts';
import { TorusGeometry } from 'three';
import { gpuTest } from './device.ts';
import { Solver } from '../src/avbd3d/ref/solver.ts';
import { Rigid } from '../src/avbd3d/ref/body.ts';
import { AnalyticField, type Field } from '../src/avbd3d/ref/sdf/shapes.ts';
import { bakeVoxelSdf } from '../src/avbd3d/ref/sdf/bake.ts';
import { SdfBody } from '../src/avbd3d/ref/sdf/body.ts';
import { SdfManifold, collisionParams } from '../src/avbd3d/ref/sdf/manifold.ts';
import { SdfCollisionAdapter } from '../src/avbd3d/ref/sdf/adapter.ts';
import { Random } from '../src/avbd3d/ref/sdf/math.ts';
import { counters } from '../src/avbd3d/ref/sdf/optimize.ts';
import { SdfNarrowphasePass } from '../src/avbd3d/gpu/sdf-pass.ts';
import { GpuSolver3D } from '../src/avbd3d/gpu/solver.ts';
import { SdfCollisionBridge } from '../src/avbd3d/gpu/sdf-bridge.ts';
import { BODY_FLOATS, B_POS, B_VEL } from '../src/avbd3d/gpu/layout.ts';

const hash=(input:number)=>{let x=input>>>0;x=Math.imul(x^(x>>>16),0x7feb352d);x=Math.imul(x^(x>>>15),0x846ca68b);return (x^(x>>>16))>>>0;};
class PairRandom extends Random {
  value:number;
  constructor(step:number) {super();this.value=hash(42^hash(step)^hash(0)^hash((1+0x9e3779b9)>>>0));}
  override next() {this.value=hash((this.value+0x9e3779b9)>>>0);return this.value;}
}

gpuTest('SDF analytic and baked torus preserve cavities and inner/outer contact CPU parity',async device=>{
  const geometry=new TorusGeometry(.65,.2,12,32);geometry.rotateX(Math.PI/2);
  const voxel=bakeVoxelSdf(geometry,{resolution:24}).field;geometry.dispose();
  const fields:Field[]=[new AnalyticField({kind:'torus',majorRadius:.65,minorRadius:.2}),voxel];
  const sphere=new AnalyticField({kind:'sphere',radius:.15});
  for(const field of fields) {
    const a=new SdfBody(field,[0,0,0],[0,0,0,1],1,true),b=new SdfBody(sphere);
    const params={...collisionParams(),cacheSize:1,randomSamples:0,alpha:0};
    const pass=await SdfNarrowphasePass.create(device,[field,sphere],{maxBodies:2,maxPairs:1});
    try {
      // Hole centre, inner clearance, inner wall, outer wall, exterior, and re-entry.
      const cases=[{x:0,active:false},{x:.24,active:false},{x:.42,active:true},{x:.94,active:true},{x:1.1,active:false},{x:.42,active:true}];
      for(let frame=0;frame<cases.length;frame++) {
        b.position=[cases[frame].x,0,0];
        const cpu=new SdfManifold(a,b);cpu.update(params,1/60,new PairRandom(frame),counters());
        pass.step([a,b],[{a:0,b:1,key:42}],1/60,params);
        const [result]=await pass.readResults(), active=result.points.filter(p=>p.active);
        assert.equal(active.length>0,cases[frame].active,`case ${frame}, ${field.constructor.name}`);
        assert.equal(active.length,cpu.points.filter(p=>p.active).length);
        for(const p of result.points) {
          const q=cpu.points.find(q=>Math.hypot(...p.x.map((v,k)=>v-q.x[k]))<2*params.tolerance);
          assert.ok(q,`CPU minimizer match case ${frame}`);
          assert.ok(Math.abs(p.separation-q.separation)<2*params.tolerance);
          assert.ok(Math.abs(Math.hypot(...p.normal)-1)<1e-4);
          if(p.active)assert.ok(p.normal.reduce((s,v,k)=>s+v*q.normal[k],0)>.98);
        }
        if(active.length)assert.ok(active[0].normal[0]*(cases[frame].x<.65?-1:1)>.9,'inner wall points into cavity; outer wall points out');
      }
    } finally {pass.destroy();}
  }
});

async function frictionScene(device:GPUDevice,mu:number,angle:number,sliding:boolean,regularized=false) {
  const ref=new Solver();ref.iterations=15;ref.alpha=.95;
  const sn=Math.sin(angle),cs=Math.cos(angle),s=Math.sin(angle/2),c=Math.cos(angle/2),h=Math.SQRT1_2;
  const floor=new Rigid(ref,[36,8,.5],0,mu,[-.25*sn,0,-.25*cs]);floor.positionAng.set([0,s,0,c]);
  const body=new Rigid(ref,[1.7,1.7,.4],1,mu,[.21*sn,0,.21*cs],sliding?[cs,0,-sn]:[0,0,0]);
  body.positionAng.set([c*h,s*h,-s*h,c*h]);
  const ground=new AnalyticField({kind:'box',halfExtents:[18,4,.25]});
  const ring=new AnalyticField({kind:'torus',majorRadius:.65,minorRadius:.2});
  ring.seeds=Array.from({length:8},(_,i)=>[.65*Math.cos(i*Math.PI/4),-.2,.65*Math.sin(i*Math.PI/4)]);
  const solver=new GpuSolver3D(device,ref,{spatialSort:false});solver.params.massPenalty=false;
  // Isolate Coulomb behavior: the reference distribution regularizer biases curved normals.
  const params=regularized?{}:{tolerance:.001,maxIterations:100,alpha:0};
  const bridge=await SdfCollisionBridge.create(solver,[{field:ground},{field:ring}],[{a:1,b:0,key:42}],params);
  const adapter=new SdfCollisionAdapter(ref,params);adapter.attach(floor,ground);adapter.attach(body,ring);
  let contactSamples=0,stickSamples=0;
  try {
    solver.setCollisionBackend(bridge);
    for(let frame=0;frame<180;frame++) {
      ref.step();solver.step();
      if(frame%30===29) {
        await device.queue.onSubmittedWorkDone();
        const contacts=await solver.readContactList();contactSamples+=contacts.length;stickSamples+=contacts.filter(c=>c.stick).length;
        for(const point of contacts) {
          assert.ok([...point.lam,...point.normal,...point.rA,...point.rB].every(Number.isFinite));
          assert.ok(point.lam[0]<=1e-5,'normal force is compressive');
          assert.ok(Math.hypot(point.lam[1],point.lam[2])<=mu*Math.abs(point.lam[0])+1e-4,'Coulomb cone');
        }
      }
    }
    const data=await solver.readBodies();assert.ok(data.every(Number.isFinite));
    assert.equal((await solver.readCounters()).overflow,0);assert.ok(contactSamples>0);
    const pos=Array.from(data.slice(BODY_FLOATS+B_POS,BODY_FLOATS+B_POS+3));
    const vel=Array.from(data.slice(BODY_FLOATS+B_VEL,BODY_FLOATS+B_VEL+3));
    const metrics=(p:ArrayLike<number>,v:ArrayLike<number>)=>({travel:p[0]*cs-p[2]*sn,height:p[0]*sn+p[2]*cs,speed:Math.hypot(v[0],v[1],v[2])});
    return {gpu:metrics(pos,vel),cpu:metrics(body.positionLin,body.velocityLin),stickSamples};
  } finally {adapter.dispose();solver.destroy();}
}

gpuTest('SDF non-convex ring sliding obeys friction cone and agrees with CPU behavior',async device=>{
  for(const mu of [0,.8]) {
    const result=await frictionScene(device,mu,0,true);
    console.log('ring sliding',mu,result);
    for(const m of [result.cpu,result.gpu]) {
      assert.ok(m.height>.15&&m.height<.28,`supported height ${m.height}`);
      if(mu===0) {assert.ok(m.travel>2.7&&m.travel<3.3);assert.ok(m.speed>.9&&m.speed<1.1);}
      else {assert.ok(Math.abs(m.travel-1/(2*mu*10))<.025,`stopping distance ${m.travel}`);assert.ok(m.speed<.15);}
    }
    assert.ok(Math.abs(result.gpu.travel-result.cpu.travel)<.25,'CPU/GPU travel envelope');
    if(mu>0)assert.ok(result.stickSamples>0,'static friction engages after stopping');
  }
});

gpuTest('SDF non-convex ring holds above the friction threshold and slides below it',async device=>{
  const angle=Math.PI/12;
  for(const mu of [0,.8]) {
    const result=await frictionScene(device,mu,angle,false,true);
    console.log('ring incline',mu,result);
    for(const m of [result.cpu,result.gpu]) {
      assert.ok(m.height>.14&&m.height<.3,`supported incline height ${m.height}`);
      if(mu===0)assert.ok(m.travel>8,'frictionless body accelerates downhill');
      else {assert.ok(Math.abs(m.travel)<.25,'static friction holds');assert.ok(m.speed<.15);}
    }
    assert.ok(Math.abs(result.gpu.travel-result.cpu.travel)<.5,'CPU/GPU incline travel envelope');
  }
});


gpuTest('SDF crossed non-convex rings keep multiple independent normals through temporal updates',async device=>{
  const ring=new AnalyticField({kind:'torus',majorRadius:.65,minorRadius:.2});
  ring.seeds=Array.from({length:16},(_,i)=>[.85*Math.cos(i*Math.PI/8),0,.85*Math.sin(i*Math.PI/8)]);
  const a=new SdfBody(ring,[0,0,0],[0,0,0,1],1,true);
  const b=new SdfBody(ring,[.05,0,0],[Math.SQRT1_2,0,0,Math.SQRT1_2]);
  const cpu=new SdfManifold(a,b),params=collisionParams();
  const pass=await SdfNarrowphasePass.create(device,[ring],{maxBodies:2,maxPairs:1});
  let independent=0,cached=0;
  try {
    for(let frame=0;frame<16;frame++) {
      b.position=[.05,.02*frame,0];b.velocity=[0,1.2,0];
      cpu.update(params,1/60,new PairRandom(frame),counters());
      pass.step([a,b],[{a:0,b:1,key:42,...(frame?{previous:0}:{})}],1/60,params);
      const [result]=await pass.readResults();const actual=result.points.filter(p=>p.active),expected=cpu.points.filter(p=>p.active);
      assert.ok(actual.length>=2&&expected.length>=2,'both paths retain multiple contact regions');
      assert.ok(Math.abs(Math.min(...actual.map(p=>p.separation))-Math.min(...expected.map(p=>p.separation)))<.03,'deepest separation parity');
      for(const p of actual) {
        assert.ok(expected.some(q=>Math.hypot(...p.x.map((v,k)=>v-q.x[k]))<2*params.tolerance),'CPU contact-location parity');
        assert.ok([...p.x,...p.normal,...p.anchorA,...p.anchorB,p.separation].every(Number.isFinite));
        assert.ok(Math.abs(Math.hypot(...p.normal)-1)<1e-4);
        cached+=+p.cached;
        if(actual.some(q=>q.normal.reduce((s,v,i)=>s+v*p.normal[i],0)<.8))independent++;
      }
    }
    assert.ok(cached>0);assert.ok(independent>0,'non-convex contacts do not collapse to a pair normal');
  } finally {pass.destroy();}
});

gpuTest('World analytic and voxel cavities permit free fall through the hole but report rim impact',async device=>{
  const geometry=new TorusGeometry(.65,.2,12,32);geometry.rotateX(Math.PI/2);
  const voxel=bakeVoxelSdf(geometry,{resolution:24}).field;geometry.dispose();
  for(const field of [new AnalyticField({kind:'torus',majorRadius:.65,minorRadius:.2}),voxel]) {
    const world=await World.create({device,maxBodies:3,gravity:[0,-10,0]});
    const ring=world.addBox({size:[1.7,.4,1.7],fixed:true})!;
    const through=world.addSphere({radius:.15,position:[0,1,0],reportContacts:true})!;
    const rim=world.addSphere({radius:.15,position:[.65,1,0],reportContacts:true})!;
    const events:ContactEvent[]=[];world.onContact(e=>events.push(e));
    try {
      await world.enableSdfCollisions([{body:ring,field}]);
      for(let i=0;i<90;i++) {world.step();if(i%15===14)await device.queue.onSubmittedWorkDone();}
      await world.read();
      assert.ok(through.position[1]<-9,'centre sphere falls through the hole');
      assert.ok(!events.some(e=>e.a===through||e.b===through),'no false cavity collision');
      assert.ok(events.some(e=>e.type==='begin'&&(e.a===rim||e.b===rim)&&e.impulse>0),'rim makes a physical impact');
      assert.ok([...through.position,...rim.position].every(Number.isFinite));
      assert.equal((await world.solver.readCounters()).overflow,0);
    } finally {world.destroy();}
  }
});
