// Bounded phase attribution for STE-3016; no renderer, not a browser FPS benchmark.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, type ReadAsset, type ReleasedProfile } from '../src/sdf-released/scenes.ts';
import { collisionDispatchProfiler } from './sdf-dispatch-profile.ts';
import type { StepProfile } from '../src/avbd3d/gpu/solver.ts';
const output=process.argv[2], profile=(process.argv[3]??'mesh-quality') as ReleasedProfile;
const frames=Number(process.argv[4]??180),dispatch=process.argv[5]==='dispatch';
if(!output||!['mesh-quality','reference'].includes(profile)||!Number.isInteger(frames)||frames<30||frames>360)throw Error('Usage: sdf-bunny-profile.ts output [mesh-quality|reference] [30..360 frames] [dispatch]');
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw Error('No GPU');
if(!adapter.features.has('timestamp-query'))throw Error('Diagnostic requires timestamp-query');
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
const scene=await loadReleasedScene(9,read),{world}=await instantiateReleased(scene,{device},profile);
const profiler=dispatch?collisionDispatchProfiler(device):null;
const report={profile,frames,dispatch,adapter:adapter.info,dt:world.dt,substeps:world.maxSubsteps,iterations:world.solver.params.iterations,scope:'Sequential headless steps with queue drain and 5ms idle each frame. Phase timestamps split passes; dispatch mode additionally splits collision dispatches. Not browser throughput or unsplit kernel timings.',samples:[] as unknown[]};
try{
 for(let frame=1;frame<=frames;frame++){
  const phases:Record<string,number>={};
  for(let sub=0;sub<world.maxSubsteps;sub++){
   let pending:Promise<StepProfile>|null=null;
   if(frame%30===0){if(profiler)profiler.arm();else pending=new Promise(resolve=>world.solver.profileNextStep(resolve));}
   world.step();await device.queue.onSubmittedWorkDone();
   if(pending||frame%30===0&&profiler){const p=profiler?await profiler.collect():await pending!;for(const [key,value]of Object.entries(p))phases[key]=(phases[key]??0)+value;}
  }
  const results=await world.readSdfResults();await world.read();const counters=await world.solver.readCounters();
  if(counters.overflow||errors.length)throw Error(`Invalid run ${JSON.stringify({counters,errors})}`);
  const poses=world.bodies.map(b=>({position:b.position,rotation:b.rotation,velocity:b.velocity,angularVelocity:b.angularVelocity}));
  const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
  report.samples.push({frame,phases,poses,poseHash:hash(poses),contactHash:hash(results),queries:results.reduce((s,r)=>s+r.queries,0),pairs:results.length,counters});
  await new Promise(resolve=>setTimeout(resolve,5));
 }
 await writeFile(output,JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report.samples.filter((s:any)=>Object.keys(s.phases).length).map((s:any)=>({frame:s.frame,phases:s.phases,pairs:s.pairs,queries:s.queries}))));
}finally{profiler?.destroy();world.destroy();scene.dispose();device.destroy();}
