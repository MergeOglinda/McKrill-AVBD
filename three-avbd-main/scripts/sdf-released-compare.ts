// Bounded, sequential comparison. No baseline regeneration and no timing claims.
import { readFile, writeFile } from 'node:fs/promises';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, releasedSettings, releasedScenes, type ReleasedProfile, type ReadAsset } from '../src/sdf-released/scenes.ts';
import { measuredFrame } from '../src/sdf-released/metrics.ts';
import { jitterReleased } from './sdf-jitter.ts';
const frames=Number(process.argv[2]??360),output=process.argv[3];
if(!Number.isInteger(frames)||frames<1||frames>360||!output)throw new Error('Usage: sdf-released-compare.ts [1..360 frames] output.json [mesh-quality|reference] [scene IDs, e.g. 6,9] [probe interval, default 30] [mesh-plane] [jitter=seed]');
const profile=(process.argv[4]??'mesh-quality') as ReleasedProfile;
if(!['reference','mesh-quality'].includes(profile))throw new Error('Unknown comparison profile');
const probeEvery=Number(process.argv[6]??30);
if(!Number.isInteger(probeEvery)||probeEvery<1||probeEvery>360)throw new Error('Probe interval must be 1..360 frames');
const options=process.argv.slice(7),meshVertexContacts=options.includes('mesh-plane');
const jitter=Number(options.find(o=>o.startsWith('jitter='))?.slice(7)??0);
if(options.some(o=>o!=='mesh-plane'&&!o.startsWith('jitter='))||!Number.isInteger(jitter)||jitter<0)throw new Error('Unknown option: '+options.join(' '));
const sceneIds=process.argv[5]?.split(',').map(Number);
if(sceneIds?.some(id=>!releasedScenes.some(s=>s.id===id)))throw new Error('Unknown scene ID');
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);
const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter');
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
const errors:string[]=[]; device.onuncapturederror=e=>errors.push(e.error.message);
const report={meshVertexContacts,jitter,reference:'sdf-dcd-demo 1dcdf1e7fa07d16fda15826ad8c2e6697075a76e',frames,profile,probeEvery,solver:'GPU AVBD; source CPU sequential impulses',
  conventions:'original poses, COM, mass/full inertia, resolution-100 grids/seeds; explicit profile substeps and 20 AVBD iterations; last-substep contacts/pairs/depth, summed substep queries/iterations; post-integration energy; vertex-sampled penetration at probeEvery frames (expanded source triangle corners, duplicate probes differ from source mesh indexing)',
  scenes:[] as unknown[]};
try {
  for(const entry of releasedScenes.filter(s=>!sceneIds||sceneIds.includes(s.id))){
    const source=await loadReleasedScene(entry.id,read);
    // Diagnostic opt-in only; released scenes do not enable mesh-plane contacts by default.
    const configured=meshVertexContacts?{...source,capture:{...source.capture,params:{...source.capture.params,meshVertexContacts:true}}}:source;
    const scene=jitterReleased(configured,jitter);
    const {world,bodies}=await instantiateReleased(scene,{device},profile);
    try{
      const samples=[];
      for(let frame=1;frame<=frames;frame++){
        samples.push(await measuredFrame(scene,world,bodies,frame,frame%probeEvery===0||frame===frames));
        // Release queue between frames and yield CPU; never batch thousands of steps.
        await device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,5));
      }
      const source=scene.capture.samples.slice(0,frames),last=samples.at(-1)!,baseline=source.at(-1)!;
      console.log(entry.name,JSON.stringify({gpu:last,reference:baseline}));
      report.scenes.push({name:entry.name,bodies:bodies.length,sourceSolver:scene.capture.solver,collisionParams:releasedSettings(scene,profile).collision,
        avbd:{substeps:Math.round(scene.capture.solver.dt/world.dt),iterations:world.solver.params.iterations,dt:world.dt,alpha:world.solver.params.alpha,gamma:world.solver.params.gamma,massPenalty:world.solver.params.massPenalty},
        samples,referenceSamples:source});
      await writeFile(output,JSON.stringify(report,null,2)+'\n');
    } finally {world.destroy();scene.dispose();}
  }
  if(errors.length)throw new Error(errors.join('\n'));
} finally {device.destroy();}
