// Bounded per-pair history: detector separation, actual mesh-vertex depth and solved forces.
// Optional overrides are diagnostic only; production settings are unchanged.
import { readFile, writeFile } from 'node:fs/promises';
import * as THREE from 'three';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, releasedScenes, type ReadAsset } from '../src/sdf-released/scenes.ts';
import { SdfBody } from '../src/avbd3d/ref/sdf/body.ts';
import type { Quat, Vec3 } from '../src/lib/world.ts';
import { jitterReleased } from './sdf-jitter.ts';
const [sceneId,bodyA,bodyB,frames,every]=process.argv.slice(2,7).map(Number),output=process.argv[7],flags=process.argv.slice(8);
const meshVertexContacts=flags.includes('mesh-plane'),alphaFlag=flags.find(f=>f.startsWith('alpha='));
const jitter=Number(flags.find(f=>f.startsWith('jitter='))?.slice(7)??0);
const alpha=alphaFlag?Number(alphaFlag.slice(6)):undefined;
if(!releasedScenes.some(s=>s.id===sceneId)||![bodyA,bodyB].every(i=>Number.isInteger(i)&&i>=0&&i<=48)||bodyA===bodyB||!Number.isInteger(frames)||frames<1||frames>360||!Number.isInteger(every)||every<1||!output
  ||flags.some(f=>f!=='mesh-plane'&&!f.startsWith('alpha=')&&!f.startsWith('jitter='))||(alpha!==undefined&&!(alpha>0&&alpha<=1))||!Number.isInteger(jitter)||jitter<0)
  throw new Error('Usage: sdf-pair-trace.ts scene(0..10) a b frames(1..360) every output.json [mesh-plane] [alpha=0..1] [jitter=seed]');
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);
const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No GPU adapter');
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
(globalThis as typeof globalThis & {__sdfPairGpu?:unknown}).__sdfPairGpu=[gpu,adapter,device];
const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
const source=await loadReleasedScene(sceneId,read);
const scene=jitterReleased(meshVertexContacts?{...source,capture:{...source.capture,params:{...source.capture.params,meshVertexContacts:true}}}:source,jitter);
const {world,bodies}=await instantiateReleased(scene,{device});
if(alpha!==undefined)world.solver.params.alpha=alpha;
/** Deepest vertex of `from` inside `into`'s field (positive = penetration depth). */
const vertexDepth=(from:number,into:number)=>{
  const b=bodies[into],field=new SdfBody(scene.bodies[into].field,b.position,
    new THREE.Quaternion(...b.rotation).multiply(new THREE.Quaternion(...scene.bodies[into].fieldRotation)).toArray() as Quat,scene.bodies[into].original.scale,b.fixed);
  const positions=scene.bodies[from].geometry.getAttribute('position');
  const rotation=new THREE.Quaternion(...bodies[from].rotation),origin=new THREE.Vector3(...bodies[from].position);
  let depth=0,point:Vec3|null=null;
  for(let j=0;j<positions.count;j++){
    const v=new THREE.Vector3().fromBufferAttribute(positions,j).applyQuaternion(rotation).add(origin);
    const phi=field.sample(v.toArray() as Vec3).phi;
    if(-phi>depth){depth=-phi;point=v.toArray() as Vec3;}
  }
  return {depth,point};
};
const samples=[];
try{
  for(let frame=1;frame<=frames;frame++){
    let results:Awaited<ReturnType<typeof world.readSdfResults>>=[];
    for(let s=0;s<Math.round(scene.capture.solver.dt/world.dt);s++){world.step();results=await world.readSdfResults();}
    if(frame%every===0||frame===frames){
      await world.read();
      const pair=results.find(r=>(r.a===bodyA&&r.b===bodyB)||(r.a===bodyB&&r.b===bodyA));
      const contacts=(await world.solver.readContactList()).filter(c=>(c.a===bodyA&&c.b===bodyB)||(c.a===bodyB&&c.b===bodyA));
      samples.push({frame,aInB:vertexDepth(bodyA,bodyB),bIntoA:vertexDepth(bodyB,bodyA),
        points:pair?.points.map(p=>({id:p.id,active:p.active,cached:p.cached,separation:p.separation,normal:p.normal,x:p.x}))??[],
        contacts:contacts.map(c=>({a:c.a,b:c.b,id:c.feature,stick:c.stick,normal:c.normal,force:c.lam})),
        velocityA:bodies[bodyA].velocity,velocityB:bodies[bodyB].velocity,
        poses:[bodyA,bodyB].map(i=>({position:bodies[i].position,rotation:bodies[i].rotation}))});
    }
    await device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,5));
  }
  if(errors.length)throw new Error(errors.join('\n'));
  await writeFile(output,JSON.stringify({sceneId,bodyA,bodyB,frames,every,meshVertexContacts,jitter,alpha:world.solver.params.alpha,samples},null,2)+'\n');
}finally{world.destroy();scene.dispose();device.destroy();}
