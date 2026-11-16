// Bounded audit of detector separation versus retained solver material anchors.
import { readFile, writeFile } from 'node:fs/promises';
import { Quaternion, Vector3 } from 'three';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, type ReadAsset } from '../src/sdf-released/scenes.ts';
const sceneId=Number(process.argv[2]),frames=Number(process.argv[3]),output=process.argv[4],meshVertexContacts=process.argv[5]==='mesh-plane';
if(![6,9].includes(sceneId)||!Number.isInteger(frames)||frames<1||frames>360||!output||(process.argv[5]!==undefined&&!meshVertexContacts))throw new Error('Usage: sdf-contact-audit.ts scene(6|9) frames(1..360) output.json [mesh-plane]');
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);
const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No GPU adapter');
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
(globalThis as typeof globalThis & {__sdfAuditGpu?:unknown}).__sdfAuditGpu=[gpu,adapter,device];
const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
const source=await loadReleasedScene(sceneId,read);
const scene=meshVertexContacts?{...source,capture:{...source.capture,params:{...source.capture.params,meshVertexContacts:true}}}:source;
const {world,bodies}=await instantiateReleased(scene,{device});
try{
  for(let frame=1;frame<=frames;frame++){
    for(let substep=0;substep<Math.round(scene.capture.solver.dt/world.dt);substep++){world.step();await world.readSdfResults();}
    await new Promise(resolve=>setTimeout(resolve,5));
  }
  await world.read();const results=await world.readSdfResults(),contacts=await world.solver.readContactList();
  const audited=contacts.map(c=>{
    const point=results.find(r=>r.a===c.a&&r.b===c.b)?.points.find(p=>p.id===c.feature);
    const anchor=(i:number,r:number[])=>new Vector3(...r).applyQuaternion(new Quaternion(...bodies[i].rotation)).add(new Vector3(...bodies[i].position));
    return {a:c.a,b:c.b,id:c.feature,stick:c.stick,detected:point?.separation,
      retained:anchor(c.a,c.rA).sub(anchor(c.b,c.rB)).dot(new Vector3(...c.normal)),
      moved:point?Math.max(new Vector3(...point.anchorA).distanceTo(new Vector3(...c.rA)),new Vector3(...point.anchorB).distanceTo(new Vector3(...c.rB))):null,
      normal:c.normal,force:c.lam};
  });
  if(errors.length)throw new Error(errors.join('\n'));
  await writeFile(output,JSON.stringify({sceneId,frames,meshVertexContacts,contacts:audited},null,2)+'\n');
}finally{world.destroy();scene.dispose();device.destroy();}
