// Bounded substep trace around a known body/floor impact; no solver settings changed.
import { readFile, writeFile } from 'node:fs/promises';
import * as THREE from 'three';
import { create, globals } from 'webgpu';
import { loadReleasedScene, instantiateReleased, type ReadAsset } from '../src/sdf-released/scenes.ts';
const [sceneId,bodyId,start,end]=process.argv.slice(2,6).map(Number),output=process.argv[6];
if(![6,9].includes(sceneId)||!Number.isInteger(bodyId)||bodyId<1||bodyId>27||!Number.isInteger(start)||!Number.isInteger(end)||start<1||end<start||end>360||end-start>30||!output)throw new Error('Usage: sdf-impact-trace.ts scene(6|9) body start end output.json (window <= 30 frames)');
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
Object.assign(globalThis,globals);
const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No GPU adapter');
const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage}});
// Keep Dawn wrappers alive until all asynchronous readbacks complete.
(globalThis as typeof globalThis & {__sdfTraceGpu?:unknown}).__sdfTraceGpu=[gpu,adapter,device];
const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
const scene=await loadReleasedScene(sceneId,read),{world,bodies}=await instantiateReleased(scene,{device});
const traceField=scene.bodies[bodyId].field;
const traceSeeds=traceField.seeds;
const samples=[];
try{
  for(let frame=1;frame<=end;frame++){
    for(let substep=0;substep<Math.round(scene.capture.solver.dt/world.dt);substep++){
      world.step();const results=await world.readSdfResults();
      if(frame<start)continue;
      await world.read();
      const body=bodies[bodyId],rotation=new THREE.Quaternion(...body.rotation),origin=new THREE.Vector3(...body.position);
      const vertices=scene.bodies[bodyId].geometry.getAttribute('position');let minY=Infinity;
      for(let i=0;i<vertices.count;i++)minY=Math.min(minY,new THREE.Vector3().fromBufferAttribute(vertices,i).applyQuaternion(rotation).add(origin).y);
      let minSeedY=Infinity;
      const fieldRotation=rotation.clone().multiply(new THREE.Quaternion(...scene.bodies[bodyId].fieldRotation));
      const original=scene.bodies[bodyId].original,field=traceField;
      for(const seed of traceSeeds)minSeedY=Math.min(minSeedY,new THREE.Vector3(...seed).sub(new THREE.Vector3(...field.centerOfMass)).multiplyScalar(original.scale).applyQuaternion(fieldRotation).add(origin).y);
      const pair=results.find(r=>r.a===bodyId&&r.b===0);
      const contacts=(await world.solver.readContactList()).filter(c=>c.a===bodyId&&c.b===0).map(c=>{
        const a=new THREE.Vector3(...c.rA).applyQuaternion(rotation).add(origin);
        const b=new THREE.Vector3(...c.rB).applyQuaternion(new THREE.Quaternion(...bodies[0].rotation)).add(new THREE.Vector3(...bodies[0].position));
        return {id:c.feature,stick:c.stick,normal:c.normal,force:c.lam,anchorGap:a.sub(b).dot(new THREE.Vector3(...c.normal)),detected:pair?.points.find(p=>p.id===c.feature)?.separation};
      });
      samples.push({frame,substep,minY,minSeedY,velocity:body.velocity,angularVelocity:body.angularVelocity,contacts});
    }
    await device.queue.onSubmittedWorkDone();await new Promise(resolve=>setTimeout(resolve,5));
  }
  if(errors.length)throw new Error(errors.join('\n'));
  await writeFile(output,JSON.stringify({sceneId,bodyId,start,end,dt:world.dt,samples},null,2)+'\n');
}finally{world.destroy();scene.dispose();device.destroy();}
