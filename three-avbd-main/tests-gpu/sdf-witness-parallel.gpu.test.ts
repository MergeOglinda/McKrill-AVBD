import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Quaternion, Euler } from 'three';
import { gpuTest } from './device.ts';
import { loadReleasedScene, type ReadAsset } from '../src/sdf-released/scenes.ts';
import { SdfBody } from '../src/avbd3d/ref/sdf/body.ts';
import { SdfNarrowphasePass } from '../src/avbd3d/gpu/sdf-pass.ts';
import { collisionParams } from '../src/avbd3d/ref/sdf/manifold.ts';
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
gpuTest('cooperative mesh witnesses retain serial contact order, IDs and history',async device=>{
 const scene=await loadReleasedScene(9,read),mesh=scene.bodies[1],ground=scene.bodies[0];
 const fields=[ground.field,mesh.field];
 const direct=await SdfNarrowphasePass.create(device,fields,{maxBodies:3,maxPairs:3,meshVertexContacts:true,parallelMeshWitnesses:false});
 let parallel=await SdfNarrowphasePass.create(device,fields,{maxBodies:3,maxPairs:3,meshVertexContacts:true});
 try{
  for(let frame=0;frame<24;frame++){
   if(frame===12){const grown=parallel.resized(5);grown.inheritHistory(parallel,[true,true,true]);parallel.destroy();parallel=grown;}
   const rotation=new Quaternion().setFromEuler(new Euler(.03*frame,.07*frame,.02*frame)).toArray() as [number,number,number,number];
   const bodies=[new SdfBody(ground.field,[0,-.2,0],ground.rotation,ground.original.scale,true),new SdfBody(mesh.field,[0,.35,0],rotation,.8),new SdfBody(mesh.field,[.18,.6,.04],rotation,1.1)];
   const pairs=frame%2?[{a:2,b:1,key:3,previous:2},{a:0,b:1,key:1,previous:0},{a:1,b:2,key:2,previous:1}]:[{a:0,b:1,key:1,previous:1},{a:1,b:2,key:2,previous:2},{a:2,b:1,key:3,previous:0}];
   if(frame===0)for(const p of pairs)delete (p as {previous?:number}).previous;
   const params={...collisionParams(),meshVertexContacts:frame%7!==6,signedSeparation:true,refreshDeeperContacts:true,optimizer:frame%3 as 0|1|2};
   direct.step(bodies,pairs,.004,params);parallel.step(bodies,pairs,.004,params);
   assert.deepEqual(await parallel.readResults(),await direct.readResults(),`frame ${frame}`);
  }
 }finally{direct.destroy();parallel.destroy();scene.dispose();}
});
