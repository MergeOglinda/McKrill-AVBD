import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gpuTest } from './device.ts';
import { loadReleasedScene, instantiateReleased, releasedScenes, type ReadAsset } from '../src/sdf-released/scenes.ts';
import { measuredFrame } from '../src/sdf-released/metrics.ts';
import { BODY_FLOATS, B_SIZE, B_MOMENT } from '../src/avbd3d/gpu/layout.ts';
const read:ReadAsset=async path=>{const b=await readFile(new URL(`../tests/fixtures/${path}`,import.meta.url));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
for(const entry of releasedScenes)gpuTest(`released SDF ${entry.name}: original mass and bounded 30-frame collision smoke`,async device=>{
  const scene=await loadReleasedScene(entry.id,read);
  const {world,bodies}=await instantiateReleased(scene,{device});
  try{
    const data=await world.solver.readBodies();
    scene.bodies.forEach((b,i)=>{
      assert.equal(data[i*BODY_FLOATS+B_SIZE+3],Math.fround(b.original.mass));
      if(!b.original.static)b.moments.forEach((v,k)=>assert.equal(data[i*BODY_FLOATS+B_MOMENT+k],Math.fround(v)));
    });
    for(let frame=1;frame<=30;frame++){
      const sample=await measuredFrame(scene,world,bodies,frame);
      assert.ok(sample.minCenterY>0,'no body COM passes through the floor');
      assert.ok(Number.isFinite(sample.kineticEnergy));
      if(frame===30)assert.ok(sample.stats.contacts>0,'scene creates active contacts');
    }
  }finally{world.destroy();scene.dispose();}
});

gpuTest('released chair pile settles without the previous motion and deep-penetration regression',async device=>{
  const scene=await loadReleasedScene(6,read);
  const {world,bodies}=await instantiateReleased(scene,{device});
  try{
    let peak=0,last;
    for(let frame=1;frame<=360;frame++){
      last=await measuredFrame(scene,world,bodies,frame,frame%30===0);
      peak=Math.max(peak,last.maxVertexPenetration??0);
      await new Promise(resolve=>setTimeout(resolve,5));
    }
    assert.ok(peak<.04,`sampled impact penetration ${peak} m`);
    assert.ok(last!.maxVertexPenetration!<.005,`final penetration ${last!.maxVertexPenetration} m`);
    assert.ok(last!.kineticEnergy<.01,`residual kinetic energy ${last!.kineticEnergy} J`);
  }finally{world.destroy();scene.dispose();}
});

gpuTest('released bunny pile bounds transient penetration and final overlap',async device=>{
  const scene=await loadReleasedScene(9,read);
  const {world,bodies}=await instantiateReleased(scene,{device});
  try{
    let peak=0,last;
    for(let frame=1;frame<=360;frame++){
      last=await measuredFrame(scene,world,bodies,frame,frame%30===0);
      peak=Math.max(peak,last.maxVertexPenetration??0);
      await new Promise(resolve=>setTimeout(resolve,5));
    }
    assert.ok(peak<.03,`sampled impact penetration ${peak} m`);
    assert.ok(last!.maxVertexPenetration!<.01,`final penetration ${last!.maxVertexPenetration} m`);
    assert.ok(last!.kineticEnergy<.01,`residual kinetic energy ${last!.kineticEnergy} J`);
  }finally{world.destroy();scene.dispose();}
});

gpuTest('released bunny contact ordering is repeatable across independent GPU runs',async device=>{
  const scene=await loadReleasedScene(9,read);
  try{
    let expected:Float32Array|undefined;
    for(let run=0;run<2;run++){
      const {world,bodies}=await instantiateReleased(scene,{device});
      try{
        for(let frame=1;frame<=180;frame++){
          await measuredFrame(scene,world,bodies,frame);
          await new Promise(resolve=>setTimeout(resolve,5));
        }
        const state=await world.solver.readBodies();
        // Compare physical state, not transient scratch or atomic allocation slots.
        const poses=new Float32Array(bodies.flatMap(b=>[...b.position,...b.rotation,...b.velocity,...b.angularVelocity]));
        assert.ok(state.every(Number.isFinite));
        if(expected)assert.deepEqual(poses,expected,'persistent point IDs break same-pair adjacency ties');
        else expected=poses;
      }finally{world.destroy();}
    }
  }finally{scene.dispose();}
});
