import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BoxGeometry } from 'three';
import { bakeVoxelSdf } from '../src/avbd3d/ref/sdf/bake.ts';
import { meshPlaneSupports, meshSurfaceVertices, MESH_WITNESS_MAX_VERTICES } from '../src/avbd3d/gpu/sdf-supports.ts';
import { snapshotSdfField } from '../src/avbd3d/gpu/sdf-snapshot.ts';
import { packSdfAtlas } from '../src/avbd3d/gpu/sdf-layout.ts';

test('mesh plane supports preserve extrema, source seeds and immutable snapshot data',()=>{
  const geometry=new BoxGeometry(1,2,3),field=bakeVoxelSdf(geometry,{resolution:8}).field;
  try{
    const seeds=structuredClone(field.seeds),snapshot=snapshotSdfField(field,true),points=meshPlaneSupports(snapshot);
    assert.equal(points.length,8);
    assert.deepEqual(points,meshPlaneSupports(snapshot));
    const surface=snapshot.surfaceTriangles!;
    for(let i=0;i<80;i++){
      const direction=[Math.sin(i*1.7),Math.cos(i*.9),Math.sin(i*.37)],dot=(p:number[])=>p.reduce((sum,v,k)=>sum+v*direction[k],0);
      let expected=-Infinity;
      for(let j=0;j<surface.length;j+=3)expected=Math.max(expected,dot(Array.from(surface.slice(j,j+3))));
      assert.equal(Math.max(...points.map(dot)),expected,'hull keeps exact plane supports');
    }
    const atlas=packSdfAtlas([snapshot,snapshot],64*1024*1024,true),words=atlas.data,floats=new Float32Array(words.buffer);
    assert.equal(packSdfAtlas([snapshot]).data[15],0,'default atlas has no support allocation');
    assert.equal(snapshotSdfField(field).surfaceTriangles,undefined);
    assert.equal(atlas.offsets.size,1);assert.equal(words[3],seeds.length);
    assert.equal(words[15],8);
    points.forEach((point,i)=>assert.deepEqual(Array.from(floats.slice((words[2]+words[3]+i)*4,(words[2]+words[3]+i)*4+3)),point));
    field.surfaceTriangles!.fill(100);
    assert.deepEqual(meshPlaneSupports(snapshot),points,'caller mutations cannot change rebuilt atlas');
    assert.deepEqual(field.seeds,seeds,'source RNG seed population is untouched');
  }finally{geometry.dispose();}
});

test('mesh-to-mesh witnesses pack every unique surface vertex, not only the hull',()=>{
  const geometry=new BoxGeometry(1,1,1,2,2,2),field=bakeVoxelSdf(geometry,{resolution:8}).field;
  try{
    const snapshot=snapshotSdfField(field,true),vertices=meshSurfaceVertices(snapshot),hull=meshPlaneSupports(snapshot);
    assert.equal(vertices.length,26,'3x3x3 surface grid without its centre');
    assert.equal(hull.length,8);
    assert.deepEqual(vertices,meshSurfaceVertices(snapshot),'stable ordering');
    const keys=new Set(vertices.map(v=>v.join(',')));
    assert.ok(hull.every(v=>keys.has(v.join(','))),'plane extrema remain available');
    const atlas=packSdfAtlas([snapshot],64*1024*1024,true);
    assert.equal(atlas.data[15],26);
  }finally{geometry.dispose();}
});

test('mesh vertex contacts refuse fields beyond the per-pair scan budget',()=>{
  const side=Math.ceil(Math.sqrt(MESH_WITNESS_MAX_VERTICES))+1,geometry=new BoxGeometry(1,1,1,side,side,side),field=bakeVoxelSdf(geometry,{resolution:8}).field;
  try{
    const snapshot=snapshotSdfField(field,true);
    assert.throws(()=>packSdfAtlas([snapshot],64*1024*1024,true),/at most 4096 unique surface vertices/);
    assert.equal(packSdfAtlas([snapshot]).data[15],0,'default atlas is unaffected');
  }finally{geometry.dispose();}
});
