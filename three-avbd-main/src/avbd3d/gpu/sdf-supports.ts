import { Vector3 } from 'three';
import { ConvexHull } from 'three/addons/math/ConvexHull.js';
import type { Field } from '../ref/sdf/shapes.ts';
import type { Vec } from '../ref/sdf/math.ts';

/** A plane's support lies on the mesh's convex hull, even for a concave mesh.
 * Keep actual vertices (not an inflated proxy) and a stable feature ordering. */
export function meshPlaneSupports(field:Field):Vec[] {
  if(!field.surfaceTriangles?.length)return [];
  const unique=new Map<string,Vector3>();
  const surface=field.surfaceTriangles;
  for(let i=0;i<surface.length;i+=3){
    const p=new Vector3(surface[i],surface[i+1],surface[i+2]);unique.set(`${p.x},${p.y},${p.z}`,p);
  }
  const points=[...unique.values()],vertices=new Set<Vector3>();
  const origin=points[0],edge=new Vector3(),normal=new Vector3(),delta=new Vector3(),cross=new Vector3();
  for(const p of points){delta.subVectors(p,origin);if(delta.lengthSq()>edge.lengthSq())edge.copy(delta);}
  for(const p of points){cross.crossVectors(edge,delta.subVectors(p,origin));if(cross.lengthSq()>normal.lengthSq())normal.copy(cross);}
  let volume=0;for(const p of points)volume=Math.max(volume,Math.abs(normal.dot(delta.subVectors(p,origin))));
  // Degenerate input cannot initialize a 3D hull; retaining all vertices is exact.
  const hull=points.length>=4 && volume>1e-12*Math.pow(edge.length(),3)?new ConvexHull().setFromPoints(points):null;
  for(const face of hull?.faces??[]){let edge=face.edge;do{vertices.add(edge.head().point);edge=edge.next;}while(edge!==face.edge);}
  return [...(vertices.size?vertices:points)].sort((a,b)=>a.x-b.x||a.y-b.y||a.z-b.z).map(p=>p.toArray() as Vec);
}

/** Unique vertices per field a pair scans serially each substep (shipped chair 164, bunny 502). */
export const MESH_WITNESS_MAX_VERTICES=4096;

/** Every unique source vertex in stable order: exact witnesses for mesh-to-mesh
 * overlap, which (unlike a plane's support) need not lie on the convex hull. */
export function meshSurfaceVertices(field:Field):Vec[] {
  if(!field.surfaceTriangles?.length)return [];
  const unique=new Map<string,Vec>(),surface=field.surfaceTriangles;
  for(let i=0;i<surface.length;i+=3)unique.set(`${surface[i]},${surface[i+1]},${surface[i+2]}`,[surface[i],surface[i+1],surface[i+2]]);
  // One GPU thread scans these per pair and substep: refuse rather than risk a stalled dispatch.
  if(unique.size>MESH_WITNESS_MAX_VERTICES)throw new Error(`meshVertexContacts supports at most ${MESH_WITNESS_MAX_VERTICES} unique surface vertices per field (got ${unique.size}); decimate the source mesh`);
  return [...unique.values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1]||a[2]-b[2]);
}
