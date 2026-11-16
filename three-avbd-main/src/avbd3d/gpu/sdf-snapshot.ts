import { AnalyticField, type Field } from '../ref/sdf/shapes.ts';
import { VoxelField } from '../ref/sdf/voxel.ts';

/** Retained CPU snapshots let later atlas rebuilds ignore mutations of caller-owned fields. */
export function snapshotSdfField(field:Field, meshVertexContacts=false):Field {
  let copy:Field;
  if(field instanceof AnalyticField)copy=new AnalyticField(structuredClone(field.spec));
  else if(field instanceof VoxelField)copy=new VoxelField(field.dimensions,field.origin,field.cell,field.grid.slice(),field.bounds,meshVertexContacts?field.surfaceTriangles?.slice():undefined);
  else throw new Error('Unsupported GPU SDF: bake custom fields to voxels first');
  copy.bounds.min=[...field.bounds.min];copy.bounds.max=[...field.bounds.max];
  copy.convex=field.convex;copy.centerOfMass=[...field.centerOfMass];copy.seeds=field.seeds.map(p=>[...p]);
  return copy;
}
