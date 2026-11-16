// Experimental SDF preparation; World.enableSdfCollisions accepts these fields.
export type { SdfBodyBinding } from './world.ts';
export { bakeVoxelSdf, bakeVoxelSdfAsync, type TriangleMesh, type BakeOptions, type BakedSdf } from '../avbd3d/ref/sdf/bake.ts';
export { finalizeSdf, curvatureSeeds, type CurvatureSeedOptions, type MassProperties, type FinalizeOptions } from '../avbd3d/ref/sdf/finalize.ts';
export { AnalyticField, type AnalyticSpec, type Field, type FieldSample } from '../avbd3d/ref/sdf/shapes.ts';
export { VoxelField } from '../avbd3d/ref/sdf/voxel.ts';
export type { Vec as SdfVector, Aabb as SdfBounds } from '../avbd3d/ref/sdf/math.ts';
export { installSdfWorker, bakeVoxelSdfInWorker, type SdfWorkerScope } from '../avbd3d/ref/sdf/worker.ts';
export { collisionParams, scaleCollisionParams, type CollisionParams } from '../avbd3d/ref/sdf/manifold.ts';

export type { SdfBroadphaseOptions } from '../avbd3d/gpu/sdf-broadphase.ts';
export { principalInertia } from './inertia.ts';
