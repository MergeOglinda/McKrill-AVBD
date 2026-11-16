import { eigenSymmetric, quatFromBasis } from '../avbd3d/hull.ts';
import type { Quat, Vec3 } from './world.ts';

/** Diagonalize a row-major symmetric COM inertia tensor, without discarding products
 * of inertia. rotation maps principal-frame coordinates into the input frame. */
export function principalInertia(tensor: ArrayLike<number>): { moments: Vec3; rotation: Quat } {
  if (tensor.length !== 9 || !Array.from(tensor).every(Number.isFinite)) throw new Error('Inertia tensor must contain nine finite values');
  const scale = Math.max(...Array.from(tensor, Math.abs));
  if (!(scale > 0)) throw new Error('Inertia tensor must be positive definite');
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) {
    if (Math.abs(tensor[3*i+j] - tensor[3*j+i]) > scale * 1e-10) throw new Error('Inertia tensor must be symmetric');
  }
  const matrix = Array.from({length: 3}, (_, i) => Array.from({length: 3}, (_, j) => (tensor[3*i+j] + tensor[3*j+i]) / (2 * scale)));
  const { values, vectors } = eigenSymmetric(matrix);
  const moments = values.map(v => v * scale) as Vec3;
  if (!moments.every(v => Math.fround(v) > 0 && Number.isFinite(Math.fround(v)))) throw new Error('Inertia tensor must be positive definite in float32');
  return { moments, rotation: quatFromBasis(vectors) };
}
