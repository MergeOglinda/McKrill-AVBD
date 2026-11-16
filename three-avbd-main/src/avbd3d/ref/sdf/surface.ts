/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// VoxelSdf::surfaceCandidates: preserve triangle/corner order and f32 PCG draws.
import { add, mul, sub, sqrt, cross, length, minus, plus, scale, Random, type Vec } from './math.ts';
/** Packed triangle corners (nine floats/triangle), in field-local coordinates. */
export function meshSurfaceSampler(triangles: Float32Array): () => Vec {
  if (!triangles.length || triangles.length % 9 || !triangles.every(Number.isFinite)) throw new Error('Invalid SDF surface triangles');
  const vertex = (t: number, corner: number): Vec => {
    const i = t * 9 + corner * 3; return [triangles[i], triangles[i + 1], triangles[i + 2]];
  };
  const cumulative = new Float32Array(triangles.length / 9); let total = 0;
  for (let t = 0; t < cumulative.length; t++) {
    const normal = cross(minus(vertex(t, 1), vertex(t, 0)), minus(vertex(t, 2), vertex(t, 0)));
    total = add(total, mul(.5, length(normal)));
    cumulative[t] = total;
  }
  if (!(total > 0) || !Number.isFinite(total)) throw new Error('SDF surface area must be positive and finite');
  const rng = new Random();
  return () => {
    const target = mul(rng.uniform(), total); let lo = 0, hi = cumulative.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (cumulative[mid] < target) lo = mid + 1; else hi = mid; }
    const t = Math.min(lo, cumulative.length - 1), r1 = sqrt(rng.uniform()), r2 = rng.uniform();
    return plus(plus(scale(vertex(t, 0), sub(1, r1)), scale(vertex(t, 1), mul(r1, sub(1, r2)))), scale(vertex(t, 2), mul(r1, r2)));
  };
}
