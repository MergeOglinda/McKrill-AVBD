/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// Unit-density integration and projected curvature seeds from Shape::finalize.
import { f, add, mul, div, sub, size, expand, Random, plus, minus, scale, dot, type Vec } from './math.ts';
import { meshSurfaceSampler } from './surface.ts';
import type { Field } from './shapes.ts';
export interface MassProperties {
  volume: number;
  centerOfMass: Vec;
  /** Row-major full inertia tensor about COM, at unit density. */
  inertia: number[];
}
export interface FinalizeOptions { resolution?: number; candidateCount?: number; signal?: AbortSignal }
/** Mutates COM/seeds only after successful completion. CPU/offline operation. */
export function finalizeSdf(field: Field, options: FinalizeOptions = {}): MassProperties {
  const resolution = options.resolution ?? 48;
  if (!Number.isInteger(resolution) || resolution < 4 || resolution > 128) throw new Error('Mass resolution must be 4–128');
  const check = () => { if (options.signal?.aborted) throw new DOMException('SDF finalization aborted', 'AbortError'); };
  check();
  const extent = size(field.bounds), L = Math.max(...extent);
  if (!extent.every(v => Number.isFinite(v) && v > 0)) throw new Error('SDF must have finite positive bounds');
  const h = div(L, resolution), n = extent.map(v => Math.max(1, Math.ceil(div(v, h))));
  const cell = extent.map((v, i) => div(v, n[i])) as Vec, dv = mul(mul(cell[0], cell[1]), cell[2]);
  let mass = 0; const first = [0, 0, 0], second = new Array<number>(9).fill(0);
  for (let z = 0; z < n[2]; z++) {
    check();
    for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++) {
      const c = plus(field.bounds.min, [mul(cell[0], x + .5), mul(cell[1], y + .5), mul(cell[2], z + .5)]);
      if (field.sample(c).phi >= 0) continue;
      mass += dv;
      for (let i = 0; i < 3; i++) {
        first[i] += mul(c[i], dv);
        for (let j = 0; j < 3; j++) second[i * 3 + j] += add(mul(mul(c[i], c[j]), dv), i === j ? mul(div(mul(cell[i], cell[i]), 12), dv) : 0);
      }
    }
  }
  if (!(mass > 0)) throw new Error('No interior sampled; increase resolution or repair mesh');
  const centerOfMass = first.map(v => f(v / mass)) as Vec;
  const S = second.map((v, i) => f(v - mass * centerOfMass[Math.floor(i / 3)] * centerOfMass[i % 3]));
  const trace = add(add(S[0], S[4]), S[8]), inertia = S.map((v, i) => sub(i % 4 === 0 ? trace : 0, v));
  const seeds = curvatureSeeds(field, options);
  check(); field.centerOfMass = centerOfMass; field.seeds = seeds;
  return { volume: f(mass), centerOfMass: [...centerOfMass], inertia };
}

export type CurvatureSeedOptions = Pick<FinalizeOptions, 'candidateCount' | 'signal'>;
/** Returns ordered Eq. 7 seeds without modifying the field or computing mass. */
export function curvatureSeeds(field: Field, options: CurvatureSeedOptions = {}): Vec[] {
  const count = options.candidateCount ?? 8000;
  if (!Number.isInteger(count) || count < 0 || count > 100000) throw new Error('candidateCount must be 0–100000');
  const check = () => { if (options.signal?.aborted) throw new DOMException('SDF seeds aborted', 'AbortError'); };
  check();
  const extent = size(field.bounds), L = Math.max(...extent);
  if (!extent.every(v => Number.isFinite(v) && v > 0)) throw new Error('SDF must have finite positive bounds');
  const rng = new Random(), box = expand(field.bounds, mul(.1, L)), seeds: Vec[] = [];
  const surface = field.surfaceTriangles && count ? meshSurfaceSampler(field.surfaceTriangles) : undefined;
  const delta = mul(.001, L), kappa = div(4, L), spacingSq = mul(mul(.08, L), mul(.08, L));
  for (let i = 0; i < count; i++) {
    check(); let p = surface ? surface() : rng.inBox(box);
    for (let j = 0; !surface && j < 32; j++) {
      const { phi, gradient } = field.sample(p);
      if (Math.abs(phi) < mul(1e-5, L)) break;
      p = minus(p, scale(gradient, div(phi, Math.max(dot(gradient, gradient), f(1e-8)))));
    }
    if (!surface && Math.abs(field.sample(p).phi) >= mul(.001, L)) continue;
    let laplacian = 0;
    for (let axis = 0; axis < 3; axis++) {
      const e: Vec = [0, 0, 0]; e[axis] = delta;
      laplacian = add(laplacian, div(sub(field.sample(plus(p, e)).gradient[axis], field.sample(minus(p, e)).gradient[axis]), mul(2, delta)));
    }
    if (Math.abs(laplacian) <= kappa || seeds.some(s => { const d = minus(s, p); return dot(d, d) < spacingSq; })) continue;
    seeds.push(p);
  }
  check(); return seeds;
}
