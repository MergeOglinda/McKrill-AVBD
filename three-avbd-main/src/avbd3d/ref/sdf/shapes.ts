/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// sdf-dcd-demo shapes.cpp; Sec. 3.6 and warped-field chain rule in Sec. 4.5.
import { f, add, sub, mul, div, sqrt, sin, cos, sign, clamp, vec, plus, minus, scale, divide, dot, length, expand, type Vec, type Aabb } from './math.ts';
export interface FieldSample { phi: number; gradient: Vec }
export interface Field {
  /** Optional source surface for area-weighted voxel seeds, packed xyz corners. */
  readonly surfaceTriangles?: Float32Array;
  bounds: Aabb;
  convex: boolean;
  centerOfMass: Vec;
  seeds: Vec[];
  sample(p: Vec): FieldSample;
}
const radial = (p: Vec, r: number): Vec => r > 1e-9 ? [div(p[0], r), 0, div(p[2], r)] : [1, 0, 0];
export function box(p: Vec, h: Vec): FieldSample {
  const q: Vec = [sub(Math.abs(p[0]), h[0]), sub(Math.abs(p[1]), h[1]), sub(Math.abs(p[2]), h[2])];
  const outside: Vec = [Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0)];
  const len = length(outside); let gradient: Vec = [0, 0, 0];
  if (len > 0) gradient = [mul(div(outside[0], len), sign(p[0])), mul(div(outside[1], len), sign(p[1])), mul(div(outside[2], len), sign(p[2]))];
  else { const axis = q[0] > q[1] ? (q[0] > q[2] ? 0 : 2) : (q[1] > q[2] ? 1 : 2); gradient[axis] = sign(p[axis]); }
  return { phi: add(len, Math.min(Math.max(...q), 0)), gradient };
}
export function sphere(p: Vec, r: number): FieldSample {
  const len = length(p); return { phi: sub(len, r), gradient: len > 1e-9 ? divide(p, len) : [0, 1, 0] };
}
export function cylinder(p: Vec, r: number, h: number): FieldSample {
  const radialLength = sqrt(add(mul(p[0], p[0]), mul(p[2], p[2])));
  const dx = sub(radialLength, r), dy = sub(Math.abs(p[1]), h), ox = Math.max(dx, 0), oy = Math.max(dy, 0);
  const outside = sqrt(add(mul(ox, ox), mul(oy, oy))), e = radial(p, radialLength);
  return { phi: add(outside, Math.min(Math.max(dx, dy), 0)), gradient: outside > 0 ? plus(scale(e, div(ox, outside)), [0, div(mul(sign(p[1]), oy), outside), 0]) : dx > dy ? e : [0, sign(p[1]), 0] };
}
export function cone(p: Vec, radius: number, h: number): FieldSample {
  const qx = sqrt(add(mul(p[0], p[0]), mul(p[2], p[2]))), qy = p[1];
  const cax = sub(qx, Math.min(qx, qy < 0 ? radius : 0)), cay = sub(Math.abs(qy), h), kx = -radius, ky = mul(2, h);
  const t = clamp(div(add(mul(qx, radius), mul(sub(h, qy), ky)), add(mul(kx, kx), mul(ky, ky))), 0, 1);
  const cbx = add(qx, mul(kx, t)), cby = add(sub(qy, h), mul(ky, t)), s = cbx < 0 && cay < 0 ? -1 : 1;
  const da = add(mul(cax, cax), mul(cay, cay)), db = add(mul(cbx, cbx), mul(cby, cby));
  let gx = da < db ? cax : cbx, gy = da < db ? mul(cay, sign(qy)) : cby;
  let len = sqrt(add(mul(gx, gx), mul(gy, gy)));
  if (len > 1e-9) { gx = mul(gx, div(s, len)); gy = mul(gy, div(s, len)); }
  else if (da < db) { gx = 0; gy = sign(qy); }
  else { gx = ky; gy = radius; len = sqrt(add(mul(gx, gx), mul(gy, gy))); gx = div(gx, len); gy = div(gy, len); }
  return { phi: mul(s, sqrt(Math.min(da, db))), gradient: plus(scale(radial(p, qx), gx), [0, gy, 0]) };
}
export function torus(p: Vec, major: number, minor: number): FieldSample {
  const r = sqrt(add(mul(p[0], p[0]), mul(p[2], p[2]))), qx = sub(r, major), qy = p[1];
  const len = sqrt(add(mul(qx, qx), mul(qy, qy))), e = radial(p, r);
  return { phi: sub(len, minor), gradient: len > 1e-9 ? plus(scale(e, div(qx, len)), [0, div(qy, len), 0]) : e };
}
export type AnalyticSpec =
  | { kind: 'box'; halfExtents: Vec }
  | { kind: 'sphere'; radius: number }
  | { kind: 'capsule'; radius: number; halfLength: number }
  | { kind: 'cylinder' | 'cone'; radius: number; halfHeight: number }
  | { kind: 'torus'; majorRadius: number; minorRadius: number }
  | { kind: 'warp'; type: 0 | 1 | 2; halfExtents: Vec; k: number }
  | { kind: 'spike-plate'; halfExtents: Vec; count: number; spikeRadius: number; minHeight: number; maxHeight: number }
  | { kind: 'spiky-wheel'; hubRadius: number; hubHalfWidth: number; spikeRadius: number; spikeHalfHeight: number; spikes: number };
function warped(p: Vec, h: Vec, type: number, k: number): FieldSample {
  if (type === 2) {
    const s: Vec = p.map(x => sin(mul(7, x))) as Vec, c: Vec = p.map(x => cos(mul(7, x))) as Vec;
    const base = box(p, h);
    return { phi: add(base.phi, mul(mul(mul(k, s[0]), s[1]), s[2])), gradient: plus(base.gradient, scale([mul(mul(c[0], s[1]), s[2]), mul(mul(s[0], c[1]), s[2]), mul(mul(s[0], s[1]), c[2])], mul(k, 7))) };
  }
  const c = cos(mul(k, p[type === 0 ? 0 : 1])), s = sin(mul(k, p[type === 0 ? 0 : 1]));
  const q: Vec = type === 0 ? [sub(mul(c, p[0]), mul(s, p[1])), add(mul(s, p[0]), mul(c, p[1])), p[2]] : [sub(mul(c, p[0]), mul(s, p[2])), add(mul(s, p[0]), mul(c, p[2])), p[1]];
  const base = box(q, h), g = base.gradient;
  const gradient: Vec = type === 0
    ? [dot([sub(c, mul(k, q[1])), add(s, mul(k, q[0])), 0], g), dot([-s, c, 0], g), g[2]]
    : [dot([c, s, 0], g), dot([-mul(k, q[1]), mul(k, q[0]), 1], g), dot([-s, c, 0], g)];
  return { phi: base.phi, gradient };
}
function hash01(x: number): number {
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); x ^= x >>> 16;
  return (x & 0xffffff) / 16777216;
}
export class AnalyticField implements Field {
  readonly spec: AnalyticSpec;
  readonly bounds: Aabb;
  readonly convex: boolean;
  centerOfMass: Vec = [0, 0, 0];
  seeds: Vec[] = [];
  constructor(spec: AnalyticSpec) {
    const dimensions: Record<AnalyticSpec['kind'], string[]> = {
      box: [], sphere: ['radius'], capsule: ['radius', 'halfLength'],
      cylinder: ['radius', 'halfHeight'], cone: ['radius', 'halfHeight'],
      torus: ['majorRadius', 'minorRadius'], warp: [],
      'spike-plate': ['spikeRadius', 'minHeight', 'maxHeight'],
      'spiky-wheel': ['hubRadius', 'hubHalfWidth', 'spikeRadius', 'spikeHalfHeight'],
    };
    if (!Object.hasOwn(dimensions, spec.kind)) throw new Error('Unknown field');
    const values = spec as unknown as Record<string, unknown>;
    for (const key of dimensions[spec.kind]) {
      const value = values[key];
      if (typeof value !== 'number' || !Number.isFinite(f(value)) || (key === 'halfLength' ? value < 0 : f(value) <= 0)) throw new Error(`Invalid SDF ${key}`);
    }
    if ('halfExtents' in spec && (spec.halfExtents.length !== 3 || !spec.halfExtents.every(v => Number.isFinite(f(v)) && f(v) > 0))) throw new Error('Invalid SDF halfExtents');
    if (spec.kind === 'warp' && (![0, 1, 2].includes(spec.type) || !Number.isFinite(f(spec.k)))) throw new Error('Invalid SDF warp');
    if (spec.kind === 'spike-plate' && (!Number.isSafeInteger(spec.count) || spec.count < 1 || spec.maxHeight < spec.minHeight)) throw new Error('Invalid SDF spike count/heights');
    if (spec.kind === 'spiky-wheel' && (!Number.isSafeInteger(spec.spikes) || spec.spikes < 1)) throw new Error('Invalid SDF spike count');
    // Float32 constructor parameters, matching upstream's float arguments.
    this.spec = Object.fromEntries(Object.entries(spec).map(([key, value]) => [key, typeof value === 'number' ? f(value) : Array.isArray(value) ? vec(value) : value])) as AnalyticSpec;
    const s = this.spec; let h: Vec;
    this.convex = ['box', 'sphere', 'capsule', 'cylinder', 'cone'].includes(s.kind);
    if (s.kind === 'box' || s.kind === 'spike-plate') h = s.halfExtents;
    else if (s.kind === 'sphere') h = [s.radius, s.radius, s.radius];
    else if (s.kind === 'capsule') h = [s.radius, add(s.radius, s.halfLength), s.radius];
    else if (s.kind === 'cone' || s.kind === 'cylinder') h = [s.radius, s.halfHeight, s.radius];
    else if (s.kind === 'torus') h = [add(s.majorRadius, s.minorRadius), s.minorRadius, add(s.majorRadius, s.minorRadius)];
    else if (s.kind === 'spiky-wheel') h = [add(s.hubRadius, mul(2, s.spikeHalfHeight)), s.hubHalfWidth, add(s.hubRadius, mul(2, s.spikeHalfHeight))];
    else if (s.kind === 'warp') {
      const r = sqrt(add(mul(s.halfExtents[0], s.halfExtents[0]), mul(s.halfExtents[1], s.halfExtents[1])));
      h = s.type === 0 ? [r, r, s.halfExtents[2]] : s.type === 1 ? [r, s.halfExtents[2], r] : s.halfExtents;
    } else throw new Error('Unknown field');
    this.bounds = { min: scale(h, -1), max: [...h] };
    if (s.kind === 'warp' && s.type === 2) this.bounds = expand(this.bounds, Math.abs(s.k));
    if (s.kind === 'spike-plate') this.bounds.max[1] = add(h[1], s.maxHeight);
    if (![...this.bounds.min, ...this.bounds.max].every(Number.isFinite)) throw new Error('SDF bounds exceed float32 range');
  }
  sample(input: Vec): FieldSample {
    const p = vec(input), s = this.spec;
    switch (s.kind) {
      case 'box': return box(p, s.halfExtents);
      case 'sphere': return sphere(p, s.radius);
      case 'capsule': return sphere([p[0], sub(p[1], clamp(p[1], -s.halfLength, s.halfLength)), p[2]], s.radius);
      case 'cylinder': return cylinder(p, s.radius, s.halfHeight);
      case 'cone': return cone(p, s.radius, s.halfHeight);
      case 'torus': return torus(p, s.majorRadius, s.minorRadius);
      case 'warp': return warped(p, s.halfExtents, s.type, s.k);
      case 'spike-plate': {
        let result = box(p, s.halfExtents); const pitch = scale(s.halfExtents, div(2, s.count));
        const ci = clamp(Math.floor(div(add(p[0], s.halfExtents[0]), pitch[0])), 0, s.count - 1), cj = clamp(Math.floor(div(add(p[2], s.halfExtents[2]), pitch[2])), 0, s.count - 1);
        for (let j = Math.max(cj - 1, 0); j <= Math.min(cj + 1, s.count - 1); j++) for (let i = Math.max(ci - 1, 0); i <= Math.min(ci + 1, s.count - 1); i++) {
          const h = add(s.minHeight, mul(sub(s.maxHeight, s.minHeight), hash01(17 * (i + j * s.count) + 11)));
          const c: Vec = [add(-s.halfExtents[0], mul(add(i, 0.5), pitch[0])), sub(add(s.halfExtents[1], mul(0.5, h)), mul(0.5, s.spikeRadius)), add(-s.halfExtents[2], mul(add(j, 0.5), pitch[2]))];
          const candidate = cone(minus(p, c), div(mul(s.spikeRadius, add(h, s.spikeRadius)), h), mul(0.5, add(h, s.spikeRadius)));
          if (candidate.phi < result.phi) result = candidate;
        }
        return result;
      }
      case 'spiky-wheel': {
        let result = cylinder(p, s.hubRadius, s.hubHalfWidth);
        const step = div(mul(2, f(Math.PI)), s.spikes), ring = sub(s.hubHalfWidth, s.spikeRadius), nearest = Math.floor(add(div(f(Math.atan2(p[2], p[0])), step), 0.5));
        for (let k = nearest - 1; k <= nearest + 1; k++) {
          const a = mul(k, step), side: Vec = [-sin(a), 0, cos(a)], dir: Vec = [cos(a), 0, sin(a)];
          for (const y of [-ring, ring]) {
            const r = minus(minus(p, scale(dir, add(s.hubRadius, s.spikeHalfHeight))), [0, y, 0]);
            const candidate = cone([dot(r, side), dot(r, dir), r[1]], s.spikeRadius, s.spikeHalfHeight);
            if (candidate.phi < result.phi) result = { phi: candidate.phi, gradient: plus(plus(scale(side, candidate.gradient[0]), scale(dir, candidate.gradient[1])), [0, candidate.gradient[2], 0]) };
          }
        }
        return result;
      }
    }
  }
}
