/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
import { add, sub, mul, div, sqrt, plus, minus, scale, divide, dot, length, project, Random, type Vec, type Aabb } from './math.ts';
import type { SdfBody } from './body.ts';
export const softmax = (a: number, b: number, epsilon: number) => mul(0.5, add(add(a, b), sqrt(add(mul(sub(a, b), sub(a, b)), epsilon))));
export interface Counters { pairs: number; contacts: number; queries: number; iterations: number }
export const counters = (): Counters => ({ pairs: 0, contacts: 0, queries: 0, iterations: 0 });
/** Eq. 3: softmax distance minus repulsion from the current manifold. */
export function objective(a: SdfBody, b: SdfBody, points: { x: Vec }[], epsilon: number, alpha: number, stats: Counters): (x: Vec) => number {
  return x => {
    stats.queries++;
    let value = softmax(a.sample(x).phi, b.sample(x).phi, epsilon);
    for (const p of points) value = sub(value, mul(alpha, length(minus(x, p.x))));
    return value;
  };
}
/** Sec. 3.3: upstream domain-clamped Nelder–Mead; stable insertion order for ties. */
export function nelderMead(g: (x: Vec) => number, seed: Vec, domain: Aabb, initialStep: Vec, tolerance: number, limit: number): { x: Vec; iterations: number } {
  const x = project(seed, domain), v: Vec[] = [x, [...x], [...x], [...x]];
  for (let i = 0; i < 3; i++) {
    v[i + 1][i] = add(v[i + 1][i], add(x[i], initialStep[i]) <= domain.max[i] ? initialStep[i] : -initialStep[i]);
    v[i + 1] = project(v[i + 1], domain);
  }
  const values = v.map(g); let it = 0;
  for (; it < limit; it++) {
    for (let i = 1; i < 4; i++) for (let j = i; j > 0 && values[j] < values[j - 1]; j--) {
      [values[j], values[j - 1]] = [values[j - 1], values[j]]; [v[j], v[j - 1]] = [v[j - 1], v[j]];
    }
    const extent = Math.max(length(minus(v[1], v[0])), length(minus(v[2], v[0])), length(minus(v[3], v[0])));
    if (extent < tolerance || sub(values[3], values[0]) < mul(0.05, tolerance)) break;
    const centroid = divide(plus(plus(v[0], v[1]), v[2]), 3);
    const reflected = project(plus(centroid, minus(centroid, v[3])), domain), fr = g(reflected);
    if (fr < values[0]) {
      const expanded = project(plus(centroid, scale(minus(centroid, v[3]), 2)), domain), fe = g(expanded);
      v[3] = fe < fr ? expanded : reflected; values[3] = Math.min(fe, fr);
    } else if (fr < values[2]) { v[3] = reflected; values[3] = fr; }
    else {
      const contracted = plus(centroid, scale(minus(fr < values[3] ? reflected : v[3], centroid), 0.5)), fc = g(contracted);
      if (fc < Math.min(fr, values[3])) { v[3] = contracted; values[3] = fc; }
      else for (let i = 1; i < 4; i++) { v[i] = plus(v[0], scale(minus(v[i], v[0]), 0.5)); values[i] = g(v[i]); }
    }
  }
  let best = 0; for (let i = 1; i < 4; i++) if (values[i] < values[best]) best = i;
  return { x: v[best], iterations: it };
}

/** 3×3 matrix as rows, matching upstream float3x3. */
export type Mat3 = [Vec, Vec, Vec];
const matVec = (m: Mat3, v: Vec): Vec => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
const matScale = (m: Mat3, s: number): Mat3 => [scale(m[0], s), scale(m[1], s), scale(m[2], s)];
const matAdd = (a: Mat3, b: Mat3): Mat3 => [plus(a[0], b[0]), plus(a[1], b[1]), plus(a[2], b[2])];
const matSub = (a: Mat3, b: Mat3): Mat3 => [minus(a[0], b[0]), minus(a[1], b[1]), minus(a[2], b[2])];
const transpose = (m: Mat3): Mat3 => [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]];
const outer = (a: Vec, b: Vec): Mat3 => [scale(b, a[0]), scale(b, a[1]), scale(b, a[2])];
export const diagonal = (d: Vec): Mat3 => [[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]];
export const blendEllipsoids = (cached: Mat3, fresh: Mat3): Mat3 => matAdd(matScale(cached, 0.75), matScale(fresh, 0.25));

/** Objective with its analytic gradient (collide.cpp Objective with grad): the softmax blend of
 * the field gradients, minus the repulsion gradient. Counts one query, like the value-only form. */
export function objectiveGradient(a: SdfBody, b: SdfBody, points: { x: Vec }[], epsilon: number, alpha: number, stats: Counters): (x: Vec) => { value: number; gradient: Vec } {
  return x => {
    stats.queries++;
    const A = a.sample(x), B = b.sample(x);
    const root = sqrt(add(mul(sub(A.phi, B.phi), sub(A.phi, B.phi)), epsilon));
    let value = mul(0.5, add(add(A.phi, B.phi), root));
    const t = root > 0 ? div(sub(A.phi, B.phi), root) : 0;
    let gradient = plus(scale(A.gradient, mul(0.5, add(1, t))), scale(B.gradient, mul(0.5, sub(1, t))));
    for (const p of points) {
      const dx = minus(x, p.x), len = length(dx);
      value = sub(value, mul(alpha, len));
      if (len > 1e-9) gradient = minus(gradient, scale(dx, div(alpha, len)));
    }
    return { value, gradient };
  };
}

/** Deep-cut ellipsoid method [LM24] (optimize.cpp ellipsoidMethod). E is updated in place-style and
 * returned for warm starting; the best feasible point is the result. */
export function ellipsoidMethod(g: (x: Vec) => { value: number; gradient: Vec }, seed: Vec, domain: Aabb, ellipsoid: Mat3, tolerance: number, limit: number): { x: Vec; iterations: number; ellipsoid: Mat3 } {
  const n = 3;
  let x: Vec = [...seed], E = ellipsoid, best = project(seed, domain), bestValue = Infinity, it = 0;
  for (; it < limit; it++) {
    let grad: Vec = [0, 0, 0], value = 0, violation = 0;
    for (let i = 0; i < 3; i++) {
      const below = sub(domain.min[i], x[i]), above = sub(x[i], domain.max[i]);
      if (Math.max(below, above) > violation) { violation = Math.max(below, above); grad = [0, 0, 0]; grad[i] = above > below ? 1 : -1; }
    }
    if (violation === 0) {
      const r = g(x); value = r.value; grad = r.gradient;
      if (value < bestValue) { bestValue = value; best = x; }
      if (length(grad) < tolerance) break;
    }
    const Eg = matVec(E, grad), gEg = dot(grad, Eg);
    if (!(gEg > 1e-12)) break;
    const s = sqrt(gEg), a = Math.min(div(violation > 0 ? violation : sub(value, bestValue), s), 0.5);
    const step = scale(Eg, div(add(1, mul(n, a)), mul(n + 1, s)));
    x = minus(x, step);
    const cut = div(mul(2, add(1, mul(n, a))), mul(mul(n + 1, add(1, a)), gEg));
    E = matScale(matSub(E, matScale(outer(Eg, Eg), cut)), div(mul(mul(n, n), sub(1, mul(a, a))), sub(mul(n, n), 1)));
    E = matScale(matAdd(E, transpose(E)), 0.5);
    if (length(step) < tolerance) break;
  }
  return { x: best, iterations: it, ellipsoid: E };
}

/** Stochastic projected gradient descent [RM51] (optimize.cpp gradientDescent): decaying step and
 * a decaying random perturbation drawn from the shared stream. `stepStop` false drops the
 * ‖Δx‖ < τ test (opt-in `sgdGradientStopOnly`); upstream always applies it. */
export function gradientDescent(g: (x: Vec) => { value: number; gradient: Vec }, seed: Vec, domain: Aabb, stepSize: number, tolerance: number, limit: number, rng: Random, stepStop = true): { x: Vec; iterations: number } {
  let x = project(seed, domain), best = x, bestValue = Infinity, it = 0;
  for (; it < limit; it++) {
    const { value, gradient } = g(x);
    if (value < bestValue) { bestValue = value; best = x; }
    const gradLength = length(gradient);
    if (gradLength < tolerance) break;
    const decay = div(1, sqrt(it + 1));
    const noise = scale(rng.onSphere(), mul(mul(0.25, decay), Math.max(gradLength, 1)));
    const next = project(minus(x, scale(plus(gradient, noise), mul(stepSize, decay))), domain);
    const converged = stepStop && length(minus(next, x)) < tolerance;
    x = next;
    if (converged) break;
  }
  return { x: best, iterations: it };
}
