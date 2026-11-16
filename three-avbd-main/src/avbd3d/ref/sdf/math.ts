/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// sdf-dcd-demo maths.h. Keep float32 operations separate from the AVBD f64 oracle.
export type Vec = [number, number, number];
export type Rotation = [number, number, number, number];
export interface Aabb { min: Vec; max: Vec }
export const f = Math.fround;
export const add = (a: number, b: number) => f(f(a) + f(b));
export const sub = (a: number, b: number) => f(f(a) - f(b));
export const mul = (a: number, b: number) => f(f(a) * f(b));
export const div = (a: number, b: number) => f(f(a) / f(b));
export const sqrt = (a: number) => f(Math.sqrt(a));
export const sin = (a: number) => f(Math.sin(a));
export const cos = (a: number) => f(Math.cos(a));
export const sign = (a: number) => a < 0 ? -1 : 1;
export const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
export const vec = (a: ArrayLike<number>): Vec => [f(a[0]), f(a[1]), f(a[2])];
export const plus = (a: Vec, b: Vec): Vec => [add(a[0], b[0]), add(a[1], b[1]), add(a[2], b[2])];
export const minus = (a: Vec, b: Vec): Vec => [sub(a[0], b[0]), sub(a[1], b[1]), sub(a[2], b[2])];
export const scale = (a: Vec, s: number): Vec => [mul(a[0], s), mul(a[1], s), mul(a[2], s)];
export const divide = (a: Vec, s: number): Vec => [div(a[0], s), div(a[1], s), div(a[2], s)];
export const dot = (a: Vec, b: Vec) => add(add(mul(a[0], b[0]), mul(a[1], b[1])), mul(a[2], b[2]));
export const length = (a: Vec) => sqrt(dot(a, a));
export const normalize = (a: Vec): Vec => length(a) > 1e-12 ? divide(a, length(a)) : [0, 1, 0];
export const cross = (a: Vec, b: Vec): Vec => [sub(mul(a[1], b[2]), mul(a[2], b[1])), sub(mul(a[2], b[0]), mul(a[0], b[2])), sub(mul(a[0], b[1]), mul(a[1], b[0]))];
export const vmin = (a: Vec, b: Vec): Vec => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
export const vmax = (a: Vec, b: Vec): Vec => [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
export const size = (b: Aabb) => minus(b.max, b.min);
export const expand = (b: Aabb, r: number): Aabb => ({ min: minus(b.min, [r, r, r]), max: plus(b.max, [r, r, r]) });
export const intersection = (a: Aabb, b: Aabb): Aabb => ({ min: vmax(a.min, b.min), max: vmin(a.max, b.max) });
export const valid = (b: Aabb) => b.min.every((x, i) => x <= b.max[i]);
export const contains = (b: Aabb, p: Vec) => p.every((x, i) => x >= b.min[i] && x <= b.max[i]);
export const project = (p: Vec, b: Aabb) => vmin(vmax(p, b.min), b.max);
export function corners(b: Aabb): Vec[] {
  return Array.from({ length: 8 }, (_, i) => [i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]]);
}
export function bounds(points: Vec[]): Aabb {
  return points.reduce((b, p) => ({ min: vmin(b.min, p), max: vmax(b.max, p) }), { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] } as Aabb);
}
export function rotate(q: Rotation, v: Vec): Vec {
  const u: Vec = [q[0], q[1], q[2]];
  const t = scale(cross(u, v), 2);
  return plus(plus(v, scale(t, q[3])), cross(u, t));
}
export const inverse = (q: Rotation): Rotation => [-q[0], -q[1], -q[2], q[3]];
export class Random {
  state = 0x853c49e6748fea9bn;
  next(): number {
    const old = this.state;
    this.state = BigInt.asUintN(64, old * 6364136223846793005n + 1442695040888963407n);
    const x = Number(BigInt.asUintN(32, ((old >> 18n) ^ old) >> 27n)), r = Number(old >> 59n);
    return ((x >>> r) | (x << ((-r) & 31))) >>> 0;
  }
  uniform(): number { return (this.next() >>> 8) / 16777216; }
  /** uniform(a, b): a + (b - a)·uniform(). */
  range(a: number, b: number): number { return add(a, mul(sub(b, a), this.uniform())); }
  /** Uniform direction (maths.h onSphere): z, then the angle, are drawn in that order. */
  onSphere(): Vec {
    const z = this.range(-1, 1), a = this.range(0, mul(2, f(Math.PI))), r = sqrt(Math.max(0, sub(1, mul(z, z))));
    return [mul(r, cos(a)), mul(r, sin(a)), z];
  }
  integer(n: number): number { return this.next() % n; }
  inBox(b: Aabb): Vec {
    const s = size(b);
    return plus(b.min, [mul(s[0], this.uniform()), mul(s[1], this.uniform()), mul(s[2], this.uniform())]);
  }
}
