/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// Upstream voxel.cpp interpolation. See bake.ts for CPU mesh preparation.
import { f, add, sub, mul, vec, plus, minus, scale, divide, length, normalize, project, type Vec, type Aabb } from './math.ts';
import type { Field, FieldSample } from './shapes.ts';
export class VoxelField implements Field {
  readonly convex = false;
  centerOfMass: Vec = [0, 0, 0];
  seeds: Vec[] = [];
  readonly bounds: Aabb;
  readonly dimensions: Vec;
  readonly origin: Vec;
  readonly cell: number;
  /** Four floats per node: phi, gradient x/y/z. X varies fastest. */
  readonly grid: Float32Array;
  readonly surfaceTriangles?: Float32Array;
  constructor(dimensions: Vec, origin: Vec, cell: number, grid: Float32Array, bounds: Aabb, surfaceTriangles?: Float32Array) {
    if (!dimensions.every(n => Number.isSafeInteger(n) && n >= 2) || !Number.isFinite(f(cell)) || f(cell) <= 0 || !origin.every(v => Number.isFinite(f(v)))) throw new Error('Invalid SDF grid geometry');
    if (![...bounds.min, ...bounds.max].every(v => Number.isFinite(f(v))) || bounds.min.some((v, i) => v > bounds.max[i])) throw new Error('Invalid SDF bounds');
    if (!grid.every(Number.isFinite)) throw new Error('SDF grid must be finite');
    if (!dimensions.every((n, i) => Number.isFinite(f(f(origin[i]) + f((n - 1) * f(cell)))))) throw new Error('SDF grid exceeds float32 range');
    if (grid.length !== dimensions[0] * dimensions[1] * dimensions[2] * 4) throw new Error('SDF grid length mismatch');
    if (surfaceTriangles && (!surfaceTriangles.length || surfaceTriangles.length % 9 || !surfaceTriangles.every(Number.isFinite))) throw new Error('Invalid SDF surface triangles');
    this.surfaceTriangles = surfaceTriangles;
    this.dimensions = [...dimensions]; this.origin = vec(origin); this.cell = f(cell); this.grid = grid;
    this.bounds = { min: vec(bounds.min), max: vec(bounds.max) };
  }
  sample(input: Vec): FieldSample {
    const p = vec(input), [nx, ny, nz] = this.dimensions;
    const hi = plus(this.origin, scale([nx - 1, ny - 1, nz - 1], this.cell));
    const q = project(p, { min: this.origin, max: hi }), frac = divide(minus(q, this.origin), this.cell);
    const x = Math.min(Math.trunc(frac[0]), nx - 2), y = Math.min(Math.trunc(frac[1]), ny - 2), z = Math.min(Math.trunc(frac[2]), nz - 2);
    const t: Vec = [sub(frac[0], x), sub(frac[1], y), sub(frac[2], z)];
    let phi = 0, gradient: Vec = [0, 0, 0];
    for (let i = 0; i < 8; i++) {
      const w = mul(mul(i & 1 ? t[0] : sub(1, t[0]), i & 2 ? t[1] : sub(1, t[1])), i & 4 ? t[2] : sub(1, t[2]));
      const j = 4 * (x + (i & 1) + nx * (y + ((i >> 1) & 1) + ny * (z + ((i >> 2) & 1))));
      phi = add(phi, mul(w, this.grid[j]));
      gradient = plus(gradient, scale([this.grid[j + 1], this.grid[j + 2], this.grid[j + 3]], w));
    }
    const outside = minus(p, q), len = length(outside);
    return { phi: add(phi, len), gradient: len > 0 ? normalize(plus(gradient, divide(outside, len))) : gradient };
  }
}
