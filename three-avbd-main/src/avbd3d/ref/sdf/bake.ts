/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// CPU mesh preparation. No renderer, device, or Three.js runtime dependency.
import type { BufferGeometry } from 'three';
import { bounds, cross, dot, minus, plus, scale, normalize, length, type Vec, type Aabb } from './math.ts';
import { VoxelField } from './voxel.ts';

export interface TriangleMesh { positions: ArrayLike<number>; indices?: ArrayLike<number> }
export interface BakeOptions {
  /** Cells along the longest bounding-box axis. Default 32; integer 4–256. Upstream
   * sdf-dcd-demo always bakes at 100 (see SDF_BAKING.md, "Differences from upstream"). */
  resolution?: number;
  /** Normalize longest axis to one; default false (preserve geometry units). Upstream
   * always normalizes; pass `{ resolution: 100, normalize: true }` to match its grids. */
  normalize?: boolean;
  /** Grid allocation ceiling, excluding input/BVH. Default 64 MiB. */
  maxGridBytes?: number;
  signal?: AbortSignal;
  onProgress?: (fraction: number) => void;
}
export interface BakedSdf {
  field: VoxelField;
  /** bakedPoint = (originalPoint - center) * scale */
  transform: { center: Vec; scale: number };
  watertight: boolean;
  warnings: string[];
}
type Triangle = [Vec, Vec, Vec];
interface Node { box: Aabb; triangles?: Triangle[]; left?: Node; right?: Node }
function tree(triangles: Triangle[]): Node {
  const box = bounds(triangles.flat());
  if (triangles.length <= 8) return { box, triangles };
  const extent = minus(box.max, box.min), axis = extent.indexOf(Math.max(...extent));
  triangles.sort((a, b) => a.reduce((s, p) => s + p[axis], 0) - b.reduce((s, p) => s + p[axis], 0));
  const mid = triangles.length >> 1;
  return { box, left: tree(triangles.slice(0, mid)), right: tree(triangles.slice(mid)) };
}
// Closest point on triangle, Ericson's vertex/edge/face Voronoi regions.
function closest(p: Vec, [a, b, c]: Triangle): Vec {
  const ab = minus(b, a), ac = minus(c, a), ap = minus(p, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const bp = minus(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return plus(a, scale(ab, d1 / (d1 - d3)));
  const cp = minus(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return plus(a, scale(ac, d2 / (d2 - d6)));
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return plus(b, scale(minus(c, b), (d4 - d3) / (d4 - d3 + d5 - d6)));
  return plus(plus(a, scale(ab, vb / (va + vb + vc))), scale(ac, vc / (va + vb + vc)));
}
function distanceSq(p: Vec, b: Aabb): number {
  return p.reduce((s, v, i) => s + Math.max(b.min[i] - v, 0, v - b.max[i]) ** 2, 0);
}
function nearest(root: Node, p: Vec): { point: Vec; normal: Vec } {
  let best = Infinity, point: Vec = p, normal: Vec = [0, 1, 0];
  const stack = [root];
  while (stack.length) {
    const n = stack.pop()!;
    if (distanceSq(p, n.box) > best) continue;
    if (n.triangles) for (const t of n.triangles) {
      const q = closest(p, t), d = minus(p, q), sq = dot(d, d);
      if (sq < best) { best = sq; point = q; normal = normalize(cross(minus(t[1], t[0]), minus(t[2], t[0]))); }
    }
    else {
      const l = n.left!, r = n.right!;
      if (distanceSq(p, l.box) < distanceSq(p, r.box)) stack.push(r, l); else stack.push(l, r);
    }
  }
  return { point, normal };
}
function abort(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException('SDF bake aborted', 'AbortError');
}
function inputMesh(input: TriangleMesh | BufferGeometry): TriangleMesh {
  if ('getAttribute' in input) {
    const attr = input.getAttribute('position');
    if (!attr || attr.itemSize < 3) throw new Error('Geometry needs a position attribute');
    const positions = new Float32Array(attr.count * 3);
    for (let i = 0; i < attr.count; i++) positions.set([attr.getX(i), attr.getY(i), attr.getZ(i)], i * 3);
    const index = input.getIndex();
    return { positions, indices: index ? Array.from({ length: index.count }, (_, i) => index.getX(i)) : undefined };
  }
  return input;
}
/** Cooperative CPU bake; each yielded slice is a cancellation/progress boundary. */
function* bake(input: TriangleMesh | BufferGeometry, options: BakeOptions): Generator<number, BakedSdf> {
  abort(options.signal);
  const resolution = options.resolution ?? 32, limit = options.maxGridBytes ?? 64 * 1024 * 1024;
  if (!Number.isInteger(resolution) || resolution < 4 || resolution > 256) throw new Error('resolution must be an integer from 4 to 256');
  if (!Number.isFinite(limit) || limit <= 0) throw new Error('maxGridBytes must be positive and finite');
  const mesh = inputMesh(input), count = mesh.positions.length / 3;
  if (!Number.isInteger(count) || count < 3) throw new Error('positions must contain complete vertices');
  const points: Vec[] = [];
  for (let i = 0; i < count; i++) {
    const p: Vec = [mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]];
    if (!p.every(v => Number.isFinite(v) && Number.isFinite(Math.fround(v)))) throw new Error('positions must be finite float32 values');
    points.push(p);
  }
  const original = bounds(points), extent = minus(original.max, original.min), longest = Math.max(...extent);
  if (!(longest > 0) || !Number.isFinite(longest)) throw new Error('Mesh must have finite nonzero extent');
  const center: Vec = options.normalize ? scale(plus(original.min, original.max), 0.5) : [0, 0, 0];
  const factor = options.normalize ? 1 / longest : 1;
  const vertices = points.map(p => scale(minus(p, center), factor));
  const box = bounds(vertices), cell = Math.fround(Math.max(...minus(box.max, box.min)) / resolution);
  if (!(cell > 0) || !vertices.every(p => p.every(Number.isFinite))) throw new Error('Mesh scale exceeds float32 range');
  const origin = minus(box.min, [3 * cell, 3 * cell, 3 * cell]);
  const dims = minus(box.max, box.min).map(v => Math.ceil(v / cell) + 7) as Vec;
  if (dims[0] * dims[1] * dims[2] * 16 > limit) throw new Error('SDF grid exceeds maxGridBytes');
  const indices = mesh.indices ?? Array.from({ length: count }, (_, i) => i);
  if (indices.length < 3 || indices.length % 3) throw new Error('indices must contain complete triangles');
  const welded = new Map<string, number>(), ids = vertices.map(p => {
    const key = p.join(','); if (!welded.has(key)) welded.set(key, welded.size); return welded.get(key)!;
  });
  const edges = new Map<string, { count: number; winding: number }>(), triangles: Triangle[] = [];
  for (let i = 0; i < indices.length; i += 3) {
    const idx = [indices[i], indices[i + 1], indices[i + 2]];
    if (!idx.every(v => Number.isInteger(v) && v >= 0 && v < count)) throw new Error('Triangle index out of range');
    const t = idx.map(j => vertices[j]) as Triangle;
    if (length(cross(minus(t[1], t[0]), minus(t[2], t[0]))) <= 0) throw new Error('Degenerate triangle');
    triangles.push(t);
    for (let j = 0; j < 3; j++) {
      const a = ids[idx[j]], b = ids[idx[(j + 1) % 3]], key = `${Math.min(a, b)},${Math.max(a, b)}`;
      const e = edges.get(key) ?? { count: 0, winding: 0 }; e.count++; e.winding += a < b ? 1 : -1; edges.set(key, e);
    }
  }
  const watertight = [...edges.values()].every(e => e.count === 2 && e.winding === 0);
  const signedVolume = triangles.reduce((s, t) => s + dot(t[0], cross(t[1], t[2])) / 6, 0);
  if (signedVolume < 0) for (const t of triangles) [t[1], t[2]] = [t[2], t[1]];
  // Retain the source corner order for oracle-compatible area sampling, before
  // canonical distance-query ordering or BVH sorting changes the triangle list.
  const surfaceTriangles = new Float32Array(triangles.length * 9);
  triangles.forEach((t, i) => t.forEach((v, j) => surfaceTriangles.set(v, i * 9 + j * 3)));
  // Canonical cyclic order makes floating-point distance queries independent of
  // which corner became first when a consistently reversed mesh was corrected.
  for (const t of triangles) {
    let first = 0;
    const compare = (a: Vec, b: Vec) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
    for (let i = 1; i < 3; i++) if (compare(t[i], t[first]) < 0) first = i;
    if (first === 1) t.push(t.shift()!);
    if (first === 2) t.unshift(t.pop()!);
  }
  const votes = new Uint8Array(dims[0] * dims[1] * dims[2]);
  // Three-axis winding votes, matching the oracle's jittered scanline convention.
  for (let a = 0; a < 3; a++) {
    const b = (a + 1) % 3, c = (a + 2) % 3;
    for (let k = 0; k < dims[c]; k++) {
      abort(options.signal);
      const rows: [number, number][][] = Array.from({ length: dims[b] }, () => []);
      for (const [v0, v1, v2] of triangles) {
        const db1 = v1[b] - v0[b], dc1 = v1[c] - v0[c], db2 = v2[b] - v0[b], dc2 = v2[c] - v0[c];
        const det = db1 * dc2 - db2 * dc1;
        if (Math.abs(det) < cell * cell * 1e-12) continue;
        const pc = origin[c] + (k + 0.3456789) * cell - v0[c];
        const lo = Math.max(0, Math.ceil((Math.min(v0[b], v1[b], v2[b]) - origin[b]) / cell - 0.1234567));
        const hi = Math.min(dims[b] - 1, Math.floor((Math.max(v0[b], v1[b], v2[b]) - origin[b]) / cell - 0.1234567));
        for (let j = lo; j <= hi; j++) {
          const pb = origin[b] + (j + 0.1234567) * cell - v0[b];
          const w1 = (pb * dc2 - db2 * pc) / det, w2 = (db1 * pc - pb * dc1) / det;
          if (w1 >= 0 && w2 >= 0 && w1 + w2 <= 1) rows[j].push([v0[a] + w1 * (v1[a] - v0[a]) + w2 * (v2[a] - v0[a]), det < 0 ? 1 : -1]);
        }
      }
      for (let j = 0; j < dims[b]; j++) {
        const row = rows[j].sort((x, y) => x[0] - y[0] || x[1] - y[1]); let w = 0, r = 0;
        for (let i = 0; i < dims[a]; i++) {
          while (r < row.length && row[r][0] < origin[a] + i * cell) w += row[r++][1];
          if (w > 0) { const p = [0, 0, 0]; p[a] = i; p[b] = j; p[c] = k; votes[p[0] + dims[0] * (p[1] + dims[1] * p[2])]++; }
        }
      }
      yield (a + (k + 1) / dims[c]) / 6;
    }
  }
  const root = tree(triangles), grid = new Float32Array(votes.length * 4);
  for (let z = 0; z < dims[2]; z++) {
    abort(options.signal);
    for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
      const p = plus(origin, scale([x, y, z], cell)), hit = nearest(root, p), delta = minus(p, hit.point), d = length(delta);
      const i = x + dims[0] * (y + dims[1] * z), sign = votes[i] >= 2 ? -1 : 1;
      const g = d > cell * 1e-6 ? scale(delta, sign / d) : hit.normal;
      grid.set([sign * d, ...g], i * 4);
    }
    yield 0.5 + 0.5 * (z + 1) / dims[2];
  }
  return { field: new VoxelField(dims, origin, cell, grid, box, surfaceTriangles), transform: { center, scale: factor }, watertight,
    warnings: watertight ? [] : ['Mesh has open, non-manifold, or inconsistently oriented edges; interior signs may be unreliable.'] };
}
/** Synchronous CPU bake, intended for offline tools or a dedicated worker. */
export function bakeVoxelSdf(input: TriangleMesh | BufferGeometry, options: BakeOptions = {}): BakedSdf {
  const work = bake(input, options);
  for (;;) { const step = work.next(); if (step.done) return step.value; options.onProgress?.(step.value); abort(options.signal); }
}
/** Yields to the event loop between slices; use inside a Worker for expensive meshes. */
export async function bakeVoxelSdfAsync(input: TriangleMesh | BufferGeometry, options: BakeOptions = {}): Promise<BakedSdf> {
  const work = bake(input, options);
  for (;;) {
    const step = work.next(); if (step.done) return step.value;
    options.onProgress?.(step.value); await new Promise<void>(resolve => setTimeout(resolve, 0)); abort(options.signal);
  }
}
