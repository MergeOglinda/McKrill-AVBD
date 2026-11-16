// Render/probe surfaces for released analytic shapes: marching tetrahedra on the field itself,
// so warped boxes, spike plates and wheels need no hand-built inverse mapping.
import * as THREE from 'three';
import type { Field } from '../lib/sdf.ts';

// Six tetrahedra around the cube diagonal 0-7 (corner bit 1: +x, 2: +y, 4: +z).
const TETS = [[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]];

/** Zero level set of `field` sampled with `cells` cells along its longest bound; phi = 0 counts as
 * outside. Surface beyond `field.bounds` (plus about a cell of padding) is not meshed. */
export function fieldSurface(field: Field, cells = 72): THREE.BufferGeometry {
  const { min, max } = field.bounds;
  const h = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / cells;
  // Padding keeps every closed surface inside the grid; the fractional offset keeps axis-aligned
  // faces (box tops, hub sides) off grid planes, where phi = 0 exactly would collapse crossings.
  const lo = min.map(v => v - 1.37 * h), n = min.map((v, k) => Math.ceil((max[k] - v) / h) + 4);
  const at = (i: number, j: number, k: number) => i + n[0] * (j + n[1] * k);
  const point = (index: number): [number, number, number] => {
    const i = index % n[0], j = Math.floor(index / n[0]) % n[1], k = Math.floor(index / (n[0] * n[1]));
    return [lo[0] + i * h, lo[1] + j * h, lo[2] + k * h];
  };
  const phi = new Float32Array(n[0] * n[1] * n[2]);
  for (let index = 0; index < phi.length; index++) phi[index] = field.sample(point(index)).phi;
  const positions: number[] = [], indices: number[] = [], edges = new Map<number, number>();
  // Shared crossing vertex per grid edge (a < b), so the surface is indexed and welded.
  const cross = (a: number, b: number) => {
    if (a > b) [a, b] = [b, a];
    const key = a * phi.length + b;
    let v = edges.get(key);
    if (v === undefined) {
      const pa = point(a), pb = point(b), t = phi[a] / (phi[a] - phi[b]);
      v = positions.length / 3; positions.push(...pa.map((x, k) => x + t * (pb[k] - x))); edges.set(key, v);
    }
    return v;
  };
  const centroid = (list: number[]) => list.map(point).reduce<number[]>((s, p) => s.map((x, k) => x + p[k] / list.length), [0, 0, 0]);
  // Emit a triangle wound so its normal points from the inside corners to the outside ones.
  const triangle = (a: number, b: number, c: number, inside: number[], outside: number[]) => {
    const p = [a, b, c].map(v => new THREE.Vector3().fromArray(positions, v * 3));
    const normal = new THREE.Vector3().subVectors(p[1], p[0]).cross(new THREE.Vector3().subVectors(p[2], p[0]));
    // Crossings that land on a grid point (phi exactly 0) collapse; drop the zero-area sliver.
    if (normal.lengthSq() < 1e-12 * h ** 4) return;
    const out = new THREE.Vector3().fromArray(centroid(outside)).sub(new THREE.Vector3().fromArray(centroid(inside)));
    if (normal.dot(out) < 0) indices.push(a, c, b); else indices.push(a, b, c);
  };
  for (let k = 0; k < n[2] - 1; k++) for (let j = 0; j < n[1] - 1; j++) for (let i = 0; i < n[0] - 1; i++) {
    const corner = (c: number) => at(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
    for (const tet of TETS) {
      const v = tet.map(corner), inside = v.filter(x => phi[x] < 0), outside = v.filter(x => phi[x] >= 0);
      if (inside.length === 0 || inside.length === 4) continue;
      if (inside.length === 2) {
        const [a, b] = inside, [c, d] = outside, q = [cross(a, c), cross(a, d), cross(b, d), cross(b, c)];
        triangle(q[0], q[1], q[2], inside, outside); triangle(q[0], q[2], q[3], inside, outside);
      } else {
        const [single] = inside.length === 1 ? inside : outside, rest = v.filter(x => x !== single);
        triangle(cross(single, rest[0]), cross(single, rest[1]), cross(single, rest[2]), inside, outside);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  return geometry;
}
