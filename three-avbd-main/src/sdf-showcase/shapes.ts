import * as THREE from 'three';
import { AnalyticField, bakeVoxelSdfAsync, curvatureSeeds, type Field } from '../lib/sdf.ts';

export type Kind = 'rings' | 'warps' | 'gears';
export interface Shape { geometry: THREE.BufferGeometry; field: Field; color: number }

export async function showcaseShapes(): Promise<Record<Kind, Shape>> {
  const ring = new THREE.TorusGeometry(.72, .22, 16, 48);
  ring.rotateX(Math.PI / 2);
  const ringField = new AnalyticField({ kind: 'torus', majorRadius: .72, minorRadius: .22 });
  const warpField = new AnalyticField({ kind: 'warp', type: 1, halfExtents: [.65, .65, .65], k: .9 });
  const warp = new THREE.BoxGeometry(1.3, 1.3, 1.3, 18, 18, 18);
  const p = warp.getAttribute('position');
  // Inverse of the field's y-dependent rotation and axis permutation.
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), c = Math.cos(.9 * z), s = Math.sin(.9 * z);
    p.setXYZ(i, c * x + s * y, z, -s * x + c * y);
  }
  // The field's axis permutation reverses handedness; keep outward winding.
  const index = warp.index!;
  for (let i = 0; i < index.count; i += 3) {
    const b = index.getX(i + 1); index.setX(i + 1, index.getX(i + 2)); index.setX(i + 2, b);
  }
  warp.computeVertexNormals();
  const outline = new THREE.Shape();
  for (let i = 0; i <= 48; i++) {
    const angle = i * Math.PI / 24, radius = i % 4 < 2 ? .93 : .74;
    const x = radius * Math.cos(angle), y = radius * Math.sin(angle);
    if (i === 0) outline.moveTo(x, y); else outline.lineTo(x, y);
  }
  const hole = new THREE.Path();
  hole.absarc(0, 0, .32, 0, Math.PI * 2, true); outline.holes.push(hole);
  const gear = new THREE.ExtrudeGeometry(outline, { depth: .34, bevelEnabled: false, curveSegments: 20, steps: 1 });
  gear.translate(0, 0, -.17); gear.rotateX(Math.PI / 2);
  const baked = await bakeVoxelSdfAsync(gear, { resolution: 32 });
  if (!baked.watertight) throw new Error(`Gear mesh is not watertight: ${baked.warnings.join(', ')}`);
  for (const field of [ringField, warpField, baked.field]) field.seeds = curvatureSeeds(field, { candidateCount: 500 });
  return { rings: { geometry: ring, field: ringField, color: 0xe3ad60 }, warps: { geometry: warp, field: warpField, color: 0x68c2bc }, gears: { geometry: gear, field: baked.field, color: 0x8e9fe0 } };
}
