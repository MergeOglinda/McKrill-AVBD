import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { Triangle, Vector3 } from 'three';
import { bakeVoxelSdf } from '../src/lib/sdf.ts';
const floats = (folder: string, name: string) => {
  const raw = gunzipSync(readFileSync(new URL(`./fixtures/${folder}/${name}.bin.gz`, import.meta.url)));
  const manifest = JSON.parse(readFileSync(new URL(`./fixtures/${folder}/manifest.json`, import.meta.url), 'utf8'));
  const entry = (manifest.grids ?? manifest.surfaces)[name];
  assert.equal(createHash('sha256').update(raw).digest('hex'), entry.sha256);
  const values = new Float32Array(raw.length / 4);
  for (let i = 0; i < values.length; i++) values[i] = raw.readFloatLE(4 * i);
  return values;
};
for (const name of ['bunny', 'gear', 'cow', 'chair']) test(`direct CPU mesh bake agrees with C++ distances/signs: ${name}`, () => {
  const surface = floats('oracle-sdf-surfaces', name), expected = floats('oracle-sdf-coarse-grids', name);
  const { field } = bakeVoxelSdf({ positions: surface }, { resolution: 16 });
  assert.equal(field.grid.length, expected.length);
  const triangles: Triangle[] = [];
  for (let t = 0; t < surface.length; t += 9) triangles.push(new Triangle(
    new Vector3().fromArray(surface, t), new Vector3().fromArray(surface, t + 3), new Vector3().fromArray(surface, t + 6)));
  const [nx, ny] = field.dimensions;
  const closest = new Vector3();
  for (let i = 0; i < expected.length; i += 4) {
    const phi = field.grid[i];
    assert.ok(Math.abs(phi - expected[i]) <= 2e-6, `${name} distance at ${i / 4}`);
    if (Math.abs(expected[i]) > 2e-6) assert.equal(Math.sign(phi), Math.sign(expected[i]));
    const gradient = Array.from(field.grid.slice(i + 1, i + 4));
    assert.ok(Math.abs(Math.hypot(...gradient) - 1) < 2e-5, `${name} normal length`);
    if (Math.hypot(...gradient.map((v, j) => v - expected[i + 1 + j])) <= .001) continue;
    // At medial-axis ties gradient is not unique. Independently verify both
    // implied closest points land on a source triangle at the same distance.
    const node = i / 4, x = node % nx, y = Math.floor(node / nx) % ny, z = Math.floor(node / (nx * ny));
    const p = new Vector3(field.origin[0] + x * field.cell, field.origin[1] + y * field.cell, field.origin[2] + z * field.cell);
    for (const data of [expected, field.grid]) {
      const q = p.clone().addScaledVector(new Vector3(data[i+1], data[i+2], data[i+3]), -data[i]);
      let distance = Infinity;
      for (const triangle of triangles) distance = Math.min(distance, triangle.closestPointToPoint(q, closest).distanceTo(q));
      assert.ok(distance < 3e-6, `${name} differing gradient does not point to surface: ${distance}`);
    }
  }
});
