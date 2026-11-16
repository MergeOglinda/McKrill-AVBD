import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Vector3 } from 'three';
import { showcaseShapes } from '../src/sdf-showcase/shapes.ts';

test('showcase render surfaces match analytic fields and the baked gear retains its hole', async () => {
  const shapes = await showcaseShapes();
  try {
    for (const kind of ['rings', 'warps'] as const) {
      const { geometry, field } = shapes[kind], p = geometry.getAttribute('position');
      for (let i = 0; i < p.count; i++) {
        assert.ok(Math.abs(field.sample([p.getX(i), p.getY(i), p.getZ(i)]).phi) < 1e-5, `${kind} vertex ${i}`);
      }
      // At a sharp edge the SDF selects either adjoining face gradient. Check
      // every triangle interior instead, where its outward normal is unique.
      const index = geometry.index!;
      for (let i = 0; i < index.count; i += 3) {
        const a = new Vector3().fromBufferAttribute(p, index.getX(i));
        const b = new Vector3().fromBufferAttribute(p, index.getX(i + 1));
        const c = new Vector3().fromBufferAttribute(p, index.getX(i + 2));
        const center = a.clone().add(b).add(c).multiplyScalar(1 / 3);
        const normal = b.clone().sub(a).cross(c.clone().sub(a)).normalize();
        const gradient = new Vector3(...field.sample(center.toArray()).gradient).normalize();
        assert.ok(normal.dot(gradient) > .99, `${kind} outward winding`);
      }
    }
    assert.ok(shapes.gears.field.sample([0, 0, 0]).phi > .2, 'baked through-hole');
    assert.ok(shapes.gears.field.sample([.55, 0, 0]).phi < -.1, 'solid gear rim');
    assert.ok(shapes.gears.field.sample([0, .5, 0]).phi > 0, 'outside');
    for (const { field } of Object.values(shapes)) assert.ok(field.seeds.length > 0);
  } finally { for (const shape of Object.values(shapes)) shape.geometry.dispose(); }
});
