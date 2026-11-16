import assert from 'node:assert/strict';
import { test } from 'node:test';
import { gearProfile, gearMass, TEETH, MODULE, CENTERS, initialAngles, angleDelta } from '../src/sdf-gears/model.ts';

test('gear profiles share pitch, have radial clearance and positive symmetric mass/inertia', () => {
  for (const teeth of TEETH) {
    const profile = gearProfile(teeth), radius = profile.map(p => Math.hypot(...p));
    assert.ok(Math.abs(Math.max(...radius) - MODULE * (teeth / 2 + 1)) < 1e-10);
    assert.ok(Math.abs(Math.min(...radius) - MODULE * (teeth / 2 - 1.25)) < 1e-10);
    const { mass, moments } = gearMass(profile);
    assert.ok(mass > 0 && moments.every(v => v > 0));
    assert.ok(Math.abs(moments[0] - moments[1]) < 1e-10);
    const areaDisk = Math.PI * (MODULE * teeth / 2) ** 2;
    assert.ok(mass / .32 > areaDisk * .85 && mass / .32 < areaDisk * 1.1);
  }
  const phases = initialAngles();
  for (let i = 1; i < TEETH.length; i++) {
    const dx = CENTERS[i][0] - CENTERS[i - 1][0], dy = CENTERS[i][1] - CENTERS[i - 1][1];
    assert.ok(Math.abs(Math.hypot(dx, dy) - MODULE * (TEETH[i] + TEETH[i - 1]) / 2) < 1e-10);
    const a = Math.atan2(dy, dx);
    const sum = TEETH[i - 1] * (a - phases[i - 1]) + TEETH[i] * (a + Math.PI - phases[i]);
    assert.ok(Math.abs(angleDelta(sum, Math.PI)) < 1e-10, 'a tooth faces a gap at each mesh');
  }
});
