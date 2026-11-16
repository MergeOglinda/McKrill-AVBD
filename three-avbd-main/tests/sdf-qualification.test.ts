import { test } from 'node:test';
import assert from 'node:assert/strict';
import { qualificationOptions, distribution, pacingAcceptance, qualificationSampleDelay, gpuBudgetPassed, validWorkingGpuSample } from '../src/sdf-showcase/qualification.ts';
import { frameReport } from '../src/sdf-showcase/performance.ts';

test('qualification is opt in, bounded, and has a timestamp-disabled fallback', () => {
  assert.equal(qualificationOptions(new URLSearchParams('count=512')), null);
  assert.deepEqual(qualificationOptions(new URLSearchParams('qualify=1&count=128&timestamps=0')), { count:128, timestamps:false });
  for (const count of ['NaN', '1024', '-1', '3.5']) assert.throws(() => qualificationOptions(new URLSearchParams(`qualify=1&count=${count}`)), /count/);
});
test('qualification retains slow tails and never approves incomplete, slow-simulation or undrained captures', () => {
  const sample = frameReport(Array(3000).fill(10), 30);
  assert.equal(pacingAcceptance(sample, 1), true);
  assert.equal(pacingAcceptance({ ...sample, durationMs: 29999 }, 1), false);
  assert.equal(pacingAcceptance({ ...sample, simulationRate:.9 }, 1), false);
  assert.equal(pacingAcceptance(sample, 40), false);
  const spike = frameReport([...Array(2990).fill(10), 110], 30);
  assert.equal(pacingAcceptance(spike, 1), false);
  assert.deepEqual(distribution([1, 2, 100]), { count:3, p50:2, p95:100, worst:100 });
  assert.equal(distribution([]), null);
});


test('sparse sampling covers both zero/one-step phases on a 120Hz display with 60Hz physics', () => {
  let previous = -Infinity, attempt = 0;
  const phases = [0, 0];
  for (let frame = 1; frame <= 3600; frame++) {
    const time = frame * 1000 / 120;
    if (time - previous < qualificationSampleDelay(attempt)) continue;
    previous = time; attempt++; phases[frame % 2]++;
  }
  assert.ok(phases.every(count => count >= 60), `Both phases need >=60 samples, got ${phases}`);
});


test('zero or unavailable timestamps cannot certify GPU performance', () => {
  assert.equal(gpuBudgetPassed(0, 0, 100, 0), false);
  assert.equal(gpuBudgetPassed(undefined, undefined, 0, 0), false);
  assert.equal(gpuBudgetPassed(1, 3, 100, 0), true);
  assert.equal(gpuBudgetPassed(1, 3, 100, 1), false);
  assert.equal(gpuBudgetPassed(1, 3, 59, 0), false);
});


test('every working timestamp must be valid, while quantized zero rendering is allowed', () => {
  assert.equal(validWorkingGpuSample({ physicsMs: 2, renderMs: 0, totalMs: 2 }), true);
  assert.equal(validWorkingGpuSample({ physicsMs: 2, renderMs: .1, totalMs: 2.1 }), true);
  for (const sample of [
    { physicsMs: 0, renderMs: 1, totalMs: 1 },
    { physicsMs: 2, renderMs: 0, totalMs: 0 },
    { physicsMs: 2, renderMs: 1, totalMs: 2 },
    { physicsMs: 2, renderMs: -1, totalMs: 1 },
    { physicsMs: NaN, renderMs: 1, totalMs: 2 },
  ]) assert.equal(validWorkingGpuSample(sample), false);
  assert.equal(gpuBudgetPassed(2, 3, 100, 1), false, 'One invalid sample must invalidate an otherwise passing distribution');
});
