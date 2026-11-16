import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FrameCapture, frameReport, waitForGpuDrain, captureDeadline } from '../src/sdf-showcase/performance.ts';

test('browser frame metrics use elapsed time, retain long frames and separate simulation speed', () => {
  const sample = frameReport([10, 10, 10, 70], .05);
  assert.equal(sample.fps, 40);
  assert.equal(sample.durationMs, 100);
  assert.equal(sample.medianMs, 10);
  assert.equal(sample.p95Ms, 70);
  assert.equal(sample.over25Ms, 1);
  assert.equal(sample.simulationRate, .5, 'fast rendering cannot conceal slow simulation');
  assert.deepEqual(sample.intervalsMs, [10, 10, 10, 70], 'raw intervals are not clipped or sorted in place');
});

test('browser capture excludes warmup and stops at its finite deadline', () => {
  const capture = new FrameCapture(500);
  assert.equal(capture.record(1400, 9), null);
  assert.equal(capture.record(1500, 9), null, 'first warm frame is the interval baseline');
  for (let t = 1600; t < 5500; t += 100) assert.equal(capture.record(t, .05), null);
  const sample = capture.record(5500, .05)!;
  assert.equal(sample.durationMs, 4000);
  assert.equal(sample.intervalsMs.length, 40);
  assert.ok(Math.abs(sample.simulationRate - .5) < 1e-12);
  assert.throws(() => capture.record(5600, .05), /finished/);
});

test('browser measurements reject missing, nonfinite or nonmonotonic samples', () => {
  for (const intervals of [[], [0], [-1], [NaN], [Infinity]]) assert.throws(() => frameReport(intervals, 0), /Invalid/);
  assert.throws(() => frameReport([16], -1), /Invalid/);
  assert.throws(() => new FrameCapture(NaN), /Invalid/);
  const capture = new FrameCapture(0);
  capture.record(1000, 0);
  assert.throws(() => capture.record(1000, 0), /increase/);
  assert.throws(() => capture.record(1200, NaN), /Invalid/);
});

test('GPU drain handles completion and device errors without leaving a deadline behind', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  await waitForGpuDrain({ onSubmittedWorkDone: () => Promise.resolve(undefined) });
  const error = new Error('Device lost');
  await assert.rejects(waitForGpuDrain({ onSubmittedWorkDone: () => Promise.reject(error) }), error);
  t.mock.timers.tick(5000);
});

test('GPU drain times out a stalled queue and a subsequent drain can succeed', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const pending = waitForGpuDrain({ onSubmittedWorkDone: () => new Promise<undefined>(() => {}) });
  const failed = assert.rejects(pending, /did not finish within 5000 ms/);
  t.mock.timers.tick(5000);
  await failed;
  await waitForGpuDrain({ onSubmittedWorkDone: () => Promise.resolve(undefined) });
});

test('sustained capture measures a full 30 seconds after the first recorded frame', () => {
  const capture = new FrameCapture(0, 0, 30000);
  assert.equal(capture.record(12, 0), null);
  assert.equal(capture.record(30000, 29.988), null, 'start-up offset cannot shorten the trial');
  assert.equal(capture.record(30012, .012)?.durationMs, 30000);
  assert.throws(() => new FrameCapture(0, 0, 30001), /duration/);
});


test('the finalization deadline covers stalled timestamp maps and inspection after a successful drain', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const pending = captureDeadline(new Promise<never>(() => {}));
  const failed = assert.rejects(pending, /finalization exceeded/);
  t.mock.timers.tick(5000); await failed;
  assert.equal(await captureDeadline(Promise.resolve('retry')), 'retry');
});


test('a queued pre-click animation frame does not cancel or shorten a capture', () => {
  const capture = new FrameCapture(505, 0, 30000);
  assert.equal(capture.record(500, .01), null);
  assert.equal(capture.snapshot(), null);
  assert.equal(capture.record(510, .01), null);
  assert.throws(() => capture.record(500, .01), /increase/);
  assert.equal(capture.record(30500, 29.99), null);
  const sample = capture.record(30510, .01)!;
  assert.equal(sample.durationMs, 30000);
  assert.equal(sample.simulationSeconds, 30);
  assert.throws(() => new FrameCapture(505).record(-1, 0), /Invalid/);
});
