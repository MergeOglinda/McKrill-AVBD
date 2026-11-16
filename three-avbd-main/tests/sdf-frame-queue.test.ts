import assert from 'node:assert/strict';
import { test } from 'node:test';
import { FrameQueue } from '../src/sdf-released/frame-queue.ts';

test('playback bounds outstanding frames and releases one slot per completion', async () => {
  const frames = new FrameQueue(), complete: (() => void)[] = [];
  const queue = { onSubmittedWorkDone: () => new Promise<undefined>(resolve => complete.push(() => resolve(undefined))) };
  const fail = () => assert.fail('unexpected queue rejection');
  frames.submitted(queue, fail); frames.submitted(queue, fail);
  assert.equal(frames.available, false);
  assert.throws(() => frames.submitted(queue, fail), /full/);
  complete[0](); await Promise.resolve();
  assert.equal(frames.available, true);
  frames.submitted(queue, fail);
  assert.equal(frames.available, false);
  complete[1](); complete[2](); await Promise.resolve();
  assert.equal(frames.available, true);
});

test('device rejection releases its frame slot and surfaces the error', async () => {
  const frames = new FrameQueue(), error = new Error('Device lost'), errors: unknown[] = [];
  frames.submitted({ onSubmittedWorkDone: () => Promise.reject(error) }, e => errors.push(e));
  await Promise.resolve();
  assert.deepEqual(errors, [error]); assert.equal(frames.available, true);
});


test('deferred playback time survives the per-frame solver cap and pause clears old debt', () => {
  const frames = new FrameQueue(), budget = 1 / 60;
  const first = frames.takeTime(3 / 120, budget);
  const second = frames.takeTime(1 / 120, budget);
  assert.equal(first, budget);
  assert.ok(Math.abs(first + second - 4 / 120) < 1e-12);
  frames.takeTime(.1, budget);
  frames.resetTime();
  assert.equal(frames.takeTime(0, budget), 0);
});
