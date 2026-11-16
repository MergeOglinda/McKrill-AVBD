import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GpuSolver3D } from '../src/avbd3d/gpu/solver.ts';
import { Solver } from '../src/avbd3d/ref/solver.ts';

test('legacy custom solve modules fail before allocating GPU resources', () => {
  let allocations = 0;
  const device = new Proxy({}, { get() { allocations++; throw new Error('unexpected GPU access'); } }) as GPUDevice;
  assert.throws(() => new GpuSolver3D(device, new Solver(), {
    shaders: { solve: '@compute @workgroup_size(64) fn primal() {}' },
  }), /shaders\.solve must retain primalChunks/);
  assert.equal(allocations, 0);
});
