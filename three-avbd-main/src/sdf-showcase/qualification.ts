import type { World } from '../lib/world.ts';
import type { FrameReport } from './performance.ts';

/** Only explicit qualification URLs alter the capture duration or scene capacity. */
export function qualificationOptions(query: URLSearchParams) {
  if (query.get('qualify') !== '1') return null;
  const count = Number(query.get('count') ?? 12);
  if (![12, 36, 128, 512].includes(count)) throw new Error('Qualification count must be 12, 36, 128 or 512');
  return { count, timestamps: query.get('timestamps') !== '0' };
}
export function distribution(values: readonly number[]) {
  if (!values.length) return null;
  if (values.some(value => !Number.isFinite(value) || value < 0)) throw new Error('Invalid qualification sample');
  const sorted = [...values].sort((a, b) => a - b);
  return { count: values.length, p50: sorted[Math.ceil(sorted.length * .5) - 1],
    p95: sorted[Math.ceil(sorted.length * .95) - 1], worst: sorted[sorted.length - 1] };
}
export function qualificationSampleDelay(attempt: number) { return 113 + (attempt % 7) * 13; }

/** Timestamp quantization can legitimately round a short render segment to zero. */
export function validWorkingGpuSample(sample: { physicsMs: number; renderMs: number; totalMs: number }) {
  const { physicsMs, renderMs, totalMs } = sample;
  return [physicsMs, renderMs, totalMs].every(Number.isFinite) && physicsMs > 0 && renderMs >= 0
    && totalMs > 0 && totalMs >= physicsMs && Math.abs(totalMs - physicsMs - renderMs) <= 1e-6;
}

export function gpuBudgetPassed(physicsP95: number | undefined, totalP95: number | undefined, count: number, errors: number) {
  return errors === 0 && count >= 60 && physicsP95 !== undefined && physicsP95 > 0 && physicsP95 <= 6
    && totalP95 !== undefined && totalP95 > 0 && totalP95 <= 14;
}

export function pacingAcceptance(sample: FrameReport, drainMs: number) {
  return sample.durationMs >= 30000 && sample.p95Ms <= 18.5
    && sample.over25Ms / sample.intervalsMs.length <= .01
    && Math.max(...sample.intervalsMs) <= 100 && sample.simulationRate >= .98
    && sample.simulationRate <= 1.02 && drainMs <= 33;
}

/** Sparse diagnostic timestamps surround ALL physics substeps and rendering in the same
 * frame. Queue-envelope durations include intervening CPU submission gaps, not just active
 * GPU execution. No timestamp is inferred from rAF or multiplied from one substep. */
export class QualificationProbe {
  private query: GPUQuerySet | null = null;
  private marker: GPUComputePipeline | null = null;
  private markerBuffer: GPUBuffer | null = null;
  private markerGroup: GPUBindGroup | null = null;
  private resolve: GPUBuffer | null = null;
  private read: GPUBuffer | null = null;
  private pending: Promise<void> | null = null;
  private selected = false;
  private sampledSteps = 0;
  private attempts = 0;
  private disposed = false;
  private begun = 0;
  private physicsEnd = 0;
  private lastSample = -Infinity;
  private previous: number | null = null;
  private slowFrames = 0;
  private errors: string[] = [];
  private cpuPhysics: number[] = [];
  private cpuFrame: number[] = [];
  private gpu: { time: number; steps: number; physicsMs: number; renderMs: number; totalMs: number; completionMs: number }[] = [];
  private heapStart = this.heap();
  private onError = (event: GPUUncapturedErrorEvent) => { this.errors.push(event.error.message); };
  private device: GPUDevice;
  constructor(device: GPUDevice, timestamps: boolean) {
    this.device = device;
    try {
    if (timestamps && device.features.has('timestamp-query')) {
      this.markerBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
      this.marker = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: '@group(0) @binding(0) var<storage, read_write> marker: atomic<u32>; @compute @workgroup_size(1) fn main() { atomicAdd(&marker, 1u); }' }), entryPoint: 'main' } });
      this.markerGroup = device.createBindGroup({ layout: this.marker.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.markerBuffer } }] });
      this.query = device.createQuerySet({ type: 'timestamp', count: 3 });
      this.resolve = device.createBuffer({ size: 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.read = device.createBuffer({ size: 24, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    } catch (error) { this.dispose(); throw error; }
    device.addEventListener('uncapturederror', this.onError);
    void device.lost.then(info => { if (!this.disposed) this.errors.push(`Device lost: ${info.message}`); });
  }
  private heap() {
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number } }).memory;
    return memory ? { usedJSHeapSize: memory.usedJSHeapSize, totalJSHeapSize: memory.totalJSHeapSize } : null;
  }
  private stamp(index: number) {
    const encoder = this.device.createCommandEncoder({ label: 'qualification timestamp boundary' });
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.query!, beginningOfPassWriteIndex: index } });
    pass.setPipeline(this.marker!); pass.setBindGroup(0, this.markerGroup!); pass.dispatchWorkgroups(1); pass.end();
    if (index === 2) { encoder.resolveQuerySet(this.query!, 0, 3, this.resolve!, 0); encoder.copyBufferToBuffer(this.resolve!, 0, this.read!, 0, 24); }
    this.device.queue.submit([encoder.finish()]);
  }
  get overloaded() { return this.slowFrames >= 5 || this.errors.length > 0; }
  begin(time: number) {
    if (this.previous !== null) this.slowFrames = time - this.previous > 100 ? this.slowFrames + 1 : 0;
    this.previous = time; this.begun = performance.now(); this.physicsEnd = this.begun;
    this.selected = !!this.query && !this.pending && time - this.lastSample >= qualificationSampleDelay(this.attempts);
    if (this.selected) { this.lastSample = time; this.attempts++; this.stamp(0); }
  }
  physics(steps: number) {
    this.sampledSteps = steps;
    this.physicsEnd = performance.now(); this.cpuPhysics.push(this.physicsEnd - this.begun);
    if (this.selected) this.stamp(1);
  }
  render() {
    this.cpuFrame.push(performance.now() - this.begun);
    if (!this.selected) return;
    this.selected = false; this.stamp(2);
    const began = this.begun, time = this.lastSample, steps = this.sampledSteps, read = this.read!;
    this.pending = read.mapAsync(GPUMapMode.READ).then(() => {
      const values = new BigUint64Array(read.getMappedRange());
      const physicsMs = Number(values[1] - values[0]) / 1e6;
      const renderMs = Number(values[2] - values[1]) / 1e6;
      const totalMs = Number(values[2] - values[0]) / 1e6;
      read.unmap();
      if (![physicsMs, renderMs, totalMs].every(value => Number.isFinite(value) && value >= 0)) throw new Error('Invalid GPU timestamps');
      this.gpu.push({ time, steps, physicsMs, renderMs, totalMs, completionMs: performance.now() - began });
    }).catch(error => { if (!this.disposed) this.errors.push(String(error)); }).finally(() => { this.pending = null; });
  }
  async finish(drainMs: number, sample: FrameReport) {
    await this.pending;
    const working = this.gpu.filter(value => value.steps > 0);
    const physics = distribution(working.map(value => value.physicsMs)), total = distribution(working.map(value => value.totalMs));
    const pacing = pacingAcceptance(sample, drainMs);
    const invalidWorkingSamples = working.filter(value => !validWorkingGpuSample(value)).length;
    return { cpuPhysicsMs: this.cpuPhysics, cpuFrameMs: this.cpuFrame, gpuSamples: this.gpu,
      cpuPhysics: distribution(this.cpuPhysics), cpuFrame: distribution(this.cpuFrame),
      gpuPhysicsEnvelope: physics, gpuTotalEnvelope: total,
      completion: distribution(this.gpu.map(value => value.completionMs)), heapStart: this.heapStart, heapEnd: this.heap(),
      timestamps: !!this.query, invalidWorkingSamples, deviceFeatures: [...this.device.features], errors: this.errors,
      complete: sample.durationMs >= 30000, endReason: this.errors.length ? 'gpu-error' : this.slowFrames >= 5 ? 'slow-frame-limit' : 'duration',
      pacingPassed: !this.errors.length && pacing, timingPassed: gpuBudgetPassed(physics?.p95, total?.p95, working.length, this.errors.length + invalidWorkingSamples),
      scope: 'Opt-in diagnostic capture. Sparse timestamp boundary dispatches (one atomic increment each; empty passes may yield zero timestamps) at varied 113–191ms spacing; timing acceptance excludes zero-step frames and validates every working sample. Envelopes include all substeps and scene/shadow rendering plus CPU queue gaps; not active GPU time or compositor presentation. Completion latency includes mapping/callback scheduling. Heap is Chromium JS heap only, not total GPU/process memory. Physical/overflow snapshot is post-capture only; no whole-run guarantee.' };
  }
  dispose() {
    this.disposed = true; this.device.removeEventListener('uncapturederror', this.onError);
    this.query?.destroy(); this.resolve?.destroy(); this.read?.destroy(); this.markerBuffer?.destroy();
  }
}

/** Outside the timed playback interval; ordinary playback adds no diagnostic pose readback. */
export async function inspectQualificationWorld(world: World): Promise<Record<string, unknown>> {
  const counters = await world.solver.readCounters(); await world.read();
  const bodies = world.bodies.filter(body => !body.fixed);
  const finite = bodies.every(body => [...body.position, ...body.rotation, ...body.velocity, ...body.angularVelocity].every(Number.isFinite));
  // Reachable GPU buffers are a useful allocation lower bound, not browser/driver memory.
  const visited = new WeakSet<object>(); let bytes = 0, buffers = 0;
  function visit(value: unknown, depth: number) {
    if (!value || typeof value !== 'object' || visited.has(value)) return;
    visited.add(value);
    if (value instanceof GPUBuffer) { bytes += value.size; buffers++; return; }
    if (depth <= 0 || ArrayBuffer.isView(value) || value instanceof ArrayBuffer || Array.isArray(value)) return;
    for (const child of Object.values(value)) visit(child, depth - 1);
  }
  visit(world, 5);
  return { finite, overflow: counters.overflow, counters, dynamicBodies: bodies.length,
    stepCount: world.stepCount, sdfError: world.sdfCollisionError?.message ?? null,
    reachableGpuBufferBytes: bytes, reachableGpuBuffers: buffers,
    poses: bodies.map(body => ({ position: body.position, rotation: body.rotation, velocity: body.velocity, angularVelocity: body.angularVelocity })) };
}
