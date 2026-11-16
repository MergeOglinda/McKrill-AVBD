import { qualificationOptions, QualificationProbe } from './qualification.ts';

/** Browser render cadence, separate from GPU duration and simulation throughput. */
export interface FrameReport {
  intervalsMs: number[];
  durationMs: number;
  fps: number;
  medianMs: number;
  p95Ms: number;
  over25Ms: number;
  simulationSeconds: number;
  simulationRate: number;
}
export function frameReport(intervalsMs: readonly number[], simulationSeconds: number): FrameReport {
  if (!intervalsMs.length || intervalsMs.some(value => !Number.isFinite(value) || value <= 0)
    || !Number.isFinite(simulationSeconds) || simulationSeconds < 0) throw new Error('Invalid browser frame sample');
  const sorted = [...intervalsMs].sort((a, b) => a - b);
  const durationMs = intervalsMs.reduce((sum, value) => sum + value, 0);
  return { intervalsMs: [...intervalsMs], durationMs, fps: intervalsMs.length * 1000 / durationMs,
    medianMs: sorted[Math.ceil(sorted.length * .5) - 1], p95Ms: sorted[Math.ceil(sorted.length * .95) - 1],
    over25Ms: intervalsMs.filter(value => value > 25).length,
    simulationSeconds, simulationRate: simulationSeconds * 1000 / durationMs };
}

/** A finite capture includes one second of warmup, then four seconds of actual render intervals. */
export class FrameCapture {
  readonly warmupMs: number;
  readonly sampleMs: number;
  private previous: number | null = null;
  private intervals: number[] = [];
  private simulated = 0;
  private done = false;
  private duration = 0;
  readonly startedAt: number;
  constructor(startedAt: number, warmupMs = 1000, sampleMs = 4000) {
    if (!Number.isFinite(startedAt) || startedAt < 0) throw new Error('Invalid browser capture start');
    if (!Number.isFinite(warmupMs) || warmupMs < 0 || !Number.isFinite(sampleMs) || sampleMs <= 0 || sampleMs > 30000) throw new Error('Invalid browser capture duration');
    this.startedAt = startedAt; this.warmupMs = warmupMs; this.sampleMs = sampleMs;
  }
  record(time: number, simulatedSeconds: number): FrameReport | null {
    if (this.done) throw new Error('Browser capture already finished');
    if (!Number.isFinite(time) || time < 0 || !Number.isFinite(simulatedSeconds) || simulatedSeconds < 0)
      throw new Error('Invalid browser capture frame');
    // A queued rAF can predate the click's performance.now() even though it runs after it.
    // Ignore it before the capture baseline, while keeping later backwards timestamps invalid.
    if (time < this.startedAt) {
      if (this.previous !== null) throw new Error('Browser capture timestamps must increase');
      return null;
    }
    if (time < this.startedAt + this.warmupMs) return null;
    if (this.previous !== null) {
      const interval = time - this.previous;
      if (interval <= 0) throw new Error('Browser capture timestamps must increase');
      this.intervals.push(interval); this.simulated += simulatedSeconds; this.duration += interval;
    }
    this.previous = time;
    if (this.duration < this.sampleMs) return null;
    this.done = true;
    return this.snapshot();
  }
  snapshot(): FrameReport | null { return this.intervals.length ? frameReport(this.intervals, this.simulated) : null; }
}

export interface BrowserFrameReport extends FrameReport {
  context: Record<string, unknown>;
  measuredAt: string;
  userAgent: string;
  viewport: { width: number; height: number; devicePixelRatio: number; drawingWidth: number; drawingHeight: number };
  captureStartedAt: number;
  warmupMs: number;
  requestedSampleMs: number;
  gpuDrainMs: number;
  fpsIncludingDrain: number;
  scope: string;
  qualification?: unknown;
}

/** A stalled device must not leave an explicit capture disabled indefinitely. */
export async function waitForGpuDrain(queue: Pick<GPUQueue, 'onSubmittedWorkDone'>, timeoutMs = 5000): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([queue.onSubmittedWorkDone(), new Promise<never>((_, reject) => {
      timeout = setTimeout(() => reject(new Error(`GPU queue did not finish within ${timeoutMs} ms`)), timeoutMs);
    })]);
  } finally { clearTimeout(timeout); }
}

/** Bound the entire finalization, including asynchronous maps and inspection. */
export async function captureDeadline<T>(operation: Promise<T>, timeoutMs = 5000): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([operation, new Promise<never>((_, reject) => {
      timeout = setTimeout(() => reject(new Error(`Capture finalization exceeded ${timeoutMs} ms`)), timeoutMs);
    })]);
  } finally { clearTimeout(timeout); }
}

/** Explicit opt-in capture; no background benchmark, automatic download or pose readback. */
export function browserPerformance(options: {
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context(): Record<string, unknown>;
  start(): boolean;
  pause(): void;
  inspect?(): Promise<Record<string, unknown>>;
}) {
  const qualification = qualificationOptions(new URLSearchParams(location.search));
  let probe: QualificationProbe | null = null;
  const label = qualification ? 'Measure 30s' : 'Measure 4s';
  const button = document.querySelector<HTMLButtonElement>('#measure')!;
  const output = document.querySelector<HTMLOutputElement>('#performance')!;
  const data = document.querySelector<HTMLPreElement>('#performance-data')!;
  const save = document.querySelector<HTMLButtonElement>('#save-performance')!;
  button.textContent = label;
  button.title = qualification ? 'Capture 30 seconds including startup, then pause and inspect' : button.title;
  let capture: FrameCapture | null = null, watchdog: ReturnType<typeof setTimeout> | undefined;
  let context: Record<string, unknown> = {}, viewport: BrowserFrameReport['viewport'];
  let downloadData: string | null = null;
  let report: BrowserFrameReport | null = null, finishing = false;
  let previous: number | null = null, live: number[] = [], lastLabel = 0, revision = 0;
  const stopTimer = () => { clearTimeout(watchdog); watchdog = undefined; };
  const clear = () => { previous = null; live = []; };
  function cancel(reason: string) {
    revision++;
    if (!capture) return;
    if (qualification) {
      downloadData = data.textContent = JSON.stringify({ ...capture.snapshot(), context, viewport, failed: true, reason,
        measuredAt: new Date().toISOString(), userAgent: navigator.userAgent, scope: 'Cancelled partial capture; no acceptance claim.' }, null, 2);
      save.disabled = false;
    }
    capture = null; probe?.dispose(); probe = null; stopTimer(); button.disabled = false; button.textContent = label;
    output.textContent = reason; clear();
  }
  function reset(reason: string) {
    cancel(reason); report = null; downloadData = null; data.textContent = 'No capture yet.'; save.disabled = true;
    output.textContent = reason; clear();
  }
  save.addEventListener('click', () => {
    if (!downloadData) return;
    const url = URL.createObjectURL(new Blob([downloadData + '\n'], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `sdf-${context.demo}-frames.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  button.addEventListener('click', () => {
    if (capture || finishing || !options.start()) return;
    context = options.context();
    viewport = { width: innerWidth, height: innerHeight, devicePixelRatio,
      drawingWidth: options.canvas.width, drawingHeight: options.canvas.height };
    report = null; downloadData = null; save.disabled = true; data.textContent = '';
    try { probe = qualification ? new QualificationProbe(options.device, qualification.timestamps) : null; }
    catch (error) {
      options.pause(); output.textContent = `Capture setup failed: ${String(error)}`;
      downloadData = data.textContent = JSON.stringify({ failed: true, error: String(error), context, viewport });
      save.disabled = false; return;
    }
    capture = new FrameCapture(performance.now(), qualification ? 0 : 1000, qualification ? 30000 : 4000); clear();
    button.disabled = true; button.textContent = 'Warming up…'; output.textContent = qualification ? '30s bounded capture · includes startup · then pause' : '1s warmup · 4s sample · then pause';
    watchdog = setTimeout(() => { cancel('Capture stopped: no complete visible frame sample.'); options.pause(); }, qualification ? 45000 : 12000);
  });
  async function finish(sample: FrameReport, measured: FrameCapture) {
    const version = revision, finishedProbe = probe; probe = null;
    capture = null; finishing = true; stopTimer(); options.pause(); clear();
    output.textContent = 'Finishing GPU work…';
    const began = performance.now();
    try {
      const result = await captureDeadline((async () => {
        await waitForGpuDrain(options.device.queue);
        const gpuDrainMs = performance.now() - began;
        const qualificationReport = finishedProbe ? await finishedProbe.finish(gpuDrainMs, sample) : undefined;
        if (version !== revision) return null;
        const inspection = finishedProbe && options.inspect ? await options.inspect() : undefined;
        return { gpuDrainMs, qualificationReport, inspection };
      })());
      if (version !== revision || !result) return;
      const { gpuDrainMs, qualificationReport, inspection } = result;
      report = { ...sample, context, measuredAt: new Date().toISOString(), userAgent: navigator.userAgent, viewport,
        captureStartedAt: measured.startedAt, warmupMs: measured.warmupMs, requestedSampleMs: measured.sampleMs, gpuDrainMs,
        fpsIncludingDrain: sample.intervalsMs.length * 1000 / (sample.durationMs + gpuDrainMs),
        ...(qualificationReport ? { qualification: { ...qualificationReport, inspection,
          screenPassed: qualificationReport.errors.length === 0 && (context.demo !== 'interactive' || inspection?.dynamicBodies === qualification?.count) && qualificationReport.pacingPassed && qualificationReport.timingPassed && inspection?.finite === true && inspection?.overflow === 0 && inspection?.sdfError === null } } : {}),
        scope: 'Visible browser render submissions at requestAnimationFrame timestamps. Includes simulation, rendering, UI and enabled overlays. Not GPU duration or verified compositor presentation. Simulation rate is fixed-step simulated seconds / sample wall seconds; GPU drain is recorded separately.' };
      output.textContent = `${report.fps.toFixed(1)} FPS · p95 ${report.p95Ms.toFixed(1)} ms · simulation ${report.simulationRate.toFixed(2)}×`;
      downloadData = data.textContent = JSON.stringify(report, null, 2);
      save.disabled = false;
    } catch (error) {
      if (version === revision) {
        output.textContent = `Capture failed: ${String(error)}`;
        downloadData = data.textContent = JSON.stringify({ ...sample, context, viewport, measuredAt: new Date().toISOString(),
          userAgent: navigator.userAgent, failed: true, error: String(error), scope: 'Incomplete finalization; no acceptance claim.' }, null, 2);
        save.disabled = false;
      }
    }
    finally { finishedProbe?.dispose(); finishing = false; button.disabled = false; button.textContent = label; }
  }
  return {
    beginFrame(time: number) { if (capture) probe?.begin(time); },
    afterPhysics(steps: number) { if (capture) probe?.physics(steps); },
    afterRender() { if (capture) probe?.render(); },
    get active() { return capture !== null || finishing; },
    get report() { return report; },
    cancel, reset,
    idle() { clear(); },
    frame(time: number, simulatedSeconds: number) {
      if (previous !== null && time > previous) { live.push(time - previous); if (live.length > 90) live.shift(); }
      previous = time;
      if (capture) {
        const measured = capture;
        let sample: FrameReport | null;
        try { sample = measured.record(time, simulatedSeconds); }
        catch { cancel('Capture stopped: insufficient valid frame samples.'); options.pause(); return; }
        button.textContent = time < measured.startedAt + measured.warmupMs ? 'Warming up…' : 'Measuring…';
        if (probe?.overloaded) sample ??= measured.snapshot();
        if (sample) void finish(sample, measured);
      } else if (!finishing && time - lastLabel > 500 && live.length > 1) {
        lastLabel = time;
        output.textContent = `${(live.length * 1000 / live.reduce((a, b) => a + b, 0)).toFixed(0)} FPS · browser cadence`;
      }
    },
  };
}
