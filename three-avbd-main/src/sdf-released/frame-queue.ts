/** Bound latency in interactive playback without changing fixed-step solver settings.
 * Call submitted after rendering. When full, defer the whole frame and retain its elapsed time. */
export class FrameQueue {
  private pending = 0;
  private debt = 0;
  resetTime() { this.debt = 0; }
  /** Offer at most one solver frame budget, keeping deferred time for later frames. */
  takeTime(elapsed: number, budget: number) {
    this.debt += elapsed;
    const time = Math.min(this.debt, budget);
    this.debt -= time;
    return time;
  }
  get available() { return this.pending < 2; }
  submitted(queue: Pick<GPUQueue, 'onSubmittedWorkDone'>, onError: (error: unknown) => void) {
    if (!this.available) throw new Error('Frame queue is full');
    this.pending++;
    void queue.onSubmittedWorkDone().then(() => { this.pending--; }, error => {
      this.pending--; onError(error);
    });
  }
}
