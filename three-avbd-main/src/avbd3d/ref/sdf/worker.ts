import { bakeVoxelSdfAsync, type BakeOptions, type BakedSdf, type TriangleMesh } from './bake.ts';
import { VoxelField } from './voxel.ts';
import type { Aabb, Vec } from './math.ts';

type WireOptions = Pick<BakeOptions, 'resolution' | 'normalize' | 'maxGridBytes'>;
interface WireField { dimensions: Vec; origin: Vec; cell: number; grid: Float32Array; bounds: Aabb; surfaceTriangles?: Float32Array }
type WireResult = Omit<BakedSdf, 'field'> & { field: WireField };
type Request = { type: 'bake'; id: number; mesh: TriangleMesh; options: WireOptions } | { type: 'cancel'; id: number };
type Response = { type: 'progress'; id: number; fraction: number } | { type: 'result'; id: number; result: WireResult } | { type: 'error'; id: number; name: string; message: string };
/** Minimal worker-global surface; call installSdfWorker(self) in a module Worker. */
export interface SdfWorkerScope {
  addEventListener(type: 'message', listener: (event: MessageEvent<Request>) => void): void;
  removeEventListener(type: 'message', listener: (event: MessageEvent<Request>) => void): void;
  postMessage(message: Response, transfer: Transferable[]): void;
}
/** Installs a single-job worker; returns cleanup. No Worker is created implicitly. */
export function installSdfWorker(scope: SdfWorkerScope): () => void {
  let active: { id: number; controller: AbortController } | undefined;
  const listener = (event: MessageEvent<Request>) => {
    const request = event.data;
    if (request.type === 'cancel') { if (active?.id === request.id) active.controller.abort(); return; }
    if (request.type !== 'bake') return;
    if (active) { scope.postMessage({ type: 'error', id: request.id, name: 'Error', message: 'SDF worker is busy' }, []); return; }
    const controller = new AbortController(); active = { id: request.id, controller };
    void bakeVoxelSdfAsync(request.mesh, { ...request.options, signal: controller.signal,
      onProgress: fraction => scope.postMessage({ type: 'progress', id: request.id, fraction }, []),
    }).then(result => {
      scope.postMessage({ type: 'result', id: request.id, result }, [result.field.grid.buffer as ArrayBuffer, ...(result.field.surfaceTriangles ? [result.field.surfaceTriangles.buffer as ArrayBuffer] : [])]);
    }).catch((error: unknown) => {
      scope.postMessage({ type: 'error', id: request.id, name: error instanceof Error ? error.name : 'Error', message: String(error instanceof Error ? error.message : error) }, []);
    }).finally(() => { active = undefined; });
  };
  scope.addEventListener('message', listener);
  return () => { active?.controller.abort(); scope.removeEventListener('message', listener); };
}
let nextId = 0;
const busy = new WeakSet<Worker>();
/** Copies input arrays before transfer; the caller's render buffers stay attached.
 * One job per supplied worker. Caller owns worker lifetime (terminate on disposal).
 */
export function bakeVoxelSdfInWorker(worker: Worker, mesh: TriangleMesh, options: BakeOptions = {}): Promise<BakedSdf> {
  if (options.signal?.aborted) return Promise.reject(new DOMException('SDF bake aborted', 'AbortError'));
  if (busy.has(worker)) return Promise.reject(new Error('SDF worker is busy'));
  const id = ++nextId;
  // Float64 copies preserve invalid fractional indices for worker-side validation.
  const positions = Float64Array.from(mesh.positions), indices = mesh.indices ? Float64Array.from(mesh.indices) : undefined;
  return new Promise((resolve, reject) => {
    busy.add(worker);
    let callbackError: unknown, callbackFailed = false;
    const cleanup = () => { busy.delete(worker); worker.removeEventListener('message', message); worker.removeEventListener('error', error); worker.removeEventListener('messageerror', error); options.signal?.removeEventListener('abort', cancel); };
    const cancel = () => worker.postMessage({ type: 'cancel', id } satisfies Request);
    const error = () => { cleanup(); reject(new Error('SDF worker failed to execute or deserialize a message')); };
    const message = (event: MessageEvent<Response>) => {
      const response = event.data; if (response.id !== id) return;
      if (response.type === 'progress') {
        try { options.onProgress?.(response.fraction); } catch (e) { callbackError = e; callbackFailed = true; cancel(); }
        return;
      }
      cleanup();
      if (callbackFailed) { reject(callbackError); return; }
      if (options.signal?.aborted) { reject(new DOMException('SDF bake aborted', 'AbortError')); return; }
      if (response.type === 'error') { const e = new Error(response.message); e.name = response.name; reject(e); return; }
      try {
        const { field, ...rest } = response.result;
        resolve({ ...rest, field: new VoxelField(field.dimensions, field.origin, field.cell, field.grid, field.bounds, field.surfaceTriangles) });
      } catch (e) { reject(e); }
    };
    worker.addEventListener('message', message); worker.addEventListener('error', error); worker.addEventListener('messageerror', error);
    options.signal?.addEventListener('abort', cancel, { once: true });
    try {
      worker.postMessage({ type: 'bake', id, mesh: { positions, indices }, options: {
        resolution: options.resolution, normalize: options.normalize, maxGridBytes: options.maxGridBytes,
      } } satisfies Request, indices ? [positions.buffer, indices.buffer] : [positions.buffer]);
    } catch (e) { cleanup(); reject(e); }
  });
}
