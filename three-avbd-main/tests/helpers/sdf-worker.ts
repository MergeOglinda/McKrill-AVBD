// Node adapter exercises the real worker protocol without a browser/GPU context.
import { parentPort } from 'node:worker_threads';
import { installSdfWorker, type SdfWorkerScope } from '../../src/lib/sdf.ts';
const listeners = new Map<unknown, (data: unknown) => void>();
installSdfWorker({
  addEventListener(_type, listener) {
    const wrapped = (data: unknown) => listener({ data } as Parameters<typeof listener>[0]);
    listeners.set(listener, wrapped); parentPort!.on('message', wrapped);
  },
  removeEventListener(_type, listener) { const wrapped = listeners.get(listener); if (wrapped) parentPort!.off('message', wrapped); },
  postMessage(message, transfer) { parentPort!.postMessage(message, transfer as ArrayBuffer[]); },
} satisfies SdfWorkerScope);
