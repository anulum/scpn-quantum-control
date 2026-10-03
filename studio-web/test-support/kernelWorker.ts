// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real Node thread adapter for built browser worker tests

import { Worker } from "node:worker_threads";

/** Real worker port using the browser entry built by Vite, with native structured cloning. */
export class BuiltKernelWorker {
  private static readonly active = new Set<BuiltKernelWorker>();
  /** Actual native thread handles not yet observed exited or terminated. */
  static get activeCount(): number { return this.active.size; }
  /** Actual native thread constructor calls across the focused test process. */
  static started = 0;
  /** Browser message listener installed by the production client. */
  onmessage: ((event: { data: unknown }) => void) | null = null;
  /** Browser error listener installed by the production client. */
  onerror: ((event: { message: string }) => void) | null = null;
  private readonly thread: Worker;

  /** Start the exact built browser worker inside a native Node worker thread. */
  constructor(entryOverride?: string | URL) {
    const entry = typeof entryOverride === "string" ? entryOverride : process.env["STUDIO_KERNEL_WORKER_ENTRY"];
    if (!entry) throw new Error("A real built kernelWorker entry is required for this test");
    const bootstrap = `
      import { parentPort } from 'node:worker_threads';
      import { pathToFileURL } from 'node:url';
      globalThis.self = globalThis;
      globalThis.location = { href: pathToFileURL(${JSON.stringify(entry)}).href };
      globalThis.addEventListener = (kind, listener) => {
        if (kind !== 'message') throw new Error('unsupported worker listener');
        parentPort.on('message', data => listener({ data }));
      };
      globalThis.postMessage = (data, transfers) => parentPort.postMessage(data, transfers);
      await import(pathToFileURL(${JSON.stringify(entry)}).href);
    `;
    this.thread = new Worker(new URL(`data:text/javascript,${encodeURIComponent(bootstrap)}`));
    BuiltKernelWorker.active.add(this);
    BuiltKernelWorker.started++;
    this.thread.on("exit", () => BuiltKernelWorker.active.delete(this));
    this.thread.on("message", (data: unknown) => this.onmessage?.({ data }));
    this.thread.on("error", (error: Error) => this.onerror?.({ message: error.message }));
  }

  /** Send the production envelope with native transfer-list semantics. */
  postMessage(message: unknown, transfers: ArrayBuffer[] = []): void {
    this.thread.postMessage(message, transfers);
  }

  /** Observe native thread disposal before acknowledging cancellation. */
  async terminate(): Promise<void> {
    await this.thread.terminate();
    BuiltKernelWorker.active.delete(this);
  }
}
