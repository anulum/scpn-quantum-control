// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the real Atlas worker entry point, run in the test's own thread

import { installAtlasWorker } from "../src/modules/atlas/atlasWorker";
import type { AtlasWorkerPort } from "../src/modules/atlas/atlasSession";
import type { AtlasWorkerRequest } from "../src/modules/atlas/atlasWorkerProtocol";

/** A worker port on the production entry point; messages cross a macrotask, as between threads. */
export class InThreadAtlasWorker implements AtlasWorkerPort {
  /** Every port started since the last reset, in start order. */
  static started: InThreadAtlasWorker[] = [];
  /** Receiver of replies, installed by the session. */
  onmessage: ((event: { readonly data: unknown }) => void) | null = null;
  /** Receiver of a worker failure, installed by the session. */
  onerror: ((event: { readonly message: string }) => void) | null = null;
  /** Whether the session stopped this worker. */
  terminated = false;
  /** Every request the session sent, in order. */
  readonly sent: AtlasWorkerRequest[] = [];
  private listener: ((event: { readonly data: unknown }) => void) | null = null;

  /** Install the production worker entry point on this port's own scope. */
  constructor() {
    InThreadAtlasWorker.started.push(this);
    installAtlasWorker({
      addEventListener: (_kind, listener) => {
        this.listener = listener;
      },
      postMessage: (reply) => {
        if (!this.terminated) setTimeout(() => this.onmessage?.({ data: reply }), 0);
      },
    });
  }

  /** Deliver one request to the entry point after the current task, unless stopped. */
  postMessage(message: AtlasWorkerRequest): void {
    this.sent.push(message);
    if (!this.terminated) setTimeout(() => this.listener?.({ data: message }), 0);
  }

  /** Stop the worker; no reply is delivered afterwards. */
  terminate(): void {
    this.terminated = true;
  }
}
