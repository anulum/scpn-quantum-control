// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — open Atlas snapshots and query them on the main thread or in a worker

import pinText from "../../../../data/studio/atlas_snapshot_20261122.json?raw";
import type { AtlasSnapshot } from "../../shared/contracts/atlas";
import {
  type AtlasQuery,
  type AtlasQueryOutcome,
  runAtlasQuery,
} from "../../shared/contracts/atlasQuery";
import { admitAtlasSnapshot } from "./atlasSnapshot";
import {
  type AtlasSnapshotSummary,
  type AtlasWorkerReply,
  type AtlasWorkerRequest,
  readAtlasWorkerReply,
} from "./atlasWorkerProtocol";

/**
 * Pins above this size, in characters of text or bytes of a file, are admitted and queried
 * in a worker, so the main thread stays responsive and a running query can be cancelled.
 *
 * Admission is dominated by the canonical digest. On the slowest reference host, a 309 kB
 * pin took 74 ms to admit and a 1.16 MB pin 697 ms, so this bound keeps main-thread
 * admission near 100 ms there.
 */
export const atlasWorkerThreshold = 400_000;

/** Where a session admits and queries its snapshot. */
export type AtlasTier = "main thread" | "worker";

/** What the view shows about the open snapshot. */
export interface AtlasSessionInfo extends AtlasSnapshotSummary {
  /** Where the pin came from. */
  readonly origin: string;
  /** Where admission and queries run. */
  readonly tier: AtlasTier;
}

/** One running query. */
export interface AtlasQueryTask {
  /** The outcome, or null once the query was cancelled. */
  readonly outcome: Promise<AtlasQueryOutcome | null>;
  /** Cancel the query; in a worker this stops the worker, so no result can arrive. */
  cancel(): void;
}

/** One open snapshot. */
export interface AtlasSession {
  /** The open snapshot. */
  readonly info: AtlasSessionInfo;
  /** The validated snapshot on the main thread, or null when a worker holds it. */
  readonly snapshot: AtlasSnapshot | null;
  /**
   * Start one query.
   *
   * @param query - The query.
   * @returns The running query.
   */
  query(query: AtlasQuery): AtlasQueryTask;
  /** Release the session and stop its worker, if it has one. */
  dispose(): void;
}

/** An opened session, or why the pin was refused. */
export type AtlasOpening =
  | {
      /** The pin was admitted. */
      readonly ok: true;
      /** The open snapshot. */
      readonly session: AtlasSession;
    }
  | {
      /** The pin was refused; nothing from it is shown. */
      readonly ok: false;
      /** Fixed explanation of the refusal. */
      readonly reason: string;
    };

/** One opening in progress. */
export interface AtlasOpeningTask {
  /** Where the pin is admitted. */
  readonly tier: AtlasTier;
  /** The opening, or null once it was cancelled. */
  readonly result: Promise<AtlasOpening | null>;
  /** Cancel the opening; in a worker this stops the worker. */
  cancel(): void;
}

/** The worker operations a session uses. */
export interface AtlasWorkerPort {
  /** Receiver of replies. */
  onmessage: ((event: { readonly data: unknown }) => void) | null;
  /** Receiver of a worker failure. */
  onerror: ((event: { readonly message: string }) => void) | null;
  /** Send one request. */
  postMessage(message: AtlasWorkerRequest): void;
  /** Stop the worker at once. */
  terminate(): void;
}

/**
 * Start the Atlas worker of this build.
 *
 * @returns A port on a new worker.
 */
export function nativeAtlasWorker(): AtlasWorkerPort {
  const worker = new Worker(new URL("./atlasWorker.ts", import.meta.url), { type: "module" });
  const port: AtlasWorkerPort = {
    onmessage: null,
    onerror: null,
    postMessage: (message) => worker.postMessage(message),
    terminate: () => worker.terminate(),
  };
  worker.onmessage = (event: MessageEvent<unknown>) => port.onmessage?.({ data: event.data });
  worker.onerror = (event: ErrorEvent) => port.onerror?.({ message: event.message });
  return port;
}

function snapshotSession(
  snapshot: AtlasSnapshot,
  info: AtlasSessionInfo,
  access = "public",
): AtlasSession {
  return {
    info,
    snapshot,
    query: (query) => mainThreadQuery(snapshot, query, access),
    dispose: () => {},
  };
}

function mainThreadQuery(
  snapshot: AtlasSnapshot,
  query: AtlasQuery,
  access: string,
): AtlasQueryTask {
  let cancelled = false;
  return {
    outcome: runAtlasQuery(snapshot, query, access).then((outcome) => (cancelled ? null : outcome)),
    cancel: () => {
      cancelled = true;
    },
  };
}

async function readSource(source: string | Blob): Promise<string | null> {
  if (typeof source === "string") return source;
  try {
    return await source.text();
  } catch {
    return null;
  }
}

function openOnMainThread(source: string | Blob, origin: string): AtlasOpeningTask {
  let cancelled = false;
  const result = (async (): Promise<AtlasOpening | null> => {
    const text = await readSource(source);
    const admission =
      text === null
        ? { ok: false as const, reason: "The Atlas snapshot pin could not be read." }
        : await admitAtlasSnapshot(text);
    if (cancelled) return null;
    if (!admission.ok) return admission;
    const snapshot = admission.snapshot;
    const info: AtlasSessionInfo = {
      digest: admission.digest,
      sourceCommit: snapshot.sourceCommit,
      nodes: snapshot.nodes.length,
      edges: snapshot.edges.length,
      origin,
      tier: "main thread",
    };
    return {
      ok: true,
      session: snapshotSession(snapshot, info),
    };
  })();
  return {
    tier: "main thread",
    result,
    cancel: () => {
      cancelled = true;
    },
  };
}

/** One worker and the replies it still owes; a reset stops the worker and drops every reply. */
class AtlasWorkerChannel {
  private port: AtlasWorkerPort | null = null;
  private sequence = 0;
  private readonly waiting = new Map<number, (reply: AtlasWorkerReply | null) => void>();

  constructor(
    private readonly source: string | Blob,
    private readonly factory: () => AtlasWorkerPort,
  ) {}

  /** Send one request, starting a worker and readmitting the pin first after a reset. */
  request(
    body: { readonly type: "open" } | { readonly type: "query"; readonly query: AtlasQuery },
  ): { readonly reply: Promise<AtlasWorkerReply | null>; cancel(): void } {
    let port = this.port;
    if (port === null) {
      port = this.factory();
      port.onmessage = (event) => this.receive(event.data);
      port.onerror = (event) => this.fail(event.message);
      this.port = port;
      if (body.type === "query") port.postMessage({ type: "open", id: 0, source: this.source });
    }
    this.sequence += 1;
    const id = this.sequence;
    const reply = new Promise<AtlasWorkerReply | null>((resolve) => this.waiting.set(id, resolve));
    port.postMessage(
      body.type === "open"
        ? { type: "open", id, source: this.source }
        : { type: "query", id, query: body.query },
    );
    return { reply, cancel: () => this.waiting.has(id) && this.reset() };
  }

  /** Stop the worker and settle every outstanding reply as cancelled. */
  reset(): void {
    this.port?.terminate();
    this.port = null;
    for (const resolve of this.waiting.values()) resolve(null);
    this.waiting.clear();
  }

  private receive(data: unknown): void {
    const reply = readAtlasWorkerReply(data);
    if (reply === null) {
      this.fail("the worker sent a message that is not an Atlas reply");
      return;
    }
    const resolve = this.waiting.get(reply.id);
    this.waiting.delete(reply.id);
    resolve?.(reply);
  }

  private fail(reason: string): void {
    const waiting = [...this.waiting.entries()];
    this.waiting.clear();
    this.reset();
    for (const [id, resolve] of waiting) resolve({ type: "failed", id, reason });
  }
}

function openInWorker(
  source: string | Blob,
  origin: string,
  factory: () => AtlasWorkerPort,
): AtlasOpeningTask {
  const channel = new AtlasWorkerChannel(source, factory);
  const opening = channel.request({ type: "open" });
  const result = opening.reply.then((reply): AtlasOpening | null => {
    if (reply?.type !== "opened") channel.reset();
    if (reply === null) return null;
    if (reply.type !== "opened") return { ok: false, reason: reasonOf(reply) };
    return {
      ok: true,
      session: {
        info: { ...reply.summary, origin, tier: "worker" },
        snapshot: null,
        query: (query) => {
          const task = channel.request({ type: "query", query });
          return {
            outcome: task.reply.then((answer) =>
              answer === null || answer.type === "answer"
                ? (answer?.outcome ?? null)
                : {
                    ok: false,
                    code: "worker_failed",
                    field: "snapshot",
                    message: reasonOf(answer),
                  },
            ),
            cancel: task.cancel,
          };
        },
        dispose: () => channel.reset(),
      },
    };
  });
  return { tier: "worker", result, cancel: opening.cancel };
}

function reasonOf(reply: AtlasWorkerReply): string {
  return "reason" in reply ? reply.reason : `the worker answered with ${reply.type}`;
}

/**
 * Open a snapshot pin: on the main thread up to {@link atlasWorkerThreshold}, in a worker above it.
 *
 * @param source - The pin's text, or a file holding it.
 * @param origin - Where the pin came from, as the view names it.
 * @param workerFactory - Starts the worker for a large pin.
 * @returns The opening in progress.
 */
export function openAtlasSession(
  source: string | Blob,
  origin: string,
  workerFactory: () => AtlasWorkerPort = nativeAtlasWorker,
): AtlasOpeningTask {
  const size = typeof source === "string" ? source.length : source.size;
  return size > atlasWorkerThreshold
    ? openInWorker(source, origin, workerFactory)
    : openOnMainThread(source, origin);
}

/** Origin of the snapshot committed with this build. */
export const committedOrigin = "committed with this build";
/** Origin of the private overlay of the committed snapshot and the admitted workspace. */
export const workspaceOverlayOrigin = "committed with this build, with your workspace";

/**
 * Open a main-thread session over a snapshot that is already validated.
 *
 * Queries run at the snapshot's own access scope, so a private overlay shows its
 * private objects to the person whose workspace it is; a pin keeps public queries.
 *
 * @param snapshot - The validated snapshot, such as a workspace overlay.
 * @param digest - Its canonical digest.
 * @param origin - Where it came from, as the view states it.
 * @returns The session; it holds no worker.
 */
export function openSnapshotSession(
  snapshot: AtlasSnapshot,
  digest: string,
  origin: string,
): AtlasSession {
  return snapshotSession(
    snapshot,
    {
      digest,
      sourceCommit: snapshot.sourceCommit,
      nodes: snapshot.nodes.length,
      edges: snapshot.edges.length,
      origin,
      tier: "main thread",
    },
    snapshot.accessScope,
  );
}

/**
 * Open the snapshot committed with this build.
 *
 * @returns The opening in progress.
 */
export function openCommittedAtlasSession(): AtlasOpeningTask {
  return openAtlasSession(pinText, committedOrigin);
}
