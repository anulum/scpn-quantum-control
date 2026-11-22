// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — worker that admits and queries a large Atlas snapshot off the main thread

import type { AtlasSnapshot } from "../../shared/contracts/atlas";
import { runAtlasQuery } from "../../shared/contracts/atlasQuery";
import { admitAtlasSnapshot } from "./atlasSnapshot";
import {
  type AtlasWorkerReply,
  type AtlasWorkerRequest,
  readAtlasWorkerRequest,
} from "./atlasWorkerProtocol";

/** Worker operations the entry point needs. */
export interface AtlasWorkerScope {
  /** Receive a structured-cloned request. */
  addEventListener(kind: "message", listener: (event: { readonly data: unknown }) => void): void;
  /** Send one reply. */
  postMessage(value: AtlasWorkerReply): void;
}

/**
 * Answer requests in arrival order on one admitted snapshot.
 *
 * An open request admits a pin with the same three checks as the main thread and replaces
 * the snapshot; a query runs the shared query owner. A request that fails is answered with
 * its reason, and the worker keeps serving.
 *
 * @param scope - The worker's global scope.
 */
export function installAtlasWorker(scope: AtlasWorkerScope): void {
  let snapshot: AtlasSnapshot | null = null;
  let queue: Promise<void> = Promise.resolve();
  const answer = async (request: AtlasWorkerRequest): Promise<AtlasWorkerReply> => {
    if (request.type === "query") {
      if (snapshot === null)
        return { type: "failed", id: request.id, reason: "no snapshot is admitted" };
      return {
        type: "answer",
        id: request.id,
        outcome: await runAtlasQuery(snapshot, request.query),
      };
    }
    snapshot = null;
    const text = typeof request.source === "string" ? request.source : await request.source.text();
    const admission = await admitAtlasSnapshot(text);
    if (!admission.ok) return { type: "refused", id: request.id, reason: admission.reason };
    snapshot = admission.snapshot;
    return {
      type: "opened",
      id: request.id,
      summary: {
        digest: admission.digest,
        sourceCommit: snapshot.sourceCommit,
        nodes: snapshot.nodes.length,
        edges: snapshot.edges.length,
      },
    };
  };
  scope.addEventListener("message", (event) => {
    queue = queue.then(async () => {
      const request = readAtlasWorkerRequest(event.data);
      if (request === null) {
        scope.postMessage({
          type: "failed",
          id: -1,
          reason: "the request is not an Atlas request",
        });
        return;
      }
      try {
        scope.postMessage(await answer(request));
      } catch (error: unknown) {
        const reason = error instanceof Error ? error.message : "the request failed";
        scope.postMessage({ type: "failed", id: request.id, reason });
      }
    });
  });
}

installAtlasWorker(globalThis as unknown as AtlasWorkerScope);
