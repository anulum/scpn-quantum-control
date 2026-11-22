// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas worker tests on the real entry point

import { describe, expect, it, vi } from "vitest";
import pinText from "../../../../data/studio/atlas_snapshot_20261122.json?raw";
import { type AtlasWorkerScope, installAtlasWorker } from "./atlasWorker";
import type { AtlasWorkerReply } from "./atlasWorkerProtocol";

const pin = JSON.parse(pinText) as {
  readonly snapshot: { readonly nodes: unknown[]; readonly edges: unknown[] };
  readonly snapshot_digest: string;
};

/** A worker scope whose messages the test sends and whose replies it reads. */
function scope(): { send: (data: unknown) => void; replies: AtlasWorkerReply[] } {
  let listener: ((event: { readonly data: unknown }) => void) | null = null;
  const replies: AtlasWorkerReply[] = [];
  const worker: AtlasWorkerScope = {
    addEventListener: (_kind, received) => {
      listener = received;
    },
    postMessage: (value) => {
      replies.push(value);
    },
  };
  installAtlasWorker(worker);
  return { send: (data) => listener?.({ data }), replies };
}

class UnreadableBlob extends Blob {
  constructor(private readonly failure: unknown) {
    super(["{}"]);
  }
  override text(): Promise<string> {
    return Promise.reject(this.failure);
  }
}

describe("Atlas worker", () => {
  it("admits a pin, answers queries in order and keeps serving after a bad request", async () => {
    const worker = scope();
    worker.send({ type: "query", id: 1, query: {} });
    worker.send({ type: "open", id: 2, source: pinText });
    worker.send({ type: "query", id: 3, query: { kinds: ["capability"] } });
    worker.send({ type: "query", id: 4, query: { kinds: ["planet"] } });
    worker.send({ type: "nonsense" });
    worker.send({ type: "open", id: 5, source: new Blob([pinText]) });
    await vi.waitFor(() => expect(worker.replies).toHaveLength(6), { timeout: 15_000 });
    expect(worker.replies.map((reply) => [reply.type, reply.id])).toEqual([
      ["failed", 1],
      ["opened", 2],
      ["answer", 3],
      ["answer", 4],
      ["failed", -1],
      ["opened", 5],
    ]);
    expect(worker.replies[0]).toMatchObject({ reason: "no snapshot is admitted" });
    expect(worker.replies[1]).toMatchObject({
      summary: {
        digest: pin.snapshot_digest,
        nodes: pin.snapshot.nodes.length,
        edges: pin.snapshot.edges.length,
      },
    });
    const answer = worker.replies[2];
    if (answer?.type !== "answer" || !answer.outcome.ok) throw new Error("expected an answer");
    expect(answer.outcome.value.nodes.every((node) => node.kind === "capability")).toBe(true);
    expect(worker.replies[3]).toMatchObject({ outcome: { ok: false, code: "unknown_node_kind" } });
    expect(worker.replies[4]).toMatchObject({ reason: "the request is not an Atlas request" });
  });

  it("drops the admitted snapshot when a later pin is refused or unreadable", async () => {
    const worker = scope();
    worker.send({ type: "open", id: 1, source: pinText });
    worker.send({ type: "open", id: 2, source: "{not json" });
    worker.send({ type: "query", id: 3, query: {} });
    worker.send({ type: "open", id: 4, source: new UnreadableBlob(new Error("unreadable")) });
    worker.send({ type: "open", id: 5, source: new UnreadableBlob("no error object") });
    await vi.waitFor(() => expect(worker.replies).toHaveLength(5), { timeout: 15_000 });
    expect(worker.replies.slice(1)).toEqual([
      { type: "refused", id: 2, reason: "The Atlas snapshot pin is not JSON." },
      { type: "failed", id: 3, reason: "no snapshot is admitted" },
      { type: "failed", id: 4, reason: "unreadable" },
      { type: "failed", id: 5, reason: "the request failed" },
    ]);
  });
});
