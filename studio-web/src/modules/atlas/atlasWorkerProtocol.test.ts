// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas worker message tests

import { describe, expect, it } from "vitest";
import { readAtlasWorkerReply, readAtlasWorkerRequest } from "./atlasWorkerProtocol";

const summary = { digest: "d", sourceCommit: "c", nodes: 2, edges: 1 };

describe("Atlas worker messages", () => {
  it("reads open and query requests and nothing else", () => {
    const file = new Blob(["{}"]);
    expect(readAtlasWorkerRequest({ type: "open", id: 0, source: "{}" })).toEqual({
      type: "open",
      id: 0,
      source: "{}",
    });
    expect(readAtlasWorkerRequest({ type: "open", id: 3, source: file })).toEqual({
      type: "open",
      id: 3,
      source: file,
    });
    expect(readAtlasWorkerRequest({ type: "query", id: 4, query: { text: "a" } })).toEqual({
      type: "query",
      id: 4,
      query: { text: "a" },
    });
    for (const malformed of [
      null,
      [],
      "open",
      { type: "open", source: "{}" },
      { type: "open", id: -1, source: "{}" },
      { type: "open", id: 1.5, source: "{}" },
      { type: "open", id: 1, source: 7 },
      { type: "query", id: 1, query: null },
      { type: "query", id: 1, query: [] },
      { type: "close", id: 1 },
    ])
      expect(readAtlasWorkerRequest(malformed)).toBeNull();
  });

  it("reads each reply kind and refuses replies with missing or mistyped fields", () => {
    expect(readAtlasWorkerReply({ type: "opened", id: 1, summary })).toEqual({
      type: "opened",
      id: 1,
      summary,
    });
    const outcome = { ok: false, code: "unknown_node", field: "start", message: "m" };
    expect(readAtlasWorkerReply({ type: "answer", id: 2, outcome })).toEqual({
      type: "answer",
      id: 2,
      outcome,
    });
    expect(readAtlasWorkerReply({ type: "refused", id: 3, reason: "r" })).toEqual({
      type: "refused",
      id: 3,
      reason: "r",
    });
    expect(readAtlasWorkerReply({ type: "failed", id: -1, reason: "r" })).toEqual({
      type: "failed",
      id: -1,
      reason: "r",
    });
    for (const malformed of [
      null,
      [],
      { type: "opened", summary },
      { type: "opened", id: -2, summary },
      { type: "opened", id: 1.5, summary },
      { type: "opened", id: 1, summary: null },
      { type: "opened", id: 1, summary: { ...summary, digest: 1 } },
      { type: "opened", id: 1, summary: { ...summary, sourceCommit: null } },
      { type: "opened", id: 1, summary: { ...summary, nodes: 0.5 } },
      { type: "opened", id: 1, summary: { ...summary, edges: "1" } },
      { type: "answer", id: 1, outcome: null },
      { type: "answer", id: 1, outcome: { ok: "yes" } },
      { type: "refused", id: 1 },
      { type: "failed", id: 1, reason: 4 },
      { type: "closed", id: 1, reason: "r" },
    ])
      expect(readAtlasWorkerReply(malformed)).toBeNull();
  });
});
