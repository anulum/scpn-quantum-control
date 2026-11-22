// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas query conformance with the Python owner

import { describe, expect, it } from "vitest";
import corpusText from "../../../../tests/data/atlas/queries.json?raw";
import type { AtlasSnapshot } from "./atlas";
import { parseAtlasSnapshot, projectAtlasPublic } from "./atlas";
import type { AtlasQuery, AtlasQueryOutcome } from "./atlasQuery";
import { atlasNodeCeiling, rememberedAtlasSnapshotDigest, runAtlasQuery } from "./atlasQuery";

interface RecordedPage {
  readonly snapshot_digest: string;
  readonly shape: string;
  readonly nodes: readonly string[];
  readonly edges: readonly (readonly string[])[];
  readonly paths: readonly (readonly string[])[];
  readonly total: number;
  readonly omitted: number;
  readonly omitted_edges: number;
  readonly next_cursor: string | null;
  readonly explanation: readonly string[];
}
interface Case {
  readonly id: string;
  readonly snapshot: string;
  readonly query: Record<string, unknown>;
  readonly access: string;
  readonly expect:
    | { readonly outcome: "answer"; readonly pages: readonly RecordedPage[] }
    | { readonly outcome: "refusal"; readonly code: string; readonly field: string };
}
// The query corpus holds only small integers, so plain JSON numbers are exact here.
const corpus = JSON.parse(corpusText) as { snapshots: Record<string, unknown>; cases: Case[] };

function snapshotOf(name: string): AtlasSnapshot {
  const parsed = parseAtlasSnapshot(corpus.snapshots[name]);
  if (!parsed.ok) throw new Error(`${parsed.code} at ${parsed.path}`);
  return parsed.value;
}
function queryOf(raw: Record<string, unknown>): AtlasQuery {
  return {
    text: (raw["text"] as string | undefined) ?? null,
    kinds: (raw["kinds"] as string[] | undefined) ?? [],
    statuses: (raw["statuses"] as string[] | undefined) ?? [],
    edgeKinds: (raw["edge_kinds"] as string[] | undefined) ?? [],
    start: (raw["start"] as string | undefined) ?? null,
    target: (raw["target"] as string | undefined) ?? null,
    direction: (raw["direction"] as string | undefined) ?? "out",
    depth: (raw["depth"] as number | undefined) ?? 1,
    pageSize: (raw["page_size"] as number | undefined) ?? 500,
  };
}
function refusal(outcome: AtlasQueryOutcome): [string, string] {
  if (outcome.ok) throw new Error("expected a refusal");
  return [outcome.code, outcome.field];
}

describe("Atlas queries", () => {
  it("answers every shared case exactly as the Python owner, page by page", async () => {
    for (const item of corpus.cases) {
      const snapshot = snapshotOf(item.snapshot);
      const query = queryOf(item.query);
      if (item.expect.outcome === "refusal") {
        expect(refusal(await runAtlasQuery(snapshot, query, item.access)), item.id).toEqual([
          item.expect.code,
          item.expect.field,
        ]);
        continue;
      }
      let cursor: string | null = null;
      for (const recorded of item.expect.pages) {
        const outcome = await runAtlasQuery(snapshot, { ...query, cursor }, item.access);
        if (!outcome.ok) throw new Error(`${item.id}: ${outcome.code}`);
        const page = outcome.value;
        expect(
          {
            snapshot_digest: page.snapshotDigest,
            shape: page.shape,
            nodes: page.nodes.map((node) => node.id),
            edges: page.edges.map((edge) => [edge.source, edge.kind, edge.target]),
            paths: page.paths.map((path) => [...path]),
            total: page.total,
            omitted: page.omitted,
            omitted_edges: page.omittedEdges,
            next_cursor: page.nextCursor,
            explanation: [...page.explanation],
          },
          item.id,
        ).toEqual(recorded);
        cursor = page.nextCursor;
      }
    }
  });

  it("refuses cursors that belong elsewhere or are not cursors", async () => {
    const diamond = snapshotOf("diamond");
    const first = await runAtlasQuery(diamond, { pageSize: 2 });
    if (!first.ok) throw new Error("first page refused");
    const cursor = first.value.nextCursor;
    expect(refusal(await runAtlasQuery(diamond, { pageSize: 2, text: "node", cursor }))).toEqual([
      "cursor_mismatch",
      "cursor",
    ]);
    expect(refusal(await runAtlasQuery(snapshotOf("cycle"), { pageSize: 2, cursor }))).toEqual([
      "cursor_mismatch",
      "cursor",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { cursor: "not base64 !" }))).toEqual([
      "invalid_cursor",
      "cursor",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { cursor: btoa("[1]") }))).toEqual([
      "invalid_cursor",
      "cursor",
    ]);
    const forged = btoa(
      JSON.stringify({
        ...JSON.parse(atob(String(cursor).replaceAll("-", "+").replaceAll("_", "/"))),
        last: "definition:fixture/zz",
      }),
    );
    expect(refusal(await runAtlasQuery(diamond, { pageSize: 2, cursor: forged }))).toEqual([
      "cursor_mismatch",
      "cursor",
    ]);
  });

  it("refuses absent, oversized and malformed inputs", async () => {
    const diamond = snapshotOf("diamond");
    const huge = {
      ...diamond,
      nodes: Array.from({ length: atlasNodeCeiling + 1 }, () => diamond.nodes[0]),
    } as AtlasSnapshot;
    expect(refusal(await runAtlasQuery(null, {}))).toEqual(["absent_snapshot", "snapshot"]);
    expect(refusal(await runAtlasQuery(huge, {}))).toEqual(["snapshot_too_large", "snapshot"]);
    expect(refusal(await runAtlasQuery(diamond, { kinds: ["module"] }))).toEqual([
      "unknown_node_kind",
      "kinds",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { statuses: ["fresh"] }))).toEqual([
      "unknown_status",
      "statuses",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { direction: "sideways" }))).toEqual([
      "invalid_direction",
      "direction",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { depth: 1.5 }))).toEqual([
      "oversized_traversal",
      "depth",
    ]);
    expect(refusal(await runAtlasQuery(diamond, { pageSize: 0 }))).toEqual([
      "oversized_page",
      "page_size",
    ]);
    expect(
      refusal(
        await runAtlasQuery(diamond, {
          start: "definition:fixture/a",
          target: "definition:fixture/zz",
        }),
      ),
    ).toEqual(["unknown_node", "target"]);
  });

  it("computes each snapshot's digest and public projection once and reuses them", async () => {
    const diamond = snapshotOf("diamond");
    const first = await runAtlasQuery(diamond, {});
    const second = await runAtlasQuery(diamond, { text: "node" });
    if (!first.ok || !second.ok) throw new Error("refused");
    expect(second.value.snapshotDigest).toBe(first.value.snapshotDigest);
    const published = projectAtlasPublic(diamond);
    expect(rememberedAtlasSnapshotDigest(published)).toBe(rememberedAtlasSnapshotDigest(published));
    const own = await runAtlasQuery(published, {});
    if (!own.ok) throw new Error("refused");
    expect(own.value.snapshotDigest).toBe(await rememberedAtlasSnapshotDigest(published));
    expect(own.value.snapshotDigest).toBe(first.value.snapshotDigest);
  });

  it("reports edge truncation and rethrows failures that are not refusals", async () => {
    const diamond = snapshotOf("diamond");
    const outcome = await runAtlasQuery(diamond, { start: "definition:fixture/a", depth: 6 });
    if (!outcome.ok) throw new Error("refused");
    expect(outcome.value.omittedEdges).toBe(0);
    const hostile = {
      ...diamond,
      get nodes(): never {
        throw new TypeError("hostile snapshot");
      },
    } as unknown as AtlasSnapshot;
    await expect(runAtlasQuery(hostile, {})).rejects.toThrow(TypeError);
  });
});
