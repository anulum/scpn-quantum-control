// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas view model tests

import { describe, expect, it } from "vitest";
import type { AtlasEdge, AtlasNode } from "../../shared/contracts/atlas";
import type { AtlasQueryAnswer } from "../../shared/contracts/atlasQuery";
import {
  aggregateAtlasAnswer,
  atlasFilterBound,
  atlasKeyTarget,
  atlasLensOfKind,
  atlasNodeZoomLimit,
  atlasViewHref,
  atlasViewQuery,
  automaticAtlasZoom,
  defaultAtlasFilter,
  defaultAtlasLens,
  formatAtlasFilter,
  layoutAtlasAnswer,
  lensOf,
  parseAtlasFilter,
} from "./atlasViewModel";

function node(id: string, kind = "definition"): AtlasNode {
  return {
    id,
    kind: kind as AtlasNode["kind"],
    label: id,
    sourceRef: id,
    sourceDigest: null,
    validityDomain: null,
    asOf: null,
    access: "public",
    status: "current",
    evidenceRefs: [],
    nullReasons: [],
  };
}
function edge(source: string, target: string): AtlasEdge {
  return {
    source,
    kind: "requires",
    target,
    relationVersion: "atlas_relation.v1",
    derivationMethod: "declared",
    sourceRef: "fixture",
    evidenceDigest: null,
    assumptions: [],
    mapping: null,
    asOf: null,
    access: "public",
  };
}
function answer(nodes: AtlasNode[], edges: AtlasEdge[]): AtlasQueryAnswer {
  return {
    snapshotDigest: "d",
    accessScope: "public",
    shape: "lookup",
    nodes,
    edges,
    paths: [],
    total: nodes.length,
    omitted: 0,
    omittedEdges: 0,
    nextCursor: null,
    explanation: [],
  };
}
const line = answer([node("a"), node("b"), node("c", "test")], [edge("a", "b")]);

describe("Atlas view model", () => {
  it("maps instruments to lenses and lens state to queries", () => {
    expect(lensOf(null)).toBe(defaultAtlasLens);
    expect(lensOf("unknown")).toBe(defaultAtlasLens);
    expect(lensOf("claims")).toBe("claims");
    expect(atlasViewQuery("evidence", "", null, 2)).toMatchObject({
      text: null,
      kinds: ["artefact"],
      start: null,
      depth: 2,
    });
    expect(atlasViewQuery("claims", "x", "a", 1).text).toBe("x");
  });

  it("lays out by kind, by distance from the focus, and keeps pinned positions", () => {
    const byKind = layoutAtlasAnswer(line, null);
    expect(byKind.placed.map((item) => [item.node.id, item.x, item.y])).toEqual([
      ["a", 40, 40],
      ["b", 40, 84],
      ["c", 260, 40],
    ]);
    const byDistance = layoutAtlasAnswer(line, "b");
    expect(byDistance.placed.map((item) => item.x)).toEqual([260, 40, 480]);
    const missingFocus = layoutAtlasAnswer(line, "zz");
    expect(missingFocus.placed.map((item) => item.x)).toEqual([480, 480, 480]);
    const pinned = layoutAtlasAnswer(line, null, new Map([["b", [7, 9] as const]]));
    expect(pinned.placed[1]).toMatchObject({ x: 7, y: 9 });
    expect(byKind.width).toBeGreaterThanOrEqual(320);
  });

  it("moves the selection along edges and through the answer order", () => {
    expect(atlasKeyTarget(answer([], []), null, "ArrowDown")).toBeNull();
    expect(atlasKeyTarget(line, null, "ArrowDown")).toBe("a");
    expect(atlasKeyTarget(line, null, "x")).toBeNull();
    expect(atlasKeyTarget(line, "a", "ArrowRight")).toBe("b");
    expect(atlasKeyTarget(line, "b", "ArrowLeft")).toBe("a");
    expect(atlasKeyTarget(line, "c", "ArrowRight")).toBeNull();
    expect(atlasKeyTarget(line, "a", "ArrowDown")).toBe("b");
    expect(atlasKeyTarget(line, "c", "ArrowDown")).toBe("c");
    expect(atlasKeyTarget(line, "a", "ArrowUp")).toBe("a");
    expect(atlasKeyTarget(line, "b", "ArrowUp")).toBe("a");
    expect(atlasKeyTarget(line, "a", "Home")).toBeNull();
  });

  it("names the first lens that keeps a node kind, or none", () => {
    expect(atlasLensOfKind("capability")).toBe("architecture");
    expect(atlasLensOfKind("claim")).toBe("claims");
    expect(atlasLensOfKind("artefact")).toBe("evidence");
    expect(atlasLensOfKind("revision")).toBe("workspace");
    expect(atlasLensOfKind("backend")).toBeNull();
  });

  it("admits a filter only with known, single, non-empty fields and a bounded hop count", () => {
    expect(parseAtlasFilter(null)).toEqual({ ok: true, filter: defaultAtlasFilter });
    expect(parseAtlasFilter("q=node+e&focus=definition%3Aa%2Fb&hops=3")).toEqual({
      ok: true,
      filter: { text: "node e", focus: "definition:a/b", depth: 3, selected: null },
    });
    const refusals = Object.fromEntries(
      ["zoom=1", "q=a&q=b", "q=", "hops=0", "hops=7", "hops=2.5", "hops=10"].map((value) => {
        const result = parseAtlasFilter(value);
        return [value, result.ok ? "admitted" : result.reason];
      }),
    );
    expect(refusals).toEqual({
      "zoom=1": "it has an unknown or repeated field",
      "q=a&q=b": "it has an unknown or repeated field",
      "q=": "its q field is empty",
      "hops=0": "its hops field must be 1 to 6",
      "hops=7": "its hops field must be 1 to 6",
      "hops=2.5": "its hops field must be 1 to 6",
      "hops=10": "its hops field must be 1 to 6",
    });
  });

  it("encodes filters in a stable order and round-trips them through a deep link", () => {
    expect(formatAtlasFilter(defaultAtlasFilter)).toBeNull();
    const filter = { text: "node e", focus: "definition:a/b", depth: 2, selected: null };
    const encoded = formatAtlasFilter(filter);
    expect(encoded).toBe("q=node+e&focus=definition%3Aa%2Fb&hops=2");
    const chosen = { ...defaultAtlasFilter, selected: "definition:repo/src:a.py" };
    expect(formatAtlasFilter(chosen)).toBe("sel=definition%3Arepo%2Fsrc%3Aa.py");
    expect(parseAtlasFilter(formatAtlasFilter(chosen))).toEqual({ ok: true, filter: chosen });
    expect(parseAtlasFilter("sel=a&sel=b")).toEqual({
      ok: false,
      reason: "it has an unknown or repeated field",
    });
    expect(parseAtlasFilter(encoded)).toEqual({ ok: true, filter });
    const href = atlasViewHref("claims", "?project=p+%CE%B1&snapshot=old&revision=r", "d1", filter);
    expect(href).toBe(
      `#/atlas/claims?project=p+%CE%B1&revision=r&snapshot=d1&filter=${encodeURIComponent(encoded as string).replaceAll("%20", "+")}`,
    );
    const context = new URLSearchParams((href as string).split("?")[1]);
    expect(parseAtlasFilter(context.get("filter"))).toEqual({ ok: true, filter });
    expect(atlasViewHref("evidence", "", "d1", defaultAtlasFilter)).toBe(
      "#/atlas/evidence?snapshot=d1",
    );
    const long = { ...defaultAtlasFilter, text: "x".repeat(atlasFilterBound) };
    expect(atlasViewHref("claims", "", "d1", long)).toBeNull();
  });

  it("opens large pages on kind groups and counts relations between kinds", () => {
    expect(automaticAtlasZoom(line)).toBe("nodes");
    const many = answer(
      Array.from({ length: atlasNodeZoomLimit + 1 }, (_unused, index) => node(`n${index}`)),
      [],
    );
    expect(automaticAtlasZoom(many)).toBe("kinds");
    const mixed = answer(
      [node("a"), node("b"), node("t", "test"), node("u", "test")],
      [edge("a", "b"), edge("t", "a"), edge("u", "a"), edge("u", "b")],
    );
    expect(aggregateAtlasAnswer(mixed)).toEqual({
      groups: [
        { kind: "definition", count: 2 },
        { kind: "test", count: 2 },
      ],
      links: [
        { source: "definition", kind: "requires", target: "definition", count: 1 },
        { source: "test", kind: "requires", target: "definition", count: 3 },
      ],
    });
  });
});
