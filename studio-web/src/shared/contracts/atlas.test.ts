// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas snapshot contract conformance

import { describe, expect, it } from "vitest";
import corpusText from "../../../../tests/data/atlas/contracts.json?raw";
import {
  atlasEdgeEndpoints,
  atlasEdgeKinds,
  atlasIdentityClasses,
  atlasNodeKinds,
  atlasProvClasses,
  atlasProvRelations,
  atlasSnapshotDigest,
  atlasSnapshotToWire,
  parseAtlasSnapshot,
  projectAtlasPublic,
} from "./atlas";
import type { AtlasParseResult, AtlasSnapshot } from "./atlas";
import { readJson } from "./jsonTransport";

interface Case {
  readonly id: string;
  readonly fixture: string;
  readonly expectation: "accept" | "reject";
  readonly digest?: string;
  readonly code?: string;
  readonly path?: string;
  readonly edge_kinds?: readonly string[];
  readonly public_digest?: string;
  readonly public_node_ids?: readonly string[];
}
const corpus = readJson(corpusText) as { fixtures: Record<string, unknown>; cases: Case[] };
const fixture = (name: string): Record<string, unknown> =>
  structuredClone(corpus.fixtures[name]) as Record<string, unknown>;
const caseOf = (name: string): Case => corpus.cases.find((item) => item.id === name) as Case;

function take(result: AtlasParseResult): AtlasSnapshot {
  if (!result.ok) throw new Error(`${result.code} at ${result.path}: ${result.message}`);
  return result.value;
}
function refusal(payload: unknown): [string, string] {
  const result = parseAtlasSnapshot(payload);
  if (result.ok) throw new Error("expected a refusal");
  return [result.code, result.path];
}
function at<T>(list: readonly T[], index: number): T {
  const value = list[index];
  if (value === undefined) throw new Error(`fixture has no element ${index}`);
  return value;
}
type Mutable = Record<string, unknown> & {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
};

describe("Atlas snapshot contract", () => {
  it("agrees with the Python owner on every shared case", async () => {
    for (const item of corpus.cases) {
      const payload = fixture(item.fixture);
      if (item.expectation === "accept") {
        const snapshot = take(parseAtlasSnapshot(payload));
        expect(await atlasSnapshotDigest(snapshot), item.id).toBe(item.digest);
        if (item.public_digest !== undefined) {
          const visible = projectAtlasPublic(snapshot);
          expect(await atlasSnapshotDigest(visible), item.id).toBe(item.public_digest);
          expect(visible.nodes.map((node) => node.id)).toEqual(item.public_node_ids);
        }
      } else {
        expect(refusal(payload), item.id).toEqual([item.code, item.path]);
      }
    }
  });

  it("normalises aliases, reverses verifies and refuses a bare contradiction", () => {
    const snapshot = take(parseAtlasSnapshot(fixture("valid")));
    expect(snapshot.edges.map((edge) => edge.kind)).toEqual(caseOf("valid").edge_kinds);
    const verified = snapshot.edges.filter((edge) => edge.kind === "verified_by");
    expect(verified.map((edge) => [edge.source, edge.target])).toEqual([
      ["definition:scpn_quantum_control.fixture/energy", "test:tests.fixture/test_energy"],
    ]);
    expect(refusal(fixture("refused_bare_contradiction"))[0]).toBe("refused_edge_kind");
  });

  it("keeps private material out of the public projection and its wire text", () => {
    const visible = projectAtlasPublic(take(parseAtlasSnapshot(fixture("valid"))));
    const wire = atlasSnapshotToWire(visible);
    expect(JSON.stringify(wire)).not.toContain("draft_notes");
    expect(take(parseAtlasSnapshot(wire))).toEqual(visible);
    expect(visible.accessScope).toBe("public");
  });

  it("round-trips the canonical wire document", async () => {
    const snapshot = take(parseAtlasSnapshot(fixture("valid_reordered")));
    const again = take(parseAtlasSnapshot(atlasSnapshotToWire(snapshot)));
    expect(again).toEqual(snapshot);
    expect(await atlasSnapshotDigest(again)).toBe(caseOf("valid").digest);
    expect(Object.isFrozen(snapshot)).toBe(true);
  });

  it("orders text by code point, as Python does", () => {
    const document = fixture("valid") as Mutable;
    document.exclusions = [
      { source_ref: "fixture/\u{1F600}", reason: "astral" },
      { source_ref: "fixture/～", reason: "basic plane" },
      { source_ref: "fixture/a", reason: "ascii" },
      { source_ref: "fixture/a", reason: "ascii, second" },
    ];
    const order = take(parseAtlasSnapshot(document)).exclusions.map((item) => item.reason);
    expect(order).toEqual(["ascii", "ascii, second", "basic plane", "astral"]);
  });

  it("maps identities and provenance consistently", () => {
    expect(Object.keys(atlasIdentityClasses).sort()).toEqual([...atlasNodeKinds].sort());
    expect(Object.keys(atlasEdgeEndpoints).sort()).toEqual([...atlasEdgeKinds].sort());
    expect(atlasProvRelations.produces).toBe("prov:wasGeneratedBy");
    expect(atlasProvClasses.run).toBe("prov:Activity");
  });

  it("refuses malformed values at their exact path", () => {
    const cases: [(document: Mutable) => void, string, string][] = [
      [(document) => (document.nodes = "x" as never), "invalid_type", "$.nodes"],
      [(document) => ((document.nodes as unknown[])[0] = "x"), "invalid_type", "$.nodes[0]"],
      [(document) => (at(document.nodes, 0).label = ""), "invalid_type", "$.nodes[0].label"],
      [
        (document) => (at(document.nodes, 0).kind = "module"),
        "unknown_node_kind",
        "$.nodes[0].kind",
      ],
      [
        (document) => (at(document.nodes, 0).id = "capability:no-separator"),
        "invalid_identity",
        "$.nodes[0].id",
      ],
      [
        (document) => (at(document.nodes, 3).id = "run:not-a-uuid/also-not"),
        "invalid_identity",
        "$.nodes[3].id",
      ],
      [
        (document) => (at(document.nodes, 0).access = "secret"),
        "unknown_access",
        "$.nodes[0].access",
      ],
      [
        (document) => (at(document.nodes, 0).status = "fresh"),
        "unknown_status",
        "$.nodes[0].status",
      ],
      [
        (document) => (at(document.nodes, 4).evidence_refs = "abc"),
        "invalid_type",
        "$.nodes[4].evidence_refs",
      ],
      [
        (document) => (at(document.nodes, 4).evidence_refs = ["ABC"]),
        "invalid_digest",
        "$.nodes[4].evidence_refs[0]",
      ],
      [
        (document) => (at(document.nodes, 8).validity_domain = ["units"]),
        "invalid_type",
        "$.nodes[8].validity_domain",
      ],
      [
        (document) =>
          ((at(document.nodes, 0).null_reasons as Record<string, string>).as_of = "as_of is set"),
        "invalid_null_reason",
        "$.nodes[0].null_reasons.as_of",
      ],
      [
        (document) => (at(document.nodes, 1).as_of = null),
        "missing_null_reason",
        "$.nodes[1].as_of",
      ],
      [(document) => (at(document.nodes, 0).colour = "red"), "unknown_field", "$.nodes[0].colour"],
      [(document) => delete at(document.edges, 0).as_of, "missing_field", "$.edges[0].as_of"],
      [
        (document) => (at(document.edges, 0).derivation_method = "guessed"),
        "unknown_derivation_method",
        "$.edges[0].derivation_method",
      ],
      [
        (document) => (at(document.edges, 3).evidence_digest = null),
        "missing_evidence",
        "$.edges[3].evidence_digest",
      ],
      [
        (document) => (at(document.edges, 0).assumptions = "none"),
        "invalid_type",
        "$.edges[0].assumptions",
      ],
      [(document) => (document.source_commit = "abc"), "invalid_commit", "$.source_commit"],
      [
        (document) => (document.evidence_set_hash = "0".repeat(63)),
        "invalid_digest",
        "$.evidence_set_hash",
      ],
      [
        (document) => (at(document.tombstones as Record<string, unknown>[], 0).kind = "module"),
        "unknown_node_kind",
        "$.tombstones[0].kind",
      ],
      [
        (document) =>
          (at(document.tombstones as Record<string, unknown>[], 0).id =
            "definition:no-owner-separator"),
        "invalid_identity",
        "$.tombstones[0].id",
      ],
      [
        (document) => (at(document.exclusions as Record<string, unknown>[], 0).reason = ""),
        "invalid_type",
        "$.exclusions[0].reason",
      ],
    ];
    for (const [mutate, code, path] of cases) {
      const document = fixture("valid") as Mutable;
      mutate(document);
      expect(refusal(document), path).toEqual([code, path]);
    }
    expect(refusal(["not", "an", "object"])).toEqual(["invalid_type", "$"]);
  });

  it("refuses duplicates and private relations in a public snapshot", () => {
    const nodes = fixture("valid") as Mutable;
    nodes.nodes.push(structuredClone(at(nodes.nodes, 0)));
    expect(refusal(nodes)[0]).toBe("duplicate_id");
    const tombstones = fixture("valid") as Mutable & { tombstones: unknown[] };
    tombstones.tombstones.push(structuredClone(at(tombstones.tombstones, 0)));
    expect(refusal(tombstones)[0]).toBe("duplicate_id");
    const edges = fixture("valid") as Mutable;
    edges.edges.push(structuredClone(at(edges.edges, 0)));
    expect(refusal(edges)[0]).toBe("duplicate_edge");
    const visible = fixture("valid") as Mutable;
    visible.nodes = visible.nodes.filter((node) => node.access === "public");
    visible.edges = visible.edges.filter((edge) => edge.access === "public");
    visible.access_scope = "public";
    at(visible.edges, 0).access = "private";
    expect(refusal(visible)).toEqual(["access_leak", "$.edges[0].access"]);
  });

  it("admits a mapping that carries units and semantics", () => {
    const document = fixture("maps_to_without_mapping") as Mutable;
    at(document.edges, document.edges.length - 1).mapping = {
      units: "hartree",
      semantics: "same estimand, other API",
    };
    const mapped = take(parseAtlasSnapshot(document)).edges.find((edge) => edge.kind === "maps_to");
    expect(mapped?.mapping).toEqual([
      ["semantics", "same estimand, other API"],
      ["units", "hartree"],
    ]);
  });

  it("rethrows a failure that is not a contract refusal", () => {
    const hostile = new Proxy(
      {},
      {
        ownKeys: () => {
          throw new TypeError("hostile object");
        },
      },
    );
    expect(() => parseAtlasSnapshot(hostile)).toThrow(TypeError);
  });
});
