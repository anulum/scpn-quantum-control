// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — deterministic Atlas graph tiers for the resource budgets

import { atlasSnapshotPinSchema } from "../src/modules/atlas/atlasSnapshot";
import {
  type AtlasSnapshot,
  atlasSnapshotDigest,
  parseAtlasSnapshot,
} from "../src/shared/contracts/atlas";

/** One graph tier of the release budgets. */
export interface AtlasTier {
  /** Nodes. */ readonly nodes: number;
  /** Relations; each node points to the next `relations / nodes` nodes, so none repeats. */
  readonly relations: number;
}

/** The budget tiers: 100, 1 000 and 10 000 nodes, up to 100 000 relations. */
export const atlasTiers: readonly AtlasTier[] = [
  { nodes: 100, relations: 300 },
  { nodes: 1_000, relations: 3_000 },
  { nodes: 10_000, relations: 30_000 },
  { nodes: 10_000, relations: 100_000 },
];

/** Identifier of tier node `index`. */
export function tierNodeId(index: number): string {
  return `definition:tier/n${String(index).padStart(5, "0")}`;
}

/**
 * The snapshot document of one tier: the same input always gives the same document.
 *
 * @param tier - Node and relation counts; relations must be a multiple of nodes, below nodes squared.
 * @returns The wire snapshot.
 */
export function tierSnapshotDocument(tier: AtlasTier): Record<string, unknown> {
  const fanOut = tier.relations / tier.nodes;
  if (!Number.isInteger(fanOut) || fanOut < 0 || fanOut >= tier.nodes)
    throw new Error("relations must be a multiple of nodes, below nodes squared");
  const nodes = Array.from({ length: tier.nodes }, (_unused, index) => ({
    id: tierNodeId(index),
    kind: "definition",
    label: `Tier node ${index}`,
    source_ref: `tier/n${index}`,
    source_digest: null,
    validity_domain: null,
    as_of: null,
    access: "public",
    status: "current",
    evidence_refs: [],
    null_reasons: { source_digest: "tier", validity_domain: "tier", as_of: "tier" },
  }));
  const edges = Array.from({ length: tier.relations }, (_unused, at) => {
    const source = Math.floor(at / fanOut);
    const target = (source + 1 + (at % fanOut)) % tier.nodes;
    return {
      source: tierNodeId(source),
      kind: "requires",
      target: tierNodeId(target),
      relation_version: "atlas_relation.v1",
      derivation_method: "declared",
      source_ref: "tier/edges",
      evidence_digest: null,
      assumptions: [],
      mapping: null,
      as_of: null,
      access: "public",
    };
  });
  return {
    schema: "quantum_atlas_snapshot.v1",
    extractor_version: "tier",
    source_commit: "0".repeat(40),
    evidence_set_hash: "a".repeat(64),
    access_scope: "public",
    nodes,
    edges,
    tombstones: [],
    exclusions: [],
  };
}

/**
 * The parsed snapshot of one tier.
 *
 * @param tier - The tier.
 * @returns The snapshot the Atlas query owner reads.
 */
export function tierSnapshot(tier: AtlasTier): AtlasSnapshot {
  const parsed = parseAtlasSnapshot(tierSnapshotDocument(tier));
  if (!parsed.ok) throw new Error(`${parsed.code} at ${parsed.path}`);
  return parsed.value;
}

/**
 * The pin text of one tier, with its recomputed digest, as the Atlas opens a file.
 *
 * @param tier - The tier.
 * @returns The pin's text and its snapshot digest.
 */
export async function tierPin(tier: AtlasTier): Promise<{ text: string; digest: string }> {
  const document = tierSnapshotDocument(tier);
  const digest = await atlasSnapshotDigest(tierSnapshot(tier));
  return {
    text: JSON.stringify({
      schema: atlasSnapshotPinSchema,
      snapshot: document,
      snapshot_digest: digest,
    }),
    digest,
  };
}
