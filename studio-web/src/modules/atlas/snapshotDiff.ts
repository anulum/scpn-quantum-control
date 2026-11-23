// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — admission of an Atlas snapshot diff record

import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";

/** Schema of the record `tools/studio_atlas_snapshot_diff.py` writes. */
export const snapshotDiffSchema = "scpn_qc_studio_atlas_snapshot_diff_v1";
/** Domain of the record's canonical digest. */
export const snapshotDiffRecordSchema = "scpn_qc_studio_atlas_snapshot_diff_record_v1";

const nodeChanges = ["added", "removed", "tombstoned", "changed"] as const;
const edgeChanges = ["added", "removed", "changed"] as const;
const categories = ["code", "data", "contract", "documentation", "calibration"] as const;
const hex64 = /^[0-9a-f]{64}$/;
const hex40 = /^[0-9a-f]{40}$/;

/** One snapshot the record compares. */
export interface DiffSide {
  /** The snapshot's canonical digest. */ readonly snapshotDigest: string;
  /** The commit its source tree is based on. */ readonly sourceCommit: string;
}
/** One node that differs. */
export interface DiffNode {
  /** Node identifier. */ readonly id: string;
  /** Node kind. */ readonly kind: string;
  /** How it differs. */ readonly change: (typeof nodeChanges)[number];
  /** Change category. */ readonly category: (typeof categories)[number];
  /** Fields that differ, for a change. */ readonly fields: readonly string[];
  /** The tombstone's reason and time, for a tombstoned node. */ readonly reason: string | null;
}
/** One relation that differs. */
export interface DiffEdge {
  /** Source node. */ readonly source: string;
  /** Relation kind. */ readonly kind: string;
  /** Target node. */ readonly target: string;
  /** How it differs. */ readonly change: (typeof edgeChanges)[number];
}
/** One relation of an impact path. */
export interface DiffStep {
  /** Source node. */ readonly source: string;
  /** Relation kind. */ readonly kind: string;
  /** Target node. */ readonly target: string;
}
/** A node a change can reach. */
export interface DiffImpact {
  /** The impacted node. */ readonly id: string;
  /** The changed node the path starts from. */ readonly origin: string;
  /** The relations followed. */ readonly path: readonly DiffStep[];
}
/** An admitted diff record, every value as its source wrote it. */
export interface SnapshotDiff {
  /** The earlier snapshot. */ readonly before: DiffSide;
  /** The later snapshot. */ readonly after: DiffSide;
  /** Differing nodes. */ readonly nodes: readonly DiffNode[];
  /** Differing relations. */ readonly edges: readonly DiffEdge[];
  /** Count of unchanged nodes. */ readonly unchanged: bigint;
  /** Potential impact. */ readonly impact: readonly DiffImpact[];
  /** Whether the analysis rests only on observed relations of complete extractions. */ readonly complete: boolean;
  /** Why it is incomplete. */ readonly incompleteReasons: readonly string[];
  /** The verified record digest. */ readonly recordDigest: string;
}
/** An admitted record, or why it was refused. */
export type SnapshotDiffAdmission =
  | {
      /** The record was admitted. */ readonly ok: true;
      /** The record. */ readonly diff: SnapshotDiff;
    }
  | {
      /** The record was refused; nothing from it is shown. */ readonly ok: false;
      /** Fixed explanation of the refusal. */ readonly reason: string;
    };

type Json = Readonly<Record<string, unknown>>;

class Refusal extends Error {}

function object(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Refusal(`${path} must be an object`);
  return value as Json;
}
function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new Refusal(`${path} must be an array`);
  return value;
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value === "") throw new Refusal(`${path} must be text`);
  return value;
}
function match(value: unknown, pattern: RegExp, path: string): string {
  const string = text(value, path);
  if (!pattern.test(string)) throw new Refusal(`${path} is not a lowercase hex digest`);
  return string;
}
function oneOf<T extends string>(value: unknown, choices: readonly T[], path: string): T {
  if (!choices.includes(value as T))
    throw new Refusal(`${path} is not one of ${choices.join(", ")}`);
  return value as T;
}
function texts(value: unknown, path: string): readonly string[] {
  return array(value, path).map((item, index) => text(item, `${path}[${index}]`));
}
function side(value: unknown, path: string): DiffSide {
  const record = object(value, path);
  return {
    snapshotDigest: match(record["snapshot_digest"], hex64, `${path}.snapshot_digest`),
    sourceCommit: match(record["source_commit"], hex40, `${path}.source_commit`),
  };
}
function step(value: unknown, path: string): DiffStep {
  const record = object(value, path);
  return {
    source: text(record["source"], `${path}.source`),
    kind: text(record["kind"], `${path}.kind`),
    target: text(record["target"], `${path}.target`),
  };
}

function parse(document: Json): Omit<SnapshotDiff, "recordDigest"> {
  if (document["schema"] !== snapshotDiffSchema)
    throw new Refusal("the record has an unknown schema");
  const diff = object(document["diff"], "$.diff");
  const unchanged = diff["unchanged"];
  if (typeof unchanged !== "bigint" || unchanged < 0n)
    throw new Refusal("$.diff.unchanged must be a non-negative integer");
  if (typeof diff["complete"] !== "boolean") throw new Refusal("$.diff.complete must be a boolean");
  return {
    before: side(document["before"], "$.before"),
    after: side(document["after"], "$.after"),
    nodes: array(diff["nodes"], "$.diff.nodes").map((value, index) => {
      const path = `$.diff.nodes[${index}]`;
      const node = object(value, path);
      const reason = node["reason"];
      return {
        id: text(node["id"], `${path}.id`),
        kind: text(node["kind"], `${path}.kind`),
        change: oneOf(node["change"], nodeChanges, `${path}.change`),
        category: oneOf(node["category"], categories, `${path}.category`),
        fields: texts(node["fields"], `${path}.fields`),
        reason: reason === null ? null : text(reason, `${path}.reason`),
      };
    }),
    edges: array(diff["edges"], "$.diff.edges").map((value, index) => {
      const path = `$.diff.edges[${index}]`;
      const edge = object(value, path);
      return { ...step(edge, path), change: oneOf(edge["change"], edgeChanges, `${path}.change`) };
    }),
    unchanged,
    impact: array(diff["impact"], "$.diff.impact").map((value, index) => {
      const path = `$.diff.impact[${index}]`;
      const item = object(value, path);
      return {
        id: text(item["id"], `${path}.id`),
        origin: text(item["origin"], `${path}.origin`),
        path: array(item["path"], `${path}.path`).map((entry, position) =>
          step(entry, `${path}.path[${position}]`),
        ),
      };
    }),
    complete: diff["complete"],
    incompleteReasons: texts(diff["incomplete_reasons"], "$.diff.incomplete_reasons"),
  };
}

/**
 * Admit a diff record: schema, structure and recomputed record digest must all hold.
 *
 * @param source - The record's JSON text.
 * @returns The admitted record, or the reason it was refused.
 */
export async function admitSnapshotDiff(source: string): Promise<SnapshotDiffAdmission> {
  let document: Json;
  try {
    document = object(readJson(source), "$");
  } catch (error) {
    return {
      ok: false,
      reason: `The snapshot comparison was refused: ${error instanceof Refusal ? error.message : "it is not JSON"}.`,
    };
  }
  try {
    const parsed = parse(document);
    const { record_digest: recorded, ...body } = document;
    const digest = await canonicalDigest(snapshotDiffRecordSchema, body);
    if (digest !== match(recorded, hex64, "$.record_digest"))
      throw new Refusal(`its canonical digest ${digest} differs from the recorded one`);
    return { ok: true, diff: { ...parsed, recordDigest: digest } };
  } catch (error) {
    return {
      ok: false,
      reason: `The snapshot comparison was refused: ${(error as Error).message}.`,
    };
  }
}
