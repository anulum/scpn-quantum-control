// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — committed evidence lineage admission and record verification

import lineageText from "../../../../data/studio/evidence_lineage_20261123.json?raw";
import type { AtlasNode } from "../../shared/contracts/atlas";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";
import type { EvidenceVerification } from "../../shared/evidence/EvidenceInspector";

/** Schema of the committed lineage written by `tools/studio_evidence_lineage.py`. */
export const evidenceLineageSchema = "scpn_qc_studio_evidence_lineage_v1";
/** Canonical domain of each record's `record_digest`. */
export const lineageRecordSchema = "scpn_qc_studio_evidence_lineage_record_v1";

type Wire = Readonly<Record<string, unknown>>;

/** One committed file and its content digest. */
export interface LineageFile {
  /** Repository-relative path. */
  readonly path: string;
  /** Lowercase hexadecimal SHA-256 of the file. */
  readonly sha256: string;
}

/** One file of a hardware result pack. */
export interface LineagePackArtefact {
  /** Repository-relative path. */
  readonly path: string;
  /** Role the manifest gives the file, such as `raw_counts`. */
  readonly role: string;
  /** Digest the manifest records. */
  readonly recordedSha256: string;
  /** Whether the committed file matches it: `digest-matches`, `digest-differs` or `missing`. */
  readonly custody: string;
}

/** How one declared evidence identifier resolved against the committed data. */
export interface LineageEvidence {
  /** The declared identifier. */
  readonly artifactId: string;
  /** Ledger field that declared it: `evidence` or `benchmark`. */
  readonly role: string;
  /** Exactly one file, several, or none. */
  readonly state: "resolved" | "ambiguous" | "missing";
  /** Every committed file that carries the identifier, with its digest. */
  readonly candidates: readonly LineageFile[];
}

/** Fields every lineage record carries. */
interface LineageRecordBase {
  /** The record as committed, for recomputing its digest. */
  readonly wire: Wire;
  /** The digest the lineage builder recorded. */
  readonly recordDigest: string;
}

/** A claim with its source-owned statuses. */
export interface LineageClaim extends LineageRecordBase {
  /** Discriminator. */
  readonly kind: "claim";
  /** Ledger claim identifier. */
  readonly claimId: string;
  /** Ledger maturity status, verbatim. */
  readonly promotionStatus: string;
  /** The schema-B bundle the bundle owner emitted for the claim. */
  readonly bundle: unknown;
  /** The certificate record, or null with the reason none is committed. */
  readonly certificate: unknown;
  /** Why no certificate is committed, or null when one is. */
  readonly absentReason: string | null;
  /** Every declared evidence identifier and how it resolved. */
  readonly evidence: readonly LineageEvidence[];
}

/** A hardware result pack with its manifest status beside its bundle. */
export interface LineagePack extends LineageRecordBase {
  /** Discriminator. */
  readonly kind: "pack";
  /** Manifest pack identifier. */
  readonly packId: string;
  /** Manifest status, verbatim. */
  readonly manifestStatus: string;
  /** The schema-B bundle the bundle owner emitted for the pack. */
  readonly bundle: unknown;
  /** The pack's scope, verbatim. */
  readonly claimScope: string;
  /** What the pack does not claim. */
  readonly nonClaims: readonly string[];
  /** Provider job identifiers. */
  readonly jobIds: readonly string[];
  /** Each artefact with its role, recorded digest and committed-file custody. */
  readonly artefacts: readonly LineagePackArtefact[];
}

/** A replication analysis with its decision, verbatim. */
export interface LineageReplication extends LineageRecordBase {
  /** Discriminator. */
  readonly kind: "replication";
  /** Committed analysis path. */
  readonly path: string;
  /** Content digest of the analysis file. */
  readonly sha256: string;
  /** Campaign the analysis belongs to. */
  readonly campaign: string;
  /** Name of the decision field. */
  readonly decisionField: string;
  /** The decision, verbatim. */
  readonly decision: boolean;
}

/** One lineage record. */
export type LineageRecord = LineageClaim | LineagePack | LineageReplication;

/** An admitted lineage, indexed by the Atlas node each record belongs to. */
export interface EvidenceLineage {
  /** Claims by ledger identifier. */
  readonly claims: ReadonlyMap<string, LineageClaim>;
  /** Packs by manifest identifier. */
  readonly packs: ReadonlyMap<string, LineagePack>;
  /** Replications by analysis digest. */
  readonly replications: ReadonlyMap<string, LineageReplication>;
}

/** Admitted lineage, or the reason it was refused. */
export type LineageAdmission =
  | {
      /** The lineage was admitted. */
      readonly ok: true;
      /** The admitted lineage. */
      readonly lineage: EvidenceLineage;
    }
  | {
      /** The lineage was refused; nothing from it is shown. */
      readonly ok: false;
      /** Fixed explanation of the refusal. */
      readonly reason: string;
    };

class LineageRefusal extends Error {}

function object(value: unknown, at: string): Wire {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new LineageRefusal(`${at} must be an object`);
  return value as Wire;
}
function list(value: unknown, at: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new LineageRefusal(`${at} must be an array`);
  return value;
}
function text(value: unknown, at: string): string {
  if (typeof value !== "string" || value === "") throw new LineageRefusal(`${at} must be text`);
  return value;
}
function texts(value: unknown, at: string): readonly string[] {
  return list(value, at).map((item, index) => text(item, `${at}[${index}]`));
}
function status(bundle: unknown, at: string): string {
  return text(object(object(bundle, at)["claim_boundary"], `${at}.claim_boundary`)["status"], at);
}

function claimOf(value: unknown, at: string): LineageClaim {
  const wire = object(value, at);
  const certification = object(wire["certification"], `${at}.certification`);
  const certified = certification["state"] === "certified";
  if (!certified && certification["state"] !== "absent")
    throw new LineageRefusal(`${at}.certification has an unknown state`);
  const bundleStatus = status(wire["bundle"], `${at}.bundle`);
  // A validated status without its certificate is laundering; the record is refused.
  if (bundleStatus === "reference-validated" && !certified)
    throw new LineageRefusal(`${at} is reference-validated without a certificate`);
  const evidence = list(wire["evidence"], `${at}.evidence`).map((item, index) => {
    const where = `${at}.evidence[${index}]`;
    const record = object(item, where);
    const state = text(record["state"], `${where}.state`);
    if (state !== "resolved" && state !== "ambiguous" && state !== "missing")
      throw new LineageRefusal(`${where}.state is unknown`);
    return {
      artifactId: text(record["artifact_id"], `${where}.artifact_id`),
      role: text(record["role"], `${where}.role`),
      state,
      candidates: list(record["candidates"], `${where}.candidates`).map((candidate, place) => {
        const file = object(candidate, `${where}.candidates[${place}]`);
        return { path: text(file["path"], where), sha256: text(file["sha256"], where) };
      }),
    } satisfies LineageEvidence;
  });
  return {
    kind: "claim",
    wire,
    recordDigest: text(wire["record_digest"], `${at}.record_digest`),
    claimId: text(wire["claim_id"], `${at}.claim_id`),
    promotionStatus: text(wire["promotion_status"], `${at}.promotion_status`),
    bundle: wire["bundle"],
    certificate: certified ? object(certification["record"], `${at}.certification.record`) : null,
    absentReason: certified ? null : text(certification["reason"], `${at}.certification.reason`),
    evidence,
  };
}

function packOf(value: unknown, at: string): LineagePack {
  const wire = object(value, at);
  status(wire["bundle"], `${at}.bundle`);
  return {
    kind: "pack",
    wire,
    recordDigest: text(wire["record_digest"], `${at}.record_digest`),
    packId: text(wire["id"], `${at}.id`),
    manifestStatus: text(wire["manifest_status"], `${at}.manifest_status`),
    bundle: wire["bundle"],
    claimScope: text(wire["claim_scope"], `${at}.claim_scope`),
    nonClaims: texts(wire["non_claims"], `${at}.non_claims`),
    jobIds: texts(wire["job_ids"], `${at}.job_ids`),
    artefacts: list(wire["artefacts"], `${at}.artefacts`).map((item, index) => {
      const where = `${at}.artefacts[${index}]`;
      const artefact = object(item, where);
      return {
        path: text(artefact["path"], `${where}.path`),
        role: text(artefact["role"], `${where}.role`),
        recordedSha256: text(artefact["recorded_sha256"], `${where}.recorded_sha256`),
        custody: text(artefact["custody"], `${where}.custody`),
      };
    }),
  };
}

function replicationOf(value: unknown, at: string): LineageReplication {
  const wire = object(value, at);
  const decision = object(wire["decision"], `${at}.decision`);
  if (typeof decision["value"] !== "boolean")
    throw new LineageRefusal(`${at}.decision.value must be a boolean`);
  return {
    kind: "replication",
    wire,
    recordDigest: text(wire["record_digest"], `${at}.record_digest`),
    path: text(wire["path"], `${at}.path`),
    sha256: text(wire["sha256"], `${at}.sha256`),
    campaign: text(wire["campaign"], `${at}.campaign`),
    decisionField: text(decision["field"], `${at}.decision.field`),
    decision: decision["value"],
  };
}

/**
 * Admit a lineage: its schema and every record's shape must hold, and a claim may not be
 * reference-validated without its certificate.
 *
 * @param source - The lineage's JSON text.
 * @returns The admitted lineage, or the refusal reason.
 */
export function admitEvidenceLineage(source: string): LineageAdmission {
  try {
    const document = object(readJson(source), "$");
    if (document["schema"] !== evidenceLineageSchema)
      throw new LineageRefusal("the lineage has an unknown schema");
    const claims = list(document["claims"], "$.claims").map((item, index) =>
      claimOf(item, `$.claims[${index}]`),
    );
    const packs = list(document["packs"], "$.packs").map((item, index) =>
      packOf(item, `$.packs[${index}]`),
    );
    const replications = list(document["replications"], "$.replications").map((item, index) =>
      replicationOf(item, `$.replications[${index}]`),
    );
    return {
      ok: true,
      lineage: {
        claims: new Map(claims.map((claim) => [claim.claimId, claim])),
        packs: new Map(packs.map((pack) => [pack.packId, pack])),
        replications: new Map(replications.map((item) => [item.sha256, item])),
      },
    };
  } catch (error) {
    const reason = error instanceof LineageRefusal ? error.message : "the lineage is not JSON";
    return { ok: false, reason: `The evidence lineage was refused: ${reason}.` };
  }
}

/**
 * Admit the lineage committed with this build.
 *
 * @returns The admitted committed lineage, or the refusal reason.
 */
export function admitCommittedEvidenceLineage(): LineageAdmission {
  return admitEvidenceLineage(lineageText);
}

const claimPrefix = "claim:differentiable.claim_ledger/";
const packSource = "data/hardware_result_packs/manifest.json#";

/**
 * Return the lineage record an Atlas node belongs to.
 *
 * @param lineage - The admitted lineage.
 * @param node - An Atlas node of the committed snapshot.
 * @returns Its claim, pack or replication record, or null for a node without one.
 */
export function lineageRecordOf(lineage: EvidenceLineage, node: AtlasNode): LineageRecord | null {
  if (node.id.startsWith(claimPrefix))
    return lineage.claims.get(node.id.slice(claimPrefix.length)) ?? null;
  if (node.sourceRef.startsWith(packSource))
    return lineage.packs.get(node.sourceRef.slice(packSource.length)) ?? null;
  return node.sourceDigest === null ? null : (lineage.replications.get(node.sourceDigest) ?? null);
}

/**
 * Recompute a record's canonical digest and compare it with the recorded one.
 *
 * The check proves that the record shown is the record the lineage builder wrote. It says
 * nothing about the science the record describes.
 *
 * @param record - A lineage record.
 * @returns `match` or `mismatch`, with what was checked.
 */
export async function verifyLineageRecord(record: LineageRecord): Promise<EvidenceVerification> {
  const { record_digest: _recorded, ...body } = record.wire;
  const digest = await canonicalDigest(lineageRecordSchema, body);
  return digest === record.recordDigest
    ? { display: "match", detail: `The lineage record's canonical digest ${digest} matches.` }
    : {
        display: "mismatch",
        detail: `The lineage record's canonical digest ${digest} differs from ${record.recordDigest}.`,
      };
}
