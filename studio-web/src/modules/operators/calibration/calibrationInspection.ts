// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — admission of a sealed calibration inspection record

import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson } from "../../../shared/contracts/jsonTransport";

/** Schema `tools/studio_calibration_inspection.py` writes. */
export const calibrationInspectionSchema = "scpn_qc_studio_calibration_inspection_v1";
/** Domain of the record's canonical digest. */
export const calibrationInspectionRecordSchema = "scpn_qc_studio_calibration_inspection_record_v1";

/** One element reading; `null` means the snapshot did not report it. */
export interface Reading {
  /** `q<index>` or `c<a>-<b>`. */ readonly element: string;
  /** The verbatim value, or unknown. */ readonly value: number | null;
}
/** One dated snapshot with its element map, as the source wrote it. */
export interface InspectedSnapshot {
  /** Declared target. */ readonly targetId: string;
  /** Committed file. */ readonly path: string;
  /** File digest. */ readonly sha256: string;
  /** The snapshot's own source text. */ readonly source: string;
  /** Capture date, date precision. */ readonly observedOn: string;
  /** Provider calibration set. */ readonly calibrationSetId: string;
  /** What coupler values mean. */ readonly fidelitySemantics: string;
  /** Per-qubit readout error. */ readonly qubits: readonly Reading[];
  /** Per-coupler CZ fidelity. */ readonly couplers: readonly Reading[];
  /** The next calibration set of the target, or null when current. */ readonly supersededBy:
    | string
    | null;
  /** Plans that name this calibration set. */ readonly referencedBy: readonly string[];
}
/** One element that differs between consecutive snapshots. */
export interface DriftChange {
  /** The element. */ readonly element: string;
  /** `readout_error` or `cz_fidelity`. */ readonly quantity: string;
  /** Earlier value or unknown. */ readonly before: number | null;
  /** Later value or unknown. */ readonly after: number | null;
  /** How it differs. */ readonly change: string;
}
/** Exact differences between two consecutive snapshots of one target. */
export interface InspectedDrift {
  /** Earlier calibration set. */ readonly before: string;
  /** Later calibration set. */ readonly after: string;
  /** Whether the calibration set changed. */ readonly calibrationSetChanged: boolean;
  /** Elements that read the same. */ readonly unchanged: number;
  /** Elements that differ. */ readonly changes: readonly DriftChange[];
}
/** One plan's verdict, judged as of its own block by the source. */
export interface InspectedPlan {
  /** Plan identifier. */ readonly planId: string;
  /** Experiment type. */ readonly experimentType: string;
  /** Block date. */ readonly blockOn: string;
  /** The set the plan names, never replaced. */ readonly calibrationSetId: string | null;
  /** `admitted` or `blocked`. */ readonly status: "admitted" | "blocked";
  /** Rules that held, or every reason it is blocked. */ readonly reasons: readonly string[];
}
/** One declared freshness policy with its citation. */
export interface InspectedPolicy {
  /** Experiment type. */ readonly experimentType: string;
  /** Cited document and section. */ readonly citation: string;
  /** Rule kinds, in order. */ readonly rules: readonly string[];
}
/** An admitted record. */
export interface CalibrationInspection {
  /** Declared policies. */ readonly policies: readonly InspectedPolicy[];
  /** Snapshots ordered by target and date. */ readonly snapshots: readonly InspectedSnapshot[];
  /** Consecutive drifts. */ readonly drifts: readonly InspectedDrift[];
  /** Judged plans. */ readonly plans: readonly InspectedPlan[];
  /** Verified record digest. */ readonly recordDigest: string;
}
/** An admitted record, or why it was refused. */
export type CalibrationInspectionAdmission =
  | {
      /** The record was admitted. */ readonly ok: true;
      /** The record. */ readonly record: CalibrationInspection;
    }
  | {
      /** The record was refused; nothing from it is shown. */ readonly ok: false;
      /** Fixed explanation. */ readonly reason: string;
    };
/** What the record says about one calibration reference. */
export type ReferenceStatus =
  | {
      /** Not a snapshot of this record. */ readonly kind: "unknown";
    }
  | {
      /** Current or superseded snapshot. */ readonly kind: "current" | "superseded";
      /** The snapshot. */ readonly snapshot: InspectedSnapshot;
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
function nullableText(value: unknown, path: string): string | null {
  return value === null ? null : text(value, path);
}
function reading(value: unknown, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Refusal(`${path} must be a number or null`);
  return value;
}
function texts(value: unknown, path: string): readonly string[] {
  return array(value, path).map((item, index) => text(item, `${path}[${index}]`));
}

function snapshotOf(value: unknown, path: string): InspectedSnapshot {
  const item = object(value, path);
  return {
    targetId: text(item["target_id"], `${path}.target_id`),
    path: text(item["path"], `${path}.path`),
    sha256: text(item["sha256"], `${path}.sha256`),
    source: text(item["source"], `${path}.source`),
    observedOn: text(item["observed_on"], `${path}.observed_on`),
    calibrationSetId: text(item["calibration_set_id"], `${path}.calibration_set_id`),
    fidelitySemantics: text(item["fidelity_semantics"], `${path}.fidelity_semantics`),
    qubits: array(item["qubits"], `${path}.qubits`).map((entry, index) => {
      const qubit = object(entry, `${path}.qubits[${index}]`);
      return {
        element: `q${String(qubit["qubit"])}`,
        value: reading(qubit["readout_error"], `${path}.qubits[${index}].readout_error`),
      };
    }),
    couplers: array(item["couplers"], `${path}.couplers`).map((entry, index) => {
      const coupler = object(entry, `${path}.couplers[${index}]`);
      const pair = array(coupler["coupler"], `${path}.couplers[${index}].coupler`);
      return {
        element: `c${pair.join("-")}`,
        value: reading(coupler["cz_fidelity"], `${path}.couplers[${index}].cz_fidelity`),
      };
    }),
    supersededBy: nullableText(item["superseded_by"], `${path}.superseded_by`),
    referencedBy: texts(item["referenced_by"], `${path}.referenced_by`),
  };
}

function driftOf(value: unknown, path: string): InspectedDrift {
  const item = object(value, path);
  if (typeof item["calibration_set_changed"] !== "boolean")
    throw new Refusal(`${path}.calibration_set_changed must be a boolean`);
  // The transport keeps JSON integers exact as bigint.
  const unchanged = item["unchanged"];
  if (typeof unchanged !== "bigint" || unchanged < 0n)
    throw new Refusal(`${path}.unchanged must be a non-negative integer`);
  return {
    before: text(item["before"], `${path}.before`),
    after: text(item["after"], `${path}.after`),
    calibrationSetChanged: item["calibration_set_changed"],
    unchanged: Number(unchanged),
    changes: array(item["changes"], `${path}.changes`).map((entry, index) => {
      const change = object(entry, `${path}.changes[${index}]`);
      return {
        element: text(change["element"], `${path}.changes[${index}].element`),
        quantity: text(change["quantity"], `${path}.changes[${index}].quantity`),
        before: reading(change["before"], `${path}.changes[${index}].before`),
        after: reading(change["after"], `${path}.changes[${index}].after`),
        change: text(change["change"], `${path}.changes[${index}].change`),
      };
    }),
  };
}

function planOf(value: unknown, path: string): InspectedPlan {
  const item = object(value, path);
  const status = item["status"];
  if (status !== "admitted" && status !== "blocked")
    throw new Refusal(`${path}.status must be admitted or blocked`);
  return {
    planId: text(item["plan_id"], `${path}.plan_id`),
    experimentType: text(item["experiment_type"], `${path}.experiment_type`),
    blockOn: text(item["block_on"], `${path}.block_on`),
    calibrationSetId: nullableText(item["calibration_set_id"], `${path}.calibration_set_id`),
    status,
    reasons: texts(item["reasons"], `${path}.reasons`),
  };
}

function policyOf(value: unknown, path: string): InspectedPolicy {
  const item = object(value, path);
  const citation = object(item["citation"], `${path}.citation`);
  return {
    experimentType: text(item["experiment_type"], `${path}.experiment_type`),
    citation: `${text(citation["path"], `${path}.citation.path`)} § ${text(citation["section"], `${path}.citation.section`)}`,
    rules: array(item["rules"], `${path}.rules`).map((rule, index) =>
      text(object(rule, `${path}.rules[${index}]`)["kind"], `${path}.rules[${index}].kind`),
    ),
  };
}

/**
 * Admit an inspection record: schema, structure and recomputed record digest must all hold.
 *
 * @param source - The record's JSON text.
 * @returns The admitted record, or the reason it was refused.
 */
export async function admitCalibrationInspection(
  source: string,
): Promise<CalibrationInspectionAdmission> {
  try {
    let parsed: unknown;
    try {
      parsed = readJson(source);
    } catch {
      throw new Refusal("it is not JSON");
    }
    const document = object(parsed, "$");
    if (document["schema"] !== calibrationInspectionSchema)
      throw new Refusal("it has an unknown schema");
    const record = {
      policies: array(document["policies"], "$.policies").map((item, index) =>
        policyOf(item, `$.policies[${index}]`),
      ),
      snapshots: array(document["snapshots"], "$.snapshots").map((item, index) =>
        snapshotOf(item, `$.snapshots[${index}]`),
      ),
      drifts: array(document["drifts"], "$.drifts").map((item, index) =>
        driftOf(item, `$.drifts[${index}]`),
      ),
      plans: array(document["plans"], "$.plans").map((item, index) =>
        planOf(item, `$.plans[${index}]`),
      ),
    };
    const { record_digest: recorded, ...body } = document;
    const digest = await canonicalDigest(calibrationInspectionRecordSchema, body);
    if (digest !== recorded)
      throw new Refusal(`its canonical digest ${digest} differs from the recorded one`);
    return { ok: true, record: { ...record, recordDigest: digest } };
  } catch (error) {
    return {
      ok: false,
      reason: `The calibration inspection record was refused: ${(error as Error).message}.`,
    };
  }
}

/**
 * Look up a calibration reference by its file digest; the record, not the view, says whether it was superseded.
 *
 * @param record - The admitted record.
 * @param sha256 - Digest a dossier or profile binds as its calibration.
 * @returns Whether the digest names a current or superseded snapshot, or none.
 */
export function referenceStatus(record: CalibrationInspection, sha256: string): ReferenceStatus {
  const snapshot = record.snapshots.find((item) => item.sha256 === sha256);
  if (snapshot === undefined) return { kind: "unknown" };
  return { kind: snapshot.supersededBy === null ? "current" : "superseded", snapshot };
}
