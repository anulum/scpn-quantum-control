// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — admission of a sealed claim comparison record

import recordText from "../../../../data/studio/atlas_claim_comparisons_20261123.json?raw";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";

/** Schema `tools/studio_atlas_claim_comparison.py` writes. */
export const claimComparisonsSchema = "scpn_qc_studio_claim_comparisons_v1";
/** Domain of the record's canonical digest. */
export const claimComparisonsRecordSchema = "scpn_qc_studio_claim_comparisons_record_v1";
/** Statuses a comparison can have. */
export const comparisonStatuses = [
  "agreement",
  "disagreement",
  "non_comparable",
  "insufficient_evidence",
] as const;
/** One comparison status. */
export type ComparisonStatus = (typeof comparisonStatuses)[number];

/** One judged pair, as its source wrote it. */
export interface ClaimComparison {
  /** First value. */ readonly left: string;
  /** Second value. */ readonly right: string;
  /** Declared criterion. */ readonly criterion: string;
  /** The verdict. */ readonly status: ComparisonStatus;
  /** Why. */ readonly reasons: readonly string[];
  /** Whether the producers or versions differ. */ readonly federated: boolean;
  /** Whether either value is counter-evidence. */ readonly negative: boolean;
}
/** An admitted record. */
export interface ClaimComparisons {
  /** The local producer and version. */ readonly home: string;
  /** Producers and versions with accepted equivalence. */ readonly accepted: readonly string[];
  /** Each value's identifier with the source it cites. */ readonly sources: ReadonlyMap<
    string,
    string
  >;
  /** The judged pairs. */ readonly comparisons: readonly ClaimComparison[];
  /** The verified record digest. */ readonly recordDigest: string;
}
/** An admitted record, or why it was refused. */
export type ClaimComparisonsAdmission =
  | {
      /** The record was admitted. */ readonly ok: true;
      /** The record. */ readonly record: ClaimComparisons;
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
function flag(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Refusal(`${path} must be a boolean`);
  return value;
}
function identity(value: unknown, path: string): string {
  const item = object(value, path);
  return `${text(item["producer"], `${path}.producer`)} ${text(item["version"], `${path}.version`)}`;
}

/**
 * Admit a comparison record: schema, structure and recomputed record digest must all hold.
 *
 * @param source - The record's JSON text.
 * @returns The admitted record, or the reason it was refused.
 */
export async function admitClaimComparisons(source: string): Promise<ClaimComparisonsAdmission> {
  try {
    let parsed: unknown;
    try {
      parsed = readJson(source);
    } catch {
      throw new Refusal("it is not JSON");
    }
    const document = object(parsed, "$");
    if (document["schema"] !== claimComparisonsSchema)
      throw new Refusal("it has an unknown schema");
    const sources = new Map<string, string>();
    array(document["values"], "$.values").forEach((value, index) => {
      const item = object(value, `$.values[${index}]`);
      const cited = object(item["source"], `$.values[${index}].source`);
      sources.set(
        text(item["value_id"], `$.values[${index}].value_id`),
        text(cited["path"], `$.values[${index}].source.path`),
      );
    });
    const comparisons = array(document["comparisons"], "$.comparisons").map((value, index) => {
      const path = `$.comparisons[${index}]`;
      const item = object(value, path);
      const status = item["status"];
      if (!comparisonStatuses.includes(status as ComparisonStatus))
        throw new Refusal(`${path}.status is not one of ${comparisonStatuses.join(", ")}`);
      return {
        left: text(item["left"], `${path}.left`),
        right: text(item["right"], `${path}.right`),
        criterion: text(item["criterion"], `${path}.criterion`),
        status: status as ComparisonStatus,
        reasons: array(item["reasons"], `${path}.reasons`).map((reason, at) =>
          text(reason, `${path}.reasons[${at}]`),
        ),
        federated: flag(item["federated"], `${path}.federated`),
        negative: flag(item["negative"], `${path}.negative`),
      };
    });
    const { record_digest: recorded, ...body } = document;
    const digest = await canonicalDigest(claimComparisonsRecordSchema, body);
    if (digest !== recorded)
      throw new Refusal(`its canonical digest ${digest} differs from the recorded one`);
    return {
      ok: true,
      record: {
        home: identity(document["home"], "$.home"),
        accepted: array(document["accepted"], "$.accepted").map((item, index) =>
          identity(item, `$.accepted[${index}]`),
        ),
        sources,
        comparisons,
        recordDigest: digest,
      },
    };
  } catch (error) {
    return {
      ok: false,
      reason: `The claim comparison record was refused: ${(error as Error).message}.`,
    };
  }
}

/**
 * Admit the record committed with this build.
 *
 * @returns The admitted record, or why it was refused.
 */
export function admitCommittedClaimComparisons(): Promise<ClaimComparisonsAdmission> {
  return admitClaimComparisons(recordText);
}
