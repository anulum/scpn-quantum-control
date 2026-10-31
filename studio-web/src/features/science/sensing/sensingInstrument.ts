// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — sensing and entropy metadata projection

import committed from "../../../../../data/studio/sensing_instrument_20261031.json";
import type { Loaded } from "../../../shared/browser/data";
import { count, finite, flag, list, record, text } from "../../../shared/browser/jsonFields";
import type { EvidenceView } from "../../../shared/evidence/evidence";

/** A source-owned Gaussian observation design; the browser runs no solver. */
export interface SensingCandidate {
  /** Source observation identifier. */ readonly id: string;
  /** Declared shot count. */ readonly shots: number;
  /** Source-model prior variance. */ readonly prior: number;
  /** Source-model observation sensitivity. */ readonly sensitivity: number;
  /** Source-model per-shot noise variance. */ readonly noise: number;
  /** Expected information in nats. */ readonly gain: number;
  /** Model posterior variance, not calibrated empirical uncertainty. */ readonly posterior: number;
}
/** Source declarations about a byte block; sample bytes and secrets are never retained. */
export interface EntropyMetadata {
  /** Stable record selector. */ readonly id: string;
  /** Declared origin, never inferred from a signature or diagnostic. */ readonly kind:
    | "synthetic"
    | "device"
    | "unknown";
  /** Public origin reference or missing provenance. */ readonly source: string | null;
  /** Declared extractor or missing provenance. */ readonly extractor: string | null;
  /** Source-model assumptions or missing provenance. */ readonly assumptions: string | null;
  /** Identity of the assessed bytes, when available. */ readonly digest: string | null;
  /** Recorded diagnostic results, never independent browser verification. */ readonly diagnostics: {
    /** Frequency-test outcome. */ readonly frequency: boolean;
    /** Runs-test outcome. */ readonly runs: boolean;
    /** Producer's signature-check outcome. */ readonly signature: boolean;
  } | null;
  /** Fixed source-owner claim boundary. */ readonly boundary: string;
}
/** Bounded source-owned designs and entropy metadata. */
export interface SensingInstrument {
  /** Source artefact identity. */ readonly id: string;
  /** Limits of the measurement-design claims. */ readonly designBoundary: string;
  /** Declared statistical observation model. */ readonly model: string;
  /** Source candidate order. */ readonly candidates: readonly SensingCandidate[];
  /** Source record order. */ readonly entropy: readonly EntropyMetadata[];
}

const entropyBoundary = text(committed.entropy[0]?.claim_boundary, "committed entropy boundary");

function nullable(value: unknown, field: string): string | null {
  if (value === null) return null;
  const result = text(value, field);
  if (result.trim().length === 0) throw new Error(`${field} must be nonblank text or null`);
  return result;
}

function entropyMetadata(raw: unknown): EntropyMetadata {
  const row = record(raw, "entropy");
  const kind = row["source_kind"];
  if (kind !== "synthetic" && kind !== "device" && kind !== "unknown")
    throw new Error("source_kind must be synthetic, device or unknown");
  if (row["source_entropy_qualified"] !== false)
    throw new Error("Statistical tests and signatures cannot qualify source entropy");
  if (row["claim_boundary"] !== entropyBoundary)
    throw new Error("Entropy claim boundary differs from its source owner");
  const digest = nullable(row["payload_sha256"], "payload_sha256");
  if (digest !== null && !/^[a-f0-9]{64}$/.test(digest)) throw new Error("Invalid SHA-256");
  const diagnostics =
    row["diagnostics"] === null ? null : record(row["diagnostics"], "diagnostics");
  if (diagnostics !== null && digest === null) throw new Error("Diagnostics need a byte digest");
  return {
    id: text(row["case_id"], "case_id"),
    kind,
    source: nullable(row["source"], "source"),
    extractor: nullable(row["extractor"], "extractor"),
    assumptions: nullable(row["assumptions"], "assumptions"),
    digest,
    diagnostics:
      diagnostics === null
        ? null
        : {
            frequency: flag(diagnostics["frequency_passed"], "frequency_passed"),
            runs: flag(diagnostics["runs_passed"], "runs_passed"),
            signature: flag(diagnostics["signature_valid"], "signature_valid"),
          },
    boundary: entropyBoundary,
  };
}

/**
 * Parse source metadata, refusing unsupported schemas and entropy promotion.
 * @param raw - Decoded source JSON; extra fields are discarded.
 * @returns Allowlisted metadata or a visible refusal.
 */
export function parseSensingInstrument(raw: unknown): Loaded<SensingInstrument> {
  try {
    const row = record(raw, "instrument");
    if (row["schema"] !== committed.schema)
      throw new Error("Unsupported sensing instrument schema");
    const candidates = list(row["candidates"], "candidates").map((rawCandidate) => {
      const candidate = record(rawCandidate, "candidate");
      return {
        id: text(candidate["observable_id"], "observable_id"),
        shots: count(candidate["shots"], "shots"),
        prior: finite(candidate["prior_variance"], "prior_variance"),
        sensitivity: finite(candidate["sensitivity"], "sensitivity"),
        noise: finite(candidate["noise_variance"], "noise_variance"),
        gain: finite(candidate["expected_information_gain_nats"], "expected_information_gain_nats"),
        posterior: finite(candidate["posterior_variance"], "posterior_variance"),
      };
    });
    for (const item of candidates) {
      if (
        item.shots === 0 ||
        item.prior <= 0 ||
        item.noise <= 0 ||
        item.gain < 0 ||
        item.posterior <= 0 ||
        item.posterior > item.prior
      )
        throw new Error("Invalid Gaussian observation-model bounds");
    }
    const entropy = list(row["entropy"], "entropy").map(entropyMetadata);
    for (const ids of [candidates.map((item) => item.id), entropy.map((item) => item.id)]) {
      if (ids.length === 0 || new Set(ids).size !== ids.length)
        throw new Error("Records need nonempty unique identifiers");
    }
    return {
      ok: true,
      value: {
        id: text(row["artifact_id"], "artifact_id"),
        designBoundary: text(row["design_boundary"], "design_boundary"),
        model: text(row["design_model"], "design_model"),
        candidates,
        entropy,
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/**
 * Export only public metadata; raw input objects are never serialised.
 * @param item - A parsed source record.
 * @returns JSON containing no private fields and no entropy certification.
 */
export function exportEntropyMetadata(item: EntropyMetadata): string {
  return JSON.stringify(
    {
      case_id: item.id,
      source_kind: item.kind,
      source: item.source,
      extractor: item.extractor,
      assumptions: item.assumptions,
      payload_sha256: item.digest,
      diagnostics:
        item.diagnostics === null
          ? null
          : {
              frequency_passed: item.diagnostics.frequency,
              runs_passed: item.diagnostics.runs,
              signature_valid: item.diagnostics.signature,
            },
      source_entropy_qualified: false,
      claim_boundary: item.boundary,
    },
    null,
    2,
  );
}

/**
 * Reuse the shared evidence inspector without declaring an unexported seal verified.
 * @param item - Parsed public record.
 * @returns Independent evidence axes; missing provenance stays visible.
 */
export function entropyEvidenceView(item: EntropyMetadata): EvidenceView {
  return {
    snapshotIdentity: exportEntropyMetadata(item),
    schema: committed.schema,
    source: item.source,
    digest: item.digest,
    kind: item.kind,
    claim: "source entropy unqualified",
    admission: "Metadata inspection only",
    boundary: item.boundary,
    freshness: null,
    substrate: item.kind,
    exactness: null,
    provenance: item.assumptions,
    seal: "missing",
    issues:
      item.source === null
        ? ["Missing source provenance; no device observation is established"]
        : [],
    identifiable: item.digest !== null,
  };
}

/** Source-owned, committed metadata parsed once at the boundary. */
export const sensingInstrument = parseSensingInstrument(committed);
