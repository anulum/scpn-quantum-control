// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — committed attestation cases and their Python verdicts

import committedText from "../../../../data/studio/attestation_instrument_20261115.json?raw";
import type { Loaded } from "../../shared/browser/data";
import { flag, list, record, text } from "../../shared/browser/jsonFields";
import { readJson } from "../../shared/contracts/jsonTransport";
import { readEnvelope } from "./attestationVerifier";
import type {
  AttestationStatus,
  AttestationVerdict,
  AttestedEnvelope,
} from "./attestationVerifier";

/** Schema of the committed instrument data. */
export const attestationInstrumentSchema = "scpn_qc_studio_attestation_instrument_v1";

/** One envelope with the verdict the Python owner recorded for it. */
export interface AttestationCase {
  /** Case identifier. */ readonly id: string;
  /** Short title. */ readonly title: string;
  /** What was changed after sealing, or `none`. */ readonly change: string;
  /** Whether the case is verified with a required signature. */ readonly requireSignature: boolean;
  /** The envelope as committed. */ readonly envelope: AttestedEnvelope;
  /** The Python owner's verdict. */ readonly python: AttestationVerdict;
}

/** The owner's canonical text and digest for one content. */
export interface OracleRow {
  /** What the row exercises. */ readonly label: string;
  /** Content hashed with empty claim axes. */ readonly content: Record<string, unknown>;
  /** Exact canonical text written by the owner. */ readonly pythonText: string;
  /** The owner's digest of that text. */ readonly digest: string;
}

/** The pinned summary every case was sealed over. */
export interface AttestationSource {
  /** Repository path. */ readonly path: string;
  /** Pinned SHA-256 of the file. */ readonly sha256: string;
  /** Reported backend. */ readonly backend: string;
  /** Scientific status of the values. */ readonly status: string;
}

/** Committed attestation instrument data. */
export interface AttestationInstrument {
  /** Artefact identifier. */ readonly id: string;
  /** Python verifier that decided the recorded verdicts. */ readonly verifier: string;
  /** Envelope schema of the owner. */ readonly envelopeSchema: string;
  /** The owner's claim boundary. */ readonly claimBoundary: string;
  /** What VERIFIED does and does not mean here. */ readonly boundary: string;
  /** Pinned source summary. */ readonly source: AttestationSource;
  /** Cases in display order. */ readonly cases: readonly AttestationCase[];
  /** Canonicalisation oracle. */ readonly oracle: readonly OracleRow[];
  /** Code points Python's `str.strip()` removes. */ readonly whitespace: ReadonlySet<number>;
}

const statuses: readonly AttestationStatus[] = ["VERIFIED", "STRIPPED", "FORGED", "UNGRADED"];

function verdict(raw: unknown): AttestationVerdict {
  const row = record(raw, "python_verdict");
  const status = text(row["status"], "status") as AttestationStatus;
  if (!statuses.includes(status)) throw new Error(`Unknown attestation status ${status}`);
  return {
    status,
    reason: text(row["reason"], "reason"),
    expectedDigest:
      row["expected_digest"] === "" ? "" : text(row["expected_digest"], "expected_digest"),
    observedDigest:
      row["observed_digest"] === "" ? "" : text(row["observed_digest"], "observed_digest"),
    claimId: text(row["claim_id"], "claim_id"),
  };
}

function caseRow(raw: unknown, whitespace: ReadonlySet<number>): AttestationCase {
  const row = record(raw, "case");
  return {
    id: text(row["id"], "id"),
    title: text(row["title"], "title"),
    change: text(row["change"], "change"),
    requireSignature: flag(row["require_signature"], "require_signature"),
    envelope: readEnvelope(row["envelope"], whitespace),
    python: verdict(row["python_verdict"]),
  };
}

function oracleRow(raw: unknown): OracleRow {
  const row = record(raw, "oracle row");
  const digest = text(row["digest"], "digest");
  if (!/^[0-9a-f]{64}$/u.test(digest)) throw new Error("Oracle digest must be lowercase SHA-256");
  return {
    label: text(row["label"], "label"),
    content: record(row["content"], "content"),
    pythonText: text(row["python_text"], "python_text"),
    digest,
  };
}

function codePoints(raw: unknown): ReadonlySet<number> {
  const points = list(raw, "python_whitespace").map((value) => {
    if (typeof value !== "bigint" || value < 0n || value > 0x10ffffn)
      throw new Error("python_whitespace holds code points");
    return Number(value);
  });
  if (points.length === 0) throw new Error("python_whitespace is empty");
  return new Set(points);
}

/**
 * Parse the committed instrument text with exact integer and float tokens.
 *
 * @param source - Committed JSON text.
 * @returns The instrument, or the reason it is unverifiable.
 */
export function parseAttestationInstrument(source: string): Loaded<AttestationInstrument> {
  try {
    const row = record(readJson(source), "instrument");
    if (row["schema"] !== attestationInstrumentSchema)
      throw new Error("Unsupported attestation instrument schema");
    const whitespace = codePoints(row["python_whitespace"]);
    const pin = record(row["source"], "source");
    return {
      ok: true,
      value: {
        id: text(row["artifact_id"], "artifact_id"),
        verifier: text(row["verifier"], "verifier"),
        envelopeSchema: text(row["envelope_schema"], "envelope_schema"),
        claimBoundary: text(row["claim_boundary"], "claim_boundary"),
        boundary: text(row["boundary"], "boundary"),
        source: {
          path: text(pin["path"], "source.path"),
          sha256: text(pin["sha256"], "source.sha256"),
          backend: text(pin["backend"], "source.backend"),
          status: text(pin["status"], "source.status"),
        },
        cases: list(row["cases"], "cases").map((item) => caseRow(item, whitespace)),
        oracle: list(row["canonical_oracle"], "canonical_oracle").map(oracleRow),
        whitespace,
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** Committed attestation data parsed once at the boundary. */
export const attestationInstrument = parseAttestationInstrument(committedText);
