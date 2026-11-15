// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — attested result-pack verification in the browser

import { readJson } from "../../shared/contracts/jsonTransport";
import { pythonCanonicalText, pythonStrip, sha256Hex } from "./pythonCanonical";

/** The four closed outcomes of the attestation owner. */
export type AttestationStatus = "VERIFIED" | "STRIPPED" | "FORGED" | "UNGRADED";

/** One attested result-pack envelope, read without coercing any field. */
export interface AttestedEnvelope {
  /** Envelope schema as written; the owner does not gate on it. */ readonly schema: string;
  /** Claim unit identifier, stripped as the owner strips it. */ readonly claimId: string;
  /** Claim and honesty axes bound into the digest. */ readonly claimAxes: Record<string, unknown>;
  /** Attested content bound into the digest. */ readonly content: Record<string, unknown>;
  /** Carried digest, stripped and lower-cased; empty when absent. */ readonly contentDigest: string;
  /** Signature token; empty when unsigned. */ readonly signature: string;
  /** Claim boundary carried by the envelope. */ readonly claimBoundary: string;
}

/** The decision for one envelope, with the digests it compared. */
export interface AttestationVerdict {
  /** Closed outcome. */ readonly status: AttestationStatus;
  /** The owner's reason text for this outcome. */ readonly reason: string;
  /** Digest recomputed here; empty when no digest was computed. */ readonly expectedDigest: string;
  /** Digest carried by the envelope. */ readonly observedDigest: string;
  /** Claim unit under verification. */ readonly claimId: string;
}

/** Axes the owner requires before any digest is compared. */
export const requiredAxes = ["claim_unit", "honesty_axes", "backend_class"] as const;

function mapping(row: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = row[key];
  if (value === undefined) return {};
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("claim_axes and content must be mappings when present");
  return value as Record<string, unknown>;
}

function field(row: Record<string, unknown>, key: string): string {
  const value = row[key];
  if (value === undefined) return "";
  // The owner would coerce with str(); the browser refuses instead of guessing a spelling.
  if (typeof value !== "string") throw new Error(`${key} must be text`);
  return value;
}

/**
 * Read an envelope with the owner's construction checks.
 *
 * @param raw - Envelope value read by `readJson`, so integers and floats stay distinct.
 * @param whitespace - Code points Python's `str.strip()` removes.
 * @returns The envelope.
 * @throws Error Where the owner refuses to construct it: a blank claim, a malformed digest, a non-mapping axis set or content.
 */
export function readEnvelope(raw: unknown, whitespace: ReadonlySet<number>): AttestedEnvelope {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw))
    throw new Error("envelope must be a mapping");
  const row = raw as Record<string, unknown>;
  const claimAxes = mapping(row, "claim_axes");
  const content = mapping(row, "content");
  const claimId = pythonStrip(field(row, "claim_id"), whitespace);
  if (claimId === "") throw new Error("claim_id must be non-empty");
  const contentDigest = pythonStrip(field(row, "content_digest"), whitespace).toLowerCase();
  if (contentDigest !== "" && !/^[0-9a-f]{64}$/u.test(contentDigest))
    throw new Error("content_digest must be empty or 64-char lowercase hex");
  return {
    schema: field(row, "schema"),
    claimId,
    claimAxes,
    content,
    contentDigest,
    signature: field(row, "signature"),
    claimBoundary: field(row, "claim_boundary"),
  };
}

/**
 * List the required axes that are absent, null, blank text or an empty list.
 *
 * @param axes - Claim axes of the envelope.
 * @param whitespace - Code points Python's `str.strip()` removes.
 * @returns Missing axis names in the owner's order.
 */
export function missingAxes(
  axes: Record<string, unknown>,
  whitespace: ReadonlySet<number>,
): readonly string[] {
  return requiredAxes.filter((key) => {
    if (!Object.hasOwn(axes, key)) return true;
    const value = axes[key];
    if (value === null) return true;
    if (typeof value === "string") return pythonStrip(value, whitespace) === "";
    return Array.isArray(value) && value.length === 0;
  });
}

/**
 * Recompute the digest the owner binds: claim axes and content together.
 *
 * @param envelope - Envelope whose content and axes are hashed.
 * @returns Lowercase SHA-256 of the canonical text.
 * @throws Error When the content cannot be written canonically.
 */
export async function boundDigest(envelope: AttestedEnvelope): Promise<string> {
  return sha256Hex(
    pythonCanonicalText({ claim_axes: envelope.claimAxes, content: envelope.content }),
  );
}

/**
 * Decide one envelope in the owner's fail-closed order.
 *
 * Missing axes are STRIPPED; a missing or undigestable digest is UNGRADED; a
 * mismatch is FORGED; a required but absent signature, or any signature
 * without a keyring, is UNGRADED; only an intact unsigned envelope is VERIFIED.
 *
 * @param envelope - Envelope read by {@link readEnvelope}.
 * @param requireSignature - Whether an unsigned envelope must stay UNGRADED.
 * @param whitespace - Code points Python's `str.strip()` removes.
 * @returns The verdict, with the same reason texts as the owner.
 */
export async function verifyEnvelope(
  envelope: AttestedEnvelope,
  requireSignature: boolean,
  whitespace: ReadonlySet<number>,
): Promise<AttestationVerdict> {
  const verdict = verdictFor(envelope);
  const missing = missingAxes(envelope.claimAxes, whitespace);
  if (missing.length > 0)
    return verdict(
      "STRIPPED",
      `required claim axes missing or empty: ${missing.join(", ")}; refuse silent-validated badge`,
    );
  if (envelope.contentDigest === "")
    return verdict(
      "UNGRADED",
      "content_digest absent; refuse invent-green attestation without digest",
    );
  let expected: string;
  try {
    expected = await boundDigest(envelope);
  } catch (cause: unknown) {
    return verdict("UNGRADED", `content not digestable: ${(cause as Error).message}`);
  }
  if (expected !== envelope.contentDigest)
    return verdict("FORGED", "content_digest does not match recomputed bound digest", expected);
  return signatureVerdict(envelope, requireSignature, whitespace, expected);
}

type VerdictOf = (
  status: AttestationStatus,
  reason: string,
  expected?: string,
) => AttestationVerdict;

/** Build verdicts that carry this envelope's claim and observed digest. */
function verdictFor(envelope: AttestedEnvelope): VerdictOf {
  return (status, reason, expected = "") => ({
    status,
    reason,
    expectedDigest: expected,
    observedDigest: envelope.contentDigest,
    claimId: envelope.claimId,
  });
}

/** The signature steps that follow a matching digest. */
function signatureVerdict(
  envelope: AttestedEnvelope,
  requireSignature: boolean,
  whitespace: ReadonlySet<number>,
  expected: string,
): AttestationVerdict {
  const verdict = verdictFor(envelope);
  const signed = pythonStrip(envelope.signature, whitespace) !== "";
  if (requireSignature && !signed)
    return verdict(
      "UNGRADED",
      "signature required but missing; absent keys yield UNGRADED, never silent-validated",
      expected,
    );
  if (signed)
    return verdict(
      "UNGRADED",
      "signature field present but pure local verifier does not validate cryptographic signatures without a configured keyring; UNGRADED",
      expected,
    );
  return verdict(
    "VERIFIED",
    "unsigned envelope intact: claim axes present and content digest matches",
    expected,
  );
}

/** Largest envelope file the browser check reads. */
export const maxEnvelopeBytes = 1024 * 1024;

/** Result of checking one envelope file: a verdict, or the reason none was reached. */
export type EnvelopeCheck =
  | {
      /** A verdict was reached. */ readonly ok: true /** The verdict. */;
      readonly verdict: AttestationVerdict;
    }
  | {
      /** The file was refused before a verdict. */ readonly ok: false /** Why. */;
      readonly reason: string;
    };

/**
 * Read and verify an envelope file's text without sending it anywhere.
 *
 * @param source - File text.
 * @param requireSignature - Whether an unsigned envelope must stay UNGRADED.
 * @param whitespace - Code points Python's `str.strip()` removes.
 * @returns The verdict, or the reason the file was refused.
 */
export async function checkEnvelopeText(
  source: string,
  requireSignature: boolean,
  whitespace: ReadonlySet<number>,
): Promise<EnvelopeCheck> {
  try {
    const envelope = readEnvelope(readJson(source), whitespace);
    return { ok: true, verdict: await verifyEnvelope(envelope, requireSignature, whitespace) };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}
