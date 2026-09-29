// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — source-owned evidence presentation

import type { ProgramAdUnit } from "../../panel/programAd";
import { canonicalBytes, writeJson } from "../contracts";

/** Independent source declarations; inspecting them never certifies their truth. */
export interface EvidenceView {
  /** Complete source identity used only to bind UI verification state. */ readonly snapshotIdentity: string;
  /** Source schema, or a visible missing value. */ readonly schema: string | null;
  /** Source entity or artefact identifier. */ readonly source: string | null;
  /** Source-owned digest, never replaced with the UI snapshot identity. */ readonly digest: string | null;
  /** Scientific evidence kind as declared by the producer. */ readonly kind: string | null;
  /** Scientific claim status as declared by the producer. */ readonly claim: string | null;
  /** Producer's admission declaration, independent of local verification. */ readonly admission: string | null;
  /** Exact declared scope, without an inferred broader claim. */ readonly boundary: string | null;
  /** Producer freshness declaration, never advanced by the viewer. */ readonly freshness: string | null;
  /** Producer substrate declaration. */ readonly substrate: string | null;
  /** Original numeric parity declarations, never an inferred aggregate. */ readonly exactness: string | null;
  /** Original provenance rendered as text without following references. */ readonly provenance: string | null;
  /** Attestation presence is a custody observation, not verification. */ readonly seal: "missing" | "present-unverified" | "malformed";
  /** Missing, unsupported or malformed presentation fields. */ readonly issues: readonly string[];
  /** Whether the source can be bound to an immutable UI identity. */ readonly identifiable: boolean;
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>> : {};
}
function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}
function identity(source: unknown): { value: string; error: string | null } {
  try {
    return { value: new TextDecoder().decode(canonicalBytes("studio_evidence_view.v1", source)), error: null };
  } catch {
    return { value: "invalid-snapshot", error: "Source cannot be represented as an immutable snapshot" };
  }
}
function finish(source: unknown, fields: Omit<EvidenceView, "snapshotIdentity" | "identifiable" | "issues">, extra: readonly string[] = []): EvidenceView {
  const snapshot = identity(source);
  const issues = [...extra];
  for (const [name, value] of [["schema", fields.schema], ["source", fields.source], ["source digest", fields.digest], ["claim status", fields.claim], ["evidence kind", fields.kind]] as const) {
    if (value === null) issues.push(`Missing ${name}`);
  }
  if (snapshot.error) issues.push(snapshot.error);
  return Object.freeze({ ...fields, snapshotIdentity: snapshot.value, identifiable: snapshot.error === null,
    issues: Object.freeze(issues) });
}

/** Project the original Python schema-B wire fields without implementing its grading rules. */
export function projectEvidenceBundle(source: unknown): EvidenceView {
  // Canonical inspection refuses accessors before any projection can read them.
  const snapshot = identity(source);
  const value: Readonly<Record<string, unknown>> = snapshot.error ? {} : record(source);
  const prov = record(value["prov"]);
  const entity = record(prov["entity"]);
  const boundary = record(value["claim_boundary"]);
  const domain = record(boundary["validity_domain"]);
  const schema = text(value["schema"]);
  const attestation = value["attestation"];
  const seal = attestation == null ? "missing"
    : Object.keys(record(attestation)).length > 0 ? "present-unverified" : "malformed";
  const activity = record(prov["activity"]);
  const agent = record(prov["agent"]);
  const parity = record(value["numeric_provenance"])["parity"];
  const exactness = Array.isArray(parity) && parity.length > 0 ? writeJson(parity) : null;
  const provenance = [text(activity["regenerated_by"]), text(activity["started"]), text(agent["operator"])]
    .filter((item): item is string => item !== null).join(" · ") || null;
  return finish(source, {
    schema, source: text(entity["id"]), digest: text(entity["digest"]),
    kind: text(value["evidence_kind"]), claim: text(boundary["status"]),
    admission: text(boundary["admission"]), boundary: text(domain["note"]),
    freshness: text(value["freshness"]), substrate: text(value["substrate"]), exactness, provenance, seal,
  }, schema !== null && !["studio.evidence-replay.v1", "studio.hardware-result-pack.v1"].includes(schema)
    ? ["Unsupported evidence schema; declarations shown without admission"] : []);
}

/** Project an original replay unit; its verifier still owns all numerical decisions. */
export function projectProgramAdEvidence(unit: ProgramAdUnit): EvidenceView {
  return finish(unit, {
    schema: unit.schema, source: unit.artifactId, digest: unit.inputSha256,
    kind: null, claim: null, admission: null, boundary: unit.claimBoundary,
    freshness: null, substrate: null, exactness: null, provenance: null, seal: "missing",
  });
}
