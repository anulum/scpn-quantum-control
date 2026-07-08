// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web committed-evidence loaders (fail-closed)

/**
 * Typed, fail-closed views over the committed repo artefacts the Phase-0
 * panel renders. Every loader returns a discriminated result instead of
 * throwing: a malformed artefact renders as a loud `unverifiable` block,
 * never as a silent blank or a downgraded-to-green card.
 */

import studioManifestJson from "../../../docs/_generated/studio_manifest.json";
import scorecardJson from "../../../data/differentiable_phase_qnode/differentiable_baseline_scorecard_20260620.json";
import gradientPlansJson from "../../../data/differentiable_phase_qnode/gradient_plan_explanations_20260709.json";
import supportMatrixJson from "../../../data/differentiable_phase_qnode/differentiable_transform_support_matrix_20260708.json";

export interface StudioVerb {
  readonly verb: string;
  readonly safetyTier: string;
  readonly sideEffect: string;
  readonly timingClass: string;
  readonly fidelity: string;
  readonly produces: readonly string[];
  readonly backends: readonly string[];
}

export interface StudioManifestView {
  readonly studio: string;
  readonly studioVersion: string;
  readonly contentDigest: string;
  readonly transportProfile: string;
  readonly verbs: readonly StudioVerb[];
  readonly evidenceTypes: readonly string[];
}

export interface SupportMatrixRowView {
  readonly rowId: string;
  readonly lane: string;
  readonly caseIds: readonly string[];
  readonly evidence: readonly string[];
  readonly transformStack: readonly string[];
  readonly status: string;
  readonly supported: boolean;
  readonly residual: number | null;
  readonly tolerance: number;
  readonly blockedReasons: readonly string[];
  readonly notes: readonly string[];
}

export interface SupportMatrixView {
  readonly artifactId: string;
  readonly claimBoundary: string;
  readonly rows: readonly SupportMatrixRowView[];
}

export interface ScorecardRowView {
  readonly category: string;
  readonly status: string;
  readonly blockers: readonly string[];
}

export interface ScorecardView {
  readonly artifactId: string;
  readonly claimBoundary: string;
  readonly rows: readonly ScorecardRowView[];
}

export interface GradientPlanExplanationRowView {
  readonly cellId: string;
  readonly operation: string;
  readonly framework: string;
  readonly backend: string;
  readonly transform: string;
  readonly supported: boolean;
  readonly status: string;
  readonly selectedMethod: string;
  readonly methodFamily: string;
  readonly evaluationMode: string;
  readonly backendFamily: string;
  readonly backendEvaluations: number;
  readonly shots: number | null;
  readonly requiresFiniteShotVariance: boolean;
  readonly requiresHardwarePolicy: boolean;
  readonly why: readonly string[];
  readonly failClosedBoundaries: readonly string[];
  readonly warnings: readonly string[];
  readonly alternatives: readonly string[];
  readonly claimBoundary: string;
}

export interface GradientPlanExplanationView {
  readonly artifactId: string;
  readonly claimBoundary: string;
  readonly methodFamilies: readonly string[];
  readonly rows: readonly GradientPlanExplanationRowView[];
}

export type Loaded<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringList(value: unknown): readonly string[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  return value.every((item) => typeof item === "string") ? (value as string[]) : null;
}

function loadManifest(raw: unknown): Loaded<StudioManifestView> {
  if (!isRecord(raw) || !isRecord(raw["schema_a"])) {
    return { ok: false, reason: "studio manifest is missing its schema_a block" };
  }
  const schemaA = raw["schema_a"];
  const evidenceTypes = stringList(schemaA["evidence_types"]);
  const verbsRaw = schemaA["verbs"];
  if (
    typeof schemaA["studio"] !== "string" ||
    typeof schemaA["studio_version"] !== "string" ||
    typeof schemaA["content_digest"] !== "string" ||
    typeof schemaA["transport_profile"] !== "string" ||
    evidenceTypes === null ||
    !Array.isArray(verbsRaw)
  ) {
    return { ok: false, reason: "studio manifest schema_a fields are malformed" };
  }
  const verbs: StudioVerb[] = [];
  for (const entry of verbsRaw) {
    if (!isRecord(entry) || !isRecord(entry["timing"])) {
      return { ok: false, reason: "studio manifest carries a malformed verb entry" };
    }
    const produces = stringList(entry["produces"]);
    const backends = stringList(entry["backends"]);
    if (
      typeof entry["verb"] !== "string" ||
      typeof entry["safety_tier"] !== "string" ||
      typeof entry["side_effect"] !== "string" ||
      typeof entry["timing"]["class"] !== "string" ||
      typeof entry["fidelity"] !== "string" ||
      produces === null ||
      backends === null
    ) {
      return { ok: false, reason: "studio manifest carries a malformed verb entry" };
    }
    verbs.push({
      verb: entry["verb"],
      safetyTier: entry["safety_tier"],
      sideEffect: entry["side_effect"],
      timingClass: entry["timing"]["class"],
      fidelity: entry["fidelity"],
      produces,
      backends,
    });
  }
  return {
    ok: true,
    value: {
      studio: schemaA["studio"],
      studioVersion: schemaA["studio_version"],
      contentDigest: schemaA["content_digest"],
      transportProfile: schemaA["transport_profile"],
      verbs,
      evidenceTypes,
    },
  };
}

function loadSupportMatrix(raw: unknown): Loaded<SupportMatrixView> {
  if (!isRecord(raw)) {
    return { ok: false, reason: "support-matrix artefact is not an object" };
  }
  const rowsRaw = raw["support_matrix"];
  if (
    typeof raw["artifact_id"] !== "string" ||
    typeof raw["claim_boundary"] !== "string" ||
    !Array.isArray(rowsRaw)
  ) {
    return { ok: false, reason: "support-matrix artefact fields are malformed" };
  }
  const rows: SupportMatrixRowView[] = [];
  for (const entry of rowsRaw) {
    if (!isRecord(entry)) {
      return { ok: false, reason: "support-matrix artefact carries a malformed row" };
    }
    const transformStack = stringList(entry["transform_stack"]);
    const caseIds = stringList(entry["case_ids"]);
    const evidence = stringList(entry["evidence"]);
    const blockedReasons = stringList(entry["blocked_reasons"]);
    const notes = stringList(entry["notes"]);
    const residual = entry["residual"];
    if (
      typeof entry["row_id"] !== "string" ||
      typeof entry["lane"] !== "string" ||
      typeof entry["status"] !== "string" ||
      typeof entry["supported"] !== "boolean" ||
      typeof entry["tolerance"] !== "number" ||
      (residual !== null && typeof residual !== "number") ||
      caseIds === null ||
      evidence === null ||
      transformStack === null ||
      blockedReasons === null ||
      notes === null
    ) {
      return { ok: false, reason: "support-matrix artefact carries a malformed row" };
    }
    rows.push({
      rowId: entry["row_id"],
      lane: entry["lane"],
      caseIds,
      evidence,
      transformStack,
      status: entry["status"],
      supported: entry["supported"],
      residual,
      tolerance: entry["tolerance"],
      blockedReasons,
      notes,
    });
  }
  return {
    ok: true,
    value: {
      artifactId: raw["artifact_id"],
      claimBoundary: raw["claim_boundary"],
      rows,
    },
  };
}

function loadScorecard(raw: unknown): Loaded<ScorecardView> {
  if (!isRecord(raw)) {
    return { ok: false, reason: "scorecard artefact is not an object" };
  }
  const rowsRaw = raw["rows"];
  if (
    typeof raw["artifact_id"] !== "string" ||
    typeof raw["claim_boundary"] !== "string" ||
    !Array.isArray(rowsRaw)
  ) {
    return { ok: false, reason: "scorecard artefact fields are malformed" };
  }
  const rows: ScorecardRowView[] = [];
  for (const entry of rowsRaw) {
    if (!isRecord(entry)) {
      return { ok: false, reason: "scorecard artefact carries a malformed row" };
    }
    const blockers = stringList(entry["blockers"]);
    if (
      typeof entry["category"] !== "string" ||
      typeof entry["status"] !== "string" ||
      blockers === null
    ) {
      return { ok: false, reason: "scorecard artefact carries a malformed row" };
    }
    rows.push({
      category: entry["category"],
      status: entry["status"],
      blockers,
    });
  }
  return {
    ok: true,
    value: {
      artifactId: raw["artifact_id"],
      claimBoundary: raw["claim_boundary"],
      rows,
    },
  };
}

function loadGradientPlanExplanations(raw: unknown): Loaded<GradientPlanExplanationView> {
  if (!isRecord(raw)) {
    return { ok: false, reason: "gradient-plan artefact is not an object" };
  }
  const rowsRaw = raw["explanations"];
  const methodFamilies = stringList(raw["method_families"]);
  if (
    typeof raw["artifact_id"] !== "string" ||
    typeof raw["claim_boundary"] !== "string" ||
    methodFamilies === null ||
    !Array.isArray(rowsRaw)
  ) {
    return { ok: false, reason: "gradient-plan artefact fields are malformed" };
  }
  const rows: GradientPlanExplanationRowView[] = [];
  for (const entry of rowsRaw) {
    if (!isRecord(entry)) {
      return { ok: false, reason: "gradient-plan artefact carries a malformed row" };
    }
    const why = stringList(entry["why"]);
    const failClosedBoundaries = stringList(entry["fail_closed_boundaries"]);
    const warnings = stringList(entry["warnings"]);
    const alternatives = stringList(entry["alternatives"]);
    const shots = entry["shots"];
    if (
      typeof entry["cell_id"] !== "string" ||
      typeof entry["operation"] !== "string" ||
      typeof entry["framework"] !== "string" ||
      typeof entry["backend"] !== "string" ||
      typeof entry["transform"] !== "string" ||
      typeof entry["supported"] !== "boolean" ||
      typeof entry["status"] !== "string" ||
      typeof entry["selected_method"] !== "string" ||
      typeof entry["method_family"] !== "string" ||
      typeof entry["evaluation_mode"] !== "string" ||
      typeof entry["backend_family"] !== "string" ||
      typeof entry["backend_evaluations"] !== "number" ||
      (shots !== null && typeof shots !== "number") ||
      typeof entry["requires_finite_shot_variance"] !== "boolean" ||
      typeof entry["requires_hardware_policy"] !== "boolean" ||
      why === null ||
      failClosedBoundaries === null ||
      warnings === null ||
      alternatives === null ||
      typeof entry["claim_boundary"] !== "string"
    ) {
      return { ok: false, reason: "gradient-plan artefact carries a malformed row" };
    }
    rows.push({
      cellId: entry["cell_id"],
      operation: entry["operation"],
      framework: entry["framework"],
      backend: entry["backend"],
      transform: entry["transform"],
      supported: entry["supported"],
      status: entry["status"],
      selectedMethod: entry["selected_method"],
      methodFamily: entry["method_family"],
      evaluationMode: entry["evaluation_mode"],
      backendFamily: entry["backend_family"],
      backendEvaluations: entry["backend_evaluations"],
      shots,
      requiresFiniteShotVariance: entry["requires_finite_shot_variance"],
      requiresHardwarePolicy: entry["requires_hardware_policy"],
      why,
      failClosedBoundaries,
      warnings,
      alternatives,
      claimBoundary: entry["claim_boundary"],
    });
  }
  return {
    ok: true,
    value: {
      artifactId: raw["artifact_id"],
      claimBoundary: raw["claim_boundary"],
      methodFamilies,
      rows,
    },
  };
}

/** Parse an arbitrary manifest payload (exported for fail-closed tests). */
export const parseManifest = loadManifest;
/** Parse an arbitrary support-matrix payload (exported for fail-closed tests). */
export const parseSupportMatrix = loadSupportMatrix;
/** Parse an arbitrary scorecard payload (exported for fail-closed tests). */
export const parseScorecard = loadScorecard;
/** Parse arbitrary gradient-plan explanations (exported for fail-closed tests). */
export const parseGradientPlanExplanations = loadGradientPlanExplanations;

/** The committed schema-A manifest, guarded. */
export const studioManifest: Loaded<StudioManifestView> = loadManifest(studioManifestJson);
/** The committed transform-algebra support matrix, guarded. */
export const supportMatrix: Loaded<SupportMatrixView> = loadSupportMatrix(supportMatrixJson);
/** The committed gradient-plan explanation artefact, guarded. */
export const gradientPlanExplanations: Loaded<GradientPlanExplanationView> =
  loadGradientPlanExplanations(gradientPlansJson);
/** The committed differentiable baseline scorecard, guarded. */
export const scorecard: Loaded<ScorecardView> = loadScorecard(scorecardJson);
