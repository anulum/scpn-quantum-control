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

/**
 * One verb the Studio manifest declares, with the safety envelope it runs under.
 */
export interface StudioVerb {
  /** Verb name as the manifest spells it. */
  readonly verb: string;
  /** Safety tier governing what the verb may do. */
  readonly safetyTier: string;
  /** Side-effect class, which decides whether a run needs confirmation. */
  readonly sideEffect: string;
  /** Timing class the verb is scheduled under. */
  readonly timingClass: string;
  /** Fidelity claim the verb's output may carry. */
  readonly fidelity: string;
  /** Evidence types a run of this verb produces. */
  readonly produces: readonly string[];
  /** Backends the verb may be dispatched to. */
  readonly backends: readonly string[];
}

/**
 * Typed view of the committed Studio manifest's `schema_a` block.
 */
export interface StudioManifestView {
  /** Studio identifier the manifest declares. */
  readonly studio: string;
  /** Version of the Studio surface this manifest describes. */
  readonly studioVersion: string;
  /** Digest of the manifest content, for matching a rendered panel to its source. */
  readonly contentDigest: string;
  /** Transport profile the verbs are dispatched over. */
  readonly transportProfile: string;
  /** Every verb the manifest declares. */
  readonly verbs: readonly StudioVerb[];
  /** Evidence types the manifest recognises. */
  readonly evidenceTypes: readonly string[];
}

/**
 * One transform-support row: a lane, its cases and whether they are supported.
 */
export interface SupportMatrixRowView {
  /** Stable row identifier. */
  readonly rowId: string;
  /** Lane the row belongs to. */
  readonly lane: string;
  /** Cases this row aggregates. */
  readonly caseIds: readonly string[];
  /** Evidence artefacts backing the row's verdict. */
  readonly evidence: readonly string[];
  /** Transform stack the row exercises, outermost first. */
  readonly transformStack: readonly string[];
  /** Row verdict as the artefact states it. */
  readonly status: string;
  /** Whether the stack is supported; false leaves `residual` meaningless. */
  readonly supported: boolean;
  /** Numerical residual, or `null` when the row produced none. */
  readonly residual: number | null;
  /** Tolerance the residual was judged against. */
  readonly tolerance: number;
  /** Why the row is unsupported, one entry per distinct reason. */
  readonly blockedReasons: readonly string[];
  /** Free-text notes carried from the artefact. */
  readonly notes: readonly string[];
}

/**
 * Typed view of the committed transform-support matrix artefact.
 */
export interface SupportMatrixView {
  /** Identifier of the artefact this view was loaded from. */
  readonly artifactId: string;
  /** What the matrix may and may not be cited as evidence for. */
  readonly claimBoundary: string;
  /** Every support row in the artefact. */
  readonly rows: readonly SupportMatrixRowView[];
}

/**
 * One baseline scorecard category and its verdict.
 */
export interface ScorecardRowView {
  /** Category the row scores. */
  readonly category: string;
  /** Verdict for the category. */
  readonly status: string;
  /** What holds the category back, one entry per distinct blocker. */
  readonly blockers: readonly string[];
}

/**
 * Typed view of the committed differentiable baseline scorecard.
 */
export interface ScorecardView {
  /** Identifier of the artefact this view was loaded from. */
  readonly artifactId: string;
  /** What the scorecard may and may not be cited as evidence for. */
  readonly claimBoundary: string;
  /** Every scored category. */
  readonly rows: readonly ScorecardRowView[];
}

/**
 * One gradient-plan cell: the method chosen for an operation, and why.
 */
export interface GradientPlanExplanationRowView {
  /** Stable identifier for this operation/framework/backend cell. */
  readonly cellId: string;
  /** Operation the plan was made for. */
  readonly operation: string;
  /** Framework the operation is expressed in. */
  readonly framework: string;
  /** Backend the plan targets. */
  readonly backend: string;
  /** Transform the plan applies. */
  readonly transform: string;
  /** Whether a gradient can be produced at all for this cell. */
  readonly supported: boolean;
  /** Plan verdict as the artefact states it. */
  readonly status: string;
  /** Differentiation method the planner chose. */
  readonly selectedMethod: string;
  /** Family the chosen method belongs to. */
  readonly methodFamily: string;
  /** How the backend is evaluated for this plan. */
  readonly evaluationMode: string;
  /** Family the backend belongs to. */
  readonly backendFamily: string;
  /** Number of backend evaluations one gradient costs. */
  readonly backendEvaluations: number;
  /** Planned shot count, or `null` when the backend is not shot-based. */
  readonly shots: number | null;
  /** Whether the method's error bound assumes finite shots. */
  readonly requiresFiniteShotVariance: boolean;
  /** Whether running this plan needs a hardware policy decision. */
  readonly requiresHardwarePolicy: boolean;
  /** The planner's reasoning, one entry per step. */
  readonly why: readonly string[];
  /** Boundaries this plan fails closed at rather than approximating past. */
  readonly failClosedBoundaries: readonly string[];
  /** Warnings attached to the plan without blocking it. */
  readonly warnings: readonly string[];
  /** Methods considered and not chosen. */
  readonly alternatives: readonly string[];
  /** What this cell may and may not be cited as evidence for. */
  readonly claimBoundary: string;
}

/**
 * Typed view of the committed gradient-plan explanation artefact.
 */
export interface GradientPlanExplanationView {
  /** Identifier of the artefact this view was loaded from. */
  readonly artifactId: string;
  /** What the explanations may and may not be cited as evidence for. */
  readonly claimBoundary: string;
  /** Every method family appearing in the rows. */
  readonly methodFamilies: readonly string[];
  /** Every explained plan cell. */
  readonly rows: readonly GradientPlanExplanationRowView[];
}

/**
 * A load result that cannot throw: either a parsed `value` or the `reason`
 * the artefact was refused. The panel renders the refusal loudly rather than
 * degrading it to a blank or a green card.
 */
export type Loaded<T> =
  | {
      /** Discriminant: the artefact parsed. */
      readonly ok: true;
      /** The parsed view. */
      readonly value: T;
    }
  | {
      /** Discriminant: the artefact was refused. */
      readonly ok: false;
      /** Why it was refused, rendered to the reader rather than swallowed. */
      readonly reason: string;
    };

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
/** Parse a Studio manifest payload without reading it from disk. */
export const parseManifest = loadManifest;
/** Parse an arbitrary support-matrix payload (exported for fail-closed tests). */
/** Parse a transform-support matrix payload without reading it from disk. */
export const parseSupportMatrix = loadSupportMatrix;
/** Parse an arbitrary scorecard payload (exported for fail-closed tests). */
/** Parse a baseline scorecard payload without reading it from disk. */
export const parseScorecard = loadScorecard;
/** Parse arbitrary gradient-plan explanations (exported for fail-closed tests). */
/** Parse a gradient-plan explanation payload without reading it from disk. */
export const parseGradientPlanExplanations = loadGradientPlanExplanations;

/** The committed schema-A manifest, guarded. */
/** The committed Studio manifest, parsed at module load. */
export const studioManifest: Loaded<StudioManifestView> = loadManifest(studioManifestJson);
/** The committed transform-algebra support matrix, guarded. */
/** The committed transform-support matrix, parsed at module load. */
export const supportMatrix: Loaded<SupportMatrixView> = loadSupportMatrix(supportMatrixJson);
/** The committed gradient-plan explanation artefact, guarded. */
/** The committed gradient-plan explanations, parsed at module load. */
export const gradientPlanExplanations: Loaded<GradientPlanExplanationView> =
  loadGradientPlanExplanations(gradientPlansJson);
/** The committed differentiable baseline scorecard, guarded. */
/** The committed differentiable baseline scorecard, parsed at module load. */
export const scorecard: Loaded<ScorecardView> = loadScorecard(scorecardJson);
