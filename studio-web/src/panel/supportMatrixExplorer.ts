// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web support-matrix explorer model

import type { SupportMatrixRowView, SupportMatrixView } from "./data";

/** Filter sentinel meaning "do not narrow on this facet". */
export const SUPPORT_MATRIX_ALL = "all";

/**
 * The reader's current filter selection; `SUPPORT_MATRIX_ALL` means unfiltered.
 */
export interface SupportMatrixFilters {
  /** Free-text query matched against the operation name. */
  readonly operationQuery: string;
  /** Selected framework, or `SUPPORT_MATRIX_ALL`. */
  readonly framework: string;
  /** Selected backend, or `SUPPORT_MATRIX_ALL`. */
  readonly backend: string;
  /** Selected exactness level, or `SUPPORT_MATRIX_ALL`. */
  readonly exactness: string;
  /** Selected claim status, or `SUPPORT_MATRIX_ALL`. */
  readonly claimStatus: string;
}

/**
 * One explorer row: a support-matrix row flattened with the facets it is filtered by.
 */
export interface SupportMatrixExplorerRow {
  /** Stable row identifier, carried from the artefact. */
  readonly rowId: string;
  /** Operation the row covers. */
  readonly operation: string;
  /** Framework the operation is expressed in. */
  readonly framework: string;
  /** Backend the row targets. */
  readonly backend: string;
  /** Exactness level the row's evidence supports. */
  readonly exactness: string;
  /** What may be claimed from this row. */
  readonly claimStatus: string;
  /** Lane the row belongs to. */
  readonly lane: string;
  /** Transform stack the row exercises, outermost first. */
  readonly transformStack: readonly string[];
  /** Cases this row aggregates. */
  readonly caseIds: readonly string[];
  /** Evidence artefacts backing the verdict. */
  readonly evidence: readonly string[];
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
 * The whole explorer: its rows, the facet values a reader can filter by, and the two counts shown above the table.
 */
export interface SupportMatrixExplorerView {
  /** Identifier of the artefact the view was built from. */
  readonly artifactId: string;
  /** What the matrix may and may not be cited as evidence for. */
  readonly claimBoundary: string;
  /** Every row, unfiltered. */
  readonly rows: readonly SupportMatrixExplorerRow[];
  /** Distinct frameworks present, for the filter control. */
  readonly frameworks: readonly string[];
  /** Distinct backends present, for the filter control. */
  readonly backends: readonly string[];
  /** Distinct exactness levels present, for the filter control. */
  readonly exactnessLevels: readonly string[];
  /** Distinct claim statuses present, for the filter control. */
  readonly claimStatuses: readonly string[];
  /** How many rows are supported. */
  readonly supportedCount: number;
  /** How many rows fail closed — counted separately so an unsupported row is never read as an absent one. */
  readonly failClosedCount: number;
}

function uniqueSorted(values: readonly string[]): readonly string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function includesAny(values: readonly string[], candidates: readonly string[]): boolean {
  return candidates.some((candidate) => values.includes(candidate));
}

function frameworkFor(row: SupportMatrixRowView): string {
  if (row.lane === "native") {
    return "native-transform";
  }
  if (row.lane === "custom_rules") {
    return "custom-rule-registry";
  }
  if (row.lane === "program_ad") {
    return "whole-program-ad";
  }
  if (row.lane === "quantum_gradients") {
    return "phase-qnode";
  }
  if (row.lane === "unsupported_boundary") {
    return "unsupported-boundary";
  }
  return row.lane;
}

function backendFor(row: SupportMatrixRowView): string {
  if (row.evidence.includes("parameter_shift")) {
    return "parameter-shift-reference";
  }
  if (row.evidence.includes("analytic_reference")) {
    return "analytic-reference";
  }
  if (row.evidence.includes("adjoint_identity")) {
    return "adjoint-identity";
  }
  if (row.evidence.includes("complex_step_real_analytic_route")) {
    return "complex-step-diagnostic";
  }
  if (row.evidence.includes("custom_rule_registry_required")) {
    return "custom-rule-registry-required";
  }
  if (row.evidence.includes("framework_parity_lane_required")) {
    return "framework-parity-required";
  }
  if (
    row.evidence.includes("finite_difference_diagnostic") ||
    row.evidence.includes("finite_difference_diagnostic_only")
  ) {
    return "finite-difference-diagnostic";
  }
  return row.evidence[0] ?? "unspecified";
}

function exactnessFor(row: SupportMatrixRowView): string {
  if (row.status === "blocked") {
    return "fail-closed-boundary";
  }
  if (
    includesAny(row.evidence, [
      "analytic_reference",
      "adjoint_identity",
      "parameter_shift",
      "CustomDerivativeRule",
    ])
  ) {
    return "reference-checked";
  }
  if (row.evidence.some((item) => item.includes("finite_difference"))) {
    return "diagnostic";
  }
  return "bounded-local";
}

function claimStatusFor(row: SupportMatrixRowView): string {
  if (row.status === "passed") {
    return "bounded-model";
  }
  if (row.status === "blocked") {
    return "fail-closed";
  }
  return "unverifiable";
}

function rowSearchText(row: SupportMatrixExplorerRow): string {
  return [
    row.rowId,
    row.operation,
    row.framework,
    row.backend,
    row.exactness,
    row.claimStatus,
    row.lane,
    ...row.transformStack,
    ...row.caseIds,
    ...row.evidence,
    ...row.blockedReasons,
    ...row.notes,
  ]
    .join(" ")
    .toLowerCase();
}

/** Flatten a support-matrix view into rows plus the facet values a reader filters by. */
export function buildSupportMatrixExplorer(
  matrix: SupportMatrixView,
): SupportMatrixExplorerView {
  const rows = matrix.rows.map((row) => ({
    rowId: row.rowId,
    operation: row.transformStack.join(" + "),
    framework: frameworkFor(row),
    backend: backendFor(row),
    exactness: exactnessFor(row),
    claimStatus: claimStatusFor(row),
    lane: row.lane,
    transformStack: row.transformStack,
    caseIds: row.caseIds,
    evidence: row.evidence,
    status: row.status,
    supported: row.supported,
    residual: row.residual,
    tolerance: row.tolerance,
    blockedReasons: row.blockedReasons,
    notes: row.notes,
  }));
  return {
    artifactId: matrix.artifactId,
    claimBoundary: matrix.claimBoundary,
    rows,
    frameworks: uniqueSorted(rows.map((row) => row.framework)),
    backends: uniqueSorted(rows.map((row) => row.backend)),
    exactnessLevels: uniqueSorted(rows.map((row) => row.exactness)),
    claimStatuses: uniqueSorted(rows.map((row) => row.claimStatus)),
    supportedCount: rows.filter((row) => row.supported).length,
    failClosedCount: rows.filter((row) => row.claimStatus === "fail-closed").length,
  };
}

/**
 * Narrow the rows to the reader's selection.
 *
 * A facet set to `SUPPORT_MATRIX_ALL` does not narrow, and the operation query
 * matches case-insensitively, so an empty selection returns every row rather
 * than none.
 */
export function filterSupportMatrixRows(
  rows: readonly SupportMatrixExplorerRow[],
  filters: SupportMatrixFilters,
): readonly SupportMatrixExplorerRow[] {
  const query = filters.operationQuery.trim().toLowerCase();
  return rows.filter((row) => {
    const matchesQuery = query === "" || rowSearchText(row).includes(query);
    return (
      matchesQuery &&
      (filters.framework === SUPPORT_MATRIX_ALL || row.framework === filters.framework) &&
      (filters.backend === SUPPORT_MATRIX_ALL || row.backend === filters.backend) &&
      (filters.exactness === SUPPORT_MATRIX_ALL || row.exactness === filters.exactness) &&
      (filters.claimStatus === SUPPORT_MATRIX_ALL || row.claimStatus === filters.claimStatus)
    );
  });
}
