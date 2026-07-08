// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web support-matrix explorer model

import type { SupportMatrixRowView, SupportMatrixView } from "./data";

export const SUPPORT_MATRIX_ALL = "all";

export interface SupportMatrixFilters {
  readonly operationQuery: string;
  readonly framework: string;
  readonly backend: string;
  readonly exactness: string;
  readonly claimStatus: string;
}

export interface SupportMatrixExplorerRow {
  readonly rowId: string;
  readonly operation: string;
  readonly framework: string;
  readonly backend: string;
  readonly exactness: string;
  readonly claimStatus: string;
  readonly lane: string;
  readonly transformStack: readonly string[];
  readonly caseIds: readonly string[];
  readonly evidence: readonly string[];
  readonly status: string;
  readonly supported: boolean;
  readonly residual: number | null;
  readonly tolerance: number;
  readonly blockedReasons: readonly string[];
  readonly notes: readonly string[];
}

export interface SupportMatrixExplorerView {
  readonly artifactId: string;
  readonly claimBoundary: string;
  readonly rows: readonly SupportMatrixExplorerRow[];
  readonly frameworks: readonly string[];
  readonly backends: readonly string[];
  readonly exactnessLevels: readonly string[];
  readonly claimStatuses: readonly string[];
  readonly supportedCount: number;
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
