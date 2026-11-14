// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — chimera, persistence and derivative-support data projection

import {
  agrees,
  count,
  finite,
  flag,
  list,
  numbers,
  record,
  text,
} from "../../shared/browser/jsonFields";

/** One population of the synthetic chimera with its mean order parameter. */
export interface ChimeraCluster {
  /** Node indices of the population. */ readonly nodes: readonly number[];
  /** Time-mean order parameter of the population over the settled window. */ readonly order: number;
}
/** The finite synthetic chimera as its owner reports it. */
export interface ChimeraTemplate {
  /** Source owner. */ readonly owner: string;
  /** Model and configuration origin. */ readonly source: string;
  /** Regime label of the configuration. */ readonly regime: string;
  /** Configuration seed. */ readonly seed: number;
  /** Populations in source order. */ readonly clusters: readonly ChimeraCluster[];
  /** Chimera index across populations. */ readonly chimeraIndex: number;
  /** Community metastability. */ readonly metastability: number;
  /** Time-mean global order parameter. */ readonly globalOrder: number;
  /** Settled final phases in radians, one per node. */ readonly finalPhases: readonly number[];
  /** Source claim boundary. */ readonly boundary: string;
}
/** Persistence summary of one homology dimension. */
export interface PersistenceDimension {
  /** Homology dimension label, H0 or H1. */ readonly dimension: string;
  /** Pairs returned by the owner. */ readonly pairs: number;
  /** Classes that never die. */ readonly essential: number;
  /** Finite classes above the lifetime threshold. */ readonly persistent: number;
  /** Longest finite lifetime. */ readonly longest: number;
  /** Finite death values in ascending order. */ readonly deaths: readonly number[];
}
/** Exact finite persistence summaries. */
export interface PersistenceTemplate {
  /** Source owner. */ readonly owner: string;
  /** Method statement. */ readonly method: string;
  /** Analytic spacing of the exact example. */ readonly spacing: number;
  /** Summary of the exact example. */ readonly exact: readonly PersistenceDimension[];
  /** Summary of the chimera's final phases. */ readonly chimera: readonly PersistenceDimension[];
  /** What the summaries may not be read as. */ readonly boundary: string;
}
/** One derivative-support decision of the topology projection. */
export interface SupportRow {
  /** Constraint capability. */ readonly capability: string;
  /** Supported or unsupported. */ readonly status: string;
  /** Mathematical derivative class. */ readonly differentiability: string;
  /** The owner's reason. */ readonly evidence: string;
}
/** One primal point and the owner's derivative decision there. */
export interface SensitivityCase {
  /** Case name. */ readonly name: string;
  /** Whether a local derivative exists. */ readonly supported: boolean;
  /** Support rows in owner order. */ readonly rows: readonly SupportRow[];
  /** Local JVP for the fixed active set, or null where none exists. */ readonly tangent:
    | readonly (readonly number[])[]
    | null;
  /** The owner's refusal, or null where a derivative exists. */ readonly refusal: string | null;
}
/** Fixed-active-set sensitivity at interior, kink and discrete points. */
export interface SensitivityTemplate {
  /** Source owner. */ readonly owner: string;
  /** Requested tangent direction. */ readonly direction: readonly (readonly number[])[];
  /** Cases in source order. */ readonly cases: readonly SensitivityCase[];
  /** Source claim boundary. */ readonly boundary: string;
}

/**
 * Parse the chimera template, refusing clusters that do not partition the nodes.
 * @param raw - Source template.
 * @returns The parsed template.
 */
export function parseChimera(raw: unknown): ChimeraTemplate {
  const row = record(raw, "chimera");
  const config = record(row["config"], "chimera.config");
  const clusters = list(row["clusters"], "chimera.clusters").map((item) => {
    const cluster = record(item, "cluster");
    return {
      nodes: list(cluster["nodes"], "nodes").map((node) => count(node, "node")),
      order: finite(cluster["mean_order_parameter"], "mean_order_parameter"),
    };
  });
  const finalPhases = numbers(row["final_phases"], "final_phases");
  const nodes = clusters.flatMap((cluster) => cluster.nodes).sort((a, b) => a - b);
  if (nodes.length !== finalPhases.length || nodes.some((node, at) => node !== at))
    throw new Error("Chimera clusters must partition the nodes");
  return {
    owner: text(row["owner"], "owner"),
    source: text(row["source"], "source"),
    regime: text(config["regime"], "regime"),
    seed: count(config["seed"], "seed"),
    clusters,
    chimeraIndex: finite(row["chimera_index"], "chimera_index"),
    metastability: finite(row["metastability"], "metastability"),
    globalOrder: finite(row["global_order_parameter_mean"], "global_order_parameter_mean"),
    finalPhases,
    boundary: text(row["claim_boundary"], "claim_boundary"),
  };
}

function parseSummary(raw: unknown, field: string): PersistenceDimension[] {
  const summary = record(raw, field);
  return Object.keys(summary)
    .sort()
    .map((dimension) => {
      const row = record(summary[dimension], dimension);
      return {
        dimension,
        pairs: count(row["pairs"], "pairs"),
        essential: count(row["essential"], "essential"),
        persistent: count(row["persistent_finite"], "persistent_finite"),
        longest: finite(row["longest_finite_lifetime"], "longest_finite_lifetime"),
        deaths: numbers(row["finite_deaths"], "finite_deaths"),
      };
    });
}

/**
 * Parse the persistence template, refusing an exact example that misses its analytic spacing.
 * @param raw - Source template.
 * @returns The parsed template.
 */
export function parsePersistence(raw: unknown): PersistenceTemplate {
  const row = record(raw, "persistence");
  const example = record(row["exact_example"], "exact_example");
  const spacing = finite(example["analytic_spacing"], "analytic_spacing");
  const exact = parseSummary(example["summary"], "exact_example.summary");
  const connected = exact.find((item) => item.dimension === "H0");
  if (!connected || connected.deaths.length === 0)
    throw new Error("The exact persistence example needs its H0 deaths");
  for (const death of connected.deaths) agrees(death, spacing, "exact H0 death");
  return {
    owner: text(row["owner"], "owner"),
    method: text(row["method"], "method"),
    spacing,
    exact,
    chimera: parseSummary(row["chimera_final_phases"], "chimera_final_phases"),
    boundary: text(row["boundary"], "boundary"),
  };
}

function parseMatrix(raw: unknown, field: string): number[][] {
  const rows = list(raw, field).map((line) => numbers(line, field));
  if (rows.length === 0 || rows.some((row) => row.length !== rows.length))
    throw new Error(`${field} must be a square matrix`);
  return rows;
}

function parseCase(raw: unknown): SensitivityCase {
  const row = record(raw, "case");
  const supported = flag(row["derivative_supported"], "derivative_supported");
  const rows = list(row["rows"], "rows").map((item) => {
    const support = record(item, "row");
    return {
      capability: text(support["capability"], "capability"),
      status: text(support["status"], "status"),
      differentiability: text(support["differentiability"], "differentiability"),
      evidence: text(support["evidence"], "evidence"),
    };
  });
  const blocked = rows.some((item) => item.status !== "supported");
  if (supported === blocked) throw new Error("Support rows contradict the derivative decision");
  if (!supported && row["projected_tangent"] !== null)
    throw new Error("An unsupported branch cannot display a derivative");
  return {
    name: text(row["case"], "case"),
    supported,
    rows,
    tangent: supported ? parseMatrix(row["projected_tangent"], "projected_tangent") : null,
    refusal: supported ? null : text(row["refusal"], "refusal"),
  };
}

/**
 * Parse the sensitivity template, refusing a derivative on an unsupported branch.
 * @param raw - Source template.
 * @returns The parsed template.
 */
export function parseSensitivity(raw: unknown): SensitivityTemplate {
  const row = record(raw, "sensitivity");
  const cases = list(row["cases"], "cases").map(parseCase);
  if (cases.length === 0) throw new Error("Sensitivity cases are required");
  return {
    owner: text(row["owner"], "owner"),
    direction: parseMatrix(row["tangent"], "tangent"),
    cases,
    boundary: text(row["claim_boundary"], "claim_boundary"),
  };
}
