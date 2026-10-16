// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native retained-population comparison admission

import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";

/** Exact native comparison domain; content identity grants no producer authentication. */
export const METHOD_COMPARISON_SCHEMA = "studio.method_comparison.v1";
/** Native UTF-8 input ceiling; complete populations are never truncated. */
export const MAX_COMPARISON_BYTES = 4 * 1024 * 1024;
/** Original numeric token, preserving exact integers and signed binary64 zero. */
export type ComparisonScalar = number | bigint;
/** Every explicit source-owned scientific meaning; shots is separately integral. */
export const methodMeaningFields = [
  "problem",
  "observable",
  "unit",
  "model",
  "precision",
  "shots",
  "shot_protocol",
  "initial_condition",
  "calibration",
  "noise",
  "estimator",
] as const;

/** Source cost allocation and resource bounds, including explicit unknown components. */
export interface MethodBudget {
  /** Original measured or declared compilation milliseconds. */ readonly compilation_ms: ComparisonScalar | null;
  /** Original measured or declared data movement milliseconds. */ readonly transfer_ms: ComparisonScalar | null;
  /** Original measured or declared execution milliseconds. */ readonly execution_ms: ComparisonScalar | null;
  /** Original exact byte observation, never coerced to Number. */ readonly peak_bytes:
    | bigint
    | null;
  /** Original exact admitted memory ceiling. */ readonly memory_limit_bytes: bigint | null;
  /** Original total wall-time ceiling in milliseconds. */ readonly wall_limit_ms: ComparisonScalar | null;
  /** Original unallocated advisory timer; it cannot fill a cost component. */ readonly observed_wall_ms: ComparisonScalar | null;
  /** Full upstream sum, or unavailable; never recomputed in a view. */ readonly total_cost_ms: ComparisonScalar | null;
  /** Upstream resource admission result. */ readonly within_budget: boolean;
}
/** Original estimator interval, distinct from repeated-run variance or a coverage proof. */
export interface MethodInterval {
  /** Original estimator point. */ readonly point: ComparisonScalar;
  /** Original estimator standard error. */ readonly standard_error: ComparisonScalar;
  /** Original lower endpoint. */ readonly low: ComparisonScalar;
  /** Original upper endpoint. */ readonly high: ComparisonScalar;
  /** Nominal source confidence fraction, not empirical coverage. */ readonly coverage: ComparisonScalar;
  /** Original analytic or bootstrap method identifier. */ readonly method: string;
  /** Exact original measurement count. */ readonly n_shots: bigint;
  /** Exact declared bootstrap replication count. */ readonly n_resamples: bigint;
}
/** A complete original repetition; failures, negative values and seed identity remain visible. */
export interface MethodRepetition {
  /** Original unique repetition identity. */ readonly run_id: string;
  /** Original method identity. */ readonly method: string;
  /** Original declared execution backend. */ readonly backend: string;
  /** Exact seed, including integers beyond the Number boundary. */ readonly seed: bigint;
  /** Source declarations; no model or estimator equivalence is inferred here. */ readonly meaning: Readonly<
    Record<(typeof methodMeaningFields)[number], string | bigint | null>
  >;
  /** Complete original cost and resource declarations. */ readonly budget: MethodBudget;
  /** Original availability state. */ readonly status: "completed" | "failed" | "missing";
  /** Unchanged finite scalar, or unavailable. */ readonly value: ComparisonScalar | null;
  /** Original failure or missing-data explanation. */ readonly reason: string | null;
  /** Unmodified original interval, or not estimated. */ readonly interval: MethodInterval | null;
}
/** Source aggregate over every declared repetition under its explicit missing policy. */
export interface MethodPopulationSummary {
  /** Original method identity. */ readonly method: string;
  /** All declared repetitions, including failed and missing rows. */ readonly population_size: bigint;
  /** Original nonmissing denominator. */ readonly observed_size: bigint;
  /** Every included original identity, in source order. */ readonly included_run_ids: readonly string[];
  /** Every failed/missing original identity, in source order. */ readonly missing_run_ids: readonly string[];
  /** Upstream mean; selection never replaces this number. */ readonly mean: ComparisonScalar | null;
  /** Upstream n-1 variance, or unavailable. */ readonly sample_variance: ComparisonScalar | null;
  /** Upstream full population cost, including failures, or unknown. */ readonly total_cost_ms: ComparisonScalar | null;
  /** Upstream candidate-minus-reference result, only when admitted. */ readonly difference_vs_reference: ComparisonScalar | null;
}
/** Full source-owned comparison; the browser performs no aggregation or scientific ranking. */
export interface NativeMethodComparison {
  /** Exact source schema version. */ readonly schema: typeof METHOD_COMPARISON_SCHEMA;
  /** Explicit original reference method. */ readonly baseline_method: string;
  /** Explicit original denominator policy. */ readonly missing_data_policy:
    | "require-complete"
    | "available-case";
  /** Complete ordered population identities. */ readonly population_run_ids: readonly string[];
  /** Source display-only selection. */ readonly selected_run_id: string | null;
  /** Upstream semantic/population/resource comparability result. */ readonly ranking_allowed: boolean;
  /** Upstream full-resource comparison result. */ readonly cost_comparison_allowed: boolean;
  /** All original reasons refusing comparison. */ readonly blockers: readonly string[];
  /** Every method's original aggregate. */ readonly methods: readonly MethodPopulationSummary[];
  /** Every original repetition, without filtering. */ readonly runs: readonly MethodRepetition[];
  /** Original qualification and no-advantage boundary. */ readonly claim_boundary: string;
}
/** Immutable admitted export and exact portable original text. */
export interface MethodComparisonArtifact {
  /** Original JSON for exact download. */ readonly text: string;
  /** Canonical content identity, not a producer credential. */ readonly sha256: string;
  /** Source declaration and complete native result. */ readonly body: {
    /** Declared native owner, never dynamically loaded or executed. */ readonly producer: string;
    /** Original no-submission property. */ readonly no_submit: true;
    /** Source-owned statistics and all retained observations. */ readonly comparison: NativeMethodComparison;
  };
}
/** Explicit admission outcome; a refusal cannot replace the prior admitted result. */
export type MethodComparisonAdmission =
  | {
      /** Successful bounded content admission. */ readonly ok: true;
      /** Unchanged admitted source values. */
      readonly value: MethodComparisonArtifact;
    }
  | {
      /** Invalid replacement. */ readonly ok: false;
      /** Observable refusal, preserving the prior admitted source. */
      readonly message: string;
    };

function require(condition: unknown): asserts condition {
  if (!condition) throw new Error("Method comparison refused");
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function fields(value: unknown, names: readonly string[]): Record<string, unknown> {
  require(isRecord(value));
  require(
    Object.keys(value).length === names.length && names.every((key) => Object.hasOwn(value, key)),
  );
  return value;
}
function text(value: unknown, maximum = 2048): string {
  require(typeof value === "string" && value.trim().length > 0 && [...value].length <= maximum);
  return value;
}
function scalar(value: unknown): ComparisonScalar {
  require(typeof value === "number" || typeof value === "bigint");
  require(Number.isFinite(Number(value)));
  return value;
}
function nullableScalar(value: unknown): void {
  if (value !== null) scalar(value);
}
function count(value: unknown): bigint {
  require(typeof value === "bigint" && value >= 0n);
  return value;
}
function strings(value: unknown): string[] {
  require(Array.isArray(value));
  return value.map((item: unknown) => text(item));
}
function repetition(value: unknown): Record<string, unknown> {
  const row = fields(value, [
    "run_id",
    "method",
    "backend",
    "seed",
    "meaning",
    "budget",
    "status",
    "value",
    "reason",
    "interval",
  ]);
  for (const key of ["run_id", "method", "backend"]) text(row[key]);
  count(row["seed"]);
  const meaning = fields(row["meaning"], methodMeaningFields);
  for (const key of methodMeaningFields) {
    if (key === "shots") {
      if (meaning[key] !== null) count(meaning[key]);
    } else text(meaning[key]);
  }
  const budget = fields(row["budget"], [
    "compilation_ms",
    "transfer_ms",
    "execution_ms",
    "peak_bytes",
    "memory_limit_bytes",
    "wall_limit_ms",
    "observed_wall_ms",
    "total_cost_ms",
    "within_budget",
  ]);
  for (const key of [
    "compilation_ms",
    "transfer_ms",
    "execution_ms",
    "wall_limit_ms",
    "observed_wall_ms",
    "total_cost_ms",
  ])
    nullableScalar(budget[key]);
  for (const key of ["peak_bytes", "memory_limit_bytes"])
    if (budget[key] !== null) count(budget[key]);
  require(typeof budget["within_budget"] === "boolean");
  const status = row["status"];
  require(status === "completed" || status === "failed" || status === "missing");
  if (status === "completed") {
    scalar(row["value"]);
    require(row["reason"] === null);
  } else {
    require(row["value"] === null && row["interval"] === null);
    text(row["reason"]);
  }
  if (row["interval"] !== null) {
    const interval = fields(row["interval"], [
      "point",
      "standard_error",
      "low",
      "high",
      "coverage",
      "method",
      "n_shots",
      "n_resamples",
    ]);
    for (const key of ["point", "standard_error", "low", "high", "coverage"]) scalar(interval[key]);
    text(interval["method"]);
    count(interval["n_shots"]);
    count(interval["n_resamples"]);
  }
  return row;
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

/** Admit original native values, exact graph conservation and content identity without estimating statistics. */
export async function parseMethodComparison(raw: string): Promise<MethodComparisonAdmission> {
  try {
    require(
      raw.length <= MAX_COMPARISON_BYTES &&
        new TextEncoder().encode(raw).length <= MAX_COMPARISON_BYTES,
    );
    const wire = fields(readJson(raw), ["schema", "body", "extensions", "sha256"]);
    require(wire["schema"] === METHOD_COMPARISON_SCHEMA);
    fields(wire["extensions"], []);
    const body = fields(wire["body"], ["producer", "comparison", "no_submit"]);
    require(
      body["producer"] ===
        "scpn_quantum_control.benchmarks.method_comparison.summarise_method_comparison" &&
        body["no_submit"] === true,
    );
    const result = fields(body["comparison"], [
      "schema",
      "baseline_method",
      "missing_data_policy",
      "population_run_ids",
      "selected_run_id",
      "ranking_allowed",
      "cost_comparison_allowed",
      "blockers",
      "methods",
      "runs",
      "claim_boundary",
    ]);
    require(result["schema"] === METHOD_COMPARISON_SCHEMA);
    text(result["baseline_method"]);
    text(result["claim_boundary"], 4096);
    require(
      result["missing_data_policy"] === "require-complete" ||
        result["missing_data_policy"] === "available-case",
    );
    require(
      typeof result["ranking_allowed"] === "boolean" &&
        typeof result["cost_comparison_allowed"] === "boolean",
    );
    require(Array.isArray(result["blockers"]));
    const blockers = result["blockers"].map((value: unknown) => text(value, 4096));
    require(result["ranking_allowed"] === (blockers.length === 0));
    require(result["cost_comparison_allowed"] !== true || result["ranking_allowed"] === true);
    const rawRuns = result["runs"];
    require(Array.isArray(rawRuns) && rawRuns.length > 0 && rawRuns.length <= 1000);
    const runs = rawRuns.map(repetition),
      identities = runs.map((row) => text(row["run_id"]));
    const declared = strings(result["population_run_ids"]);
    require(
      new Set(identities).size === identities.length &&
        declared.length === identities.length &&
        declared.every((id, i) => id === identities[i]),
    );
    if (result["selected_run_id"] !== null)
      require(identities.includes(text(result["selected_run_id"])));
    const methods = result["methods"];
    require(Array.isArray(methods) && methods.length > 0 && methods.length <= runs.length);
    const methodIds: string[] = [];
    for (const original of methods) {
      const row = fields(original, [
        "method",
        "population_size",
        "observed_size",
        "included_run_ids",
        "missing_run_ids",
        "mean",
        "sample_variance",
        "total_cost_ms",
        "difference_vs_reference",
      ]);
      const method = text(row["method"]),
        group = runs.filter((run) => run["method"] === method);
      require(!methodIds.includes(method) && group.length > 0);
      methodIds.push(method);
      const included = strings(row["included_run_ids"]),
        missing = strings(row["missing_run_ids"]);
      const observed = group
          .filter((run) => run["status"] === "completed")
          .map((run) => run["run_id"]),
        absent = group.filter((run) => run["status"] !== "completed").map((run) => run["run_id"]);
      require(
        count(row["population_size"]) === BigInt(group.length) &&
          count(row["observed_size"]) === BigInt(observed.length),
      );
      require(
        included.length === observed.length &&
          included.every((id, i) => id === observed[i]) &&
          missing.length === absent.length &&
          missing.every((id, i) => id === absent[i]),
      );
      for (const key of ["mean", "sample_variance", "total_cost_ms", "difference_vs_reference"])
        nullableScalar(row[key]);
    }
    require(
      methodIds.includes(text(result["baseline_method"])) &&
        runs.every((run) => methodIds.includes(text(run["method"]))),
    );
    const digest = text(wire["sha256"]);
    require(/^[0-9a-f]{64}$/.test(digest));
    require(
      (await canonicalDigest(METHOD_COMPARISON_SCHEMA, {
        schema: wire["schema"],
        body,
        extensions: wire["extensions"],
      })) === digest,
    );
    return {
      ok: true,
      value: freeze({
        text: raw,
        sha256: digest,
        body: body as unknown as MethodComparisonArtifact["body"],
      }),
    };
  } catch {
    return {
      ok: false,
      message:
        "Method comparison refused; the previous admitted comparison and workspace remain unchanged.",
    };
  }
}
