// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact read-only observation comparison

import { canonicalBytes } from "../../shared/contracts/canonical";
import { rawResultText, resultLimits } from "../results/resultModel";

/** Every meaning that must agree before scalar subtraction; no inferred equivalence. */
export const comparisonKeyFields = Object.freeze([
  "estimand",
  "unit",
  "model",
  "dataset",
  "backend",
  "precision",
  "shotProtocol",
  "calibration",
  "uncertainty",
] as const);
/** Explicit producer-bound comparison meanings; absence cannot authorise arithmetic. */
export type ComparisonKey = Readonly<Record<(typeof comparisonKeyFields)[number], string>>;
/** One unchanged producer value on its declared object and time grid. */
export interface ComparisonObservation {
  /** Explicit source object identity, never a sample index used as time. */ readonly key: string;
  /** Original finite time coordinate, preserving signed zero. */ readonly time: number;
  /** Original binary64 scalar, or explicit unavailable value. */ readonly value: number | null;
}
/** Read-only projection of one admitted immutable source; grants no execution authority. */
export interface ComparisonSnapshot {
  /** Original immutable revision identity. */ readonly revisionHash: string;
  /** Original run document identity, or no selected run. */ readonly runHash: string | null;
  /** Original semantic fields; canonical exact values precede numeric comparison. */ readonly semantics: Readonly<
    Record<string, unknown>
  >;
  /** Explicit comparable meaning including estimator/unit/model/environment. */ readonly key: ComparisonKey;
  /** Original ordered source observations, including unmatched samples. */ readonly observations: readonly ComparisonObservation[];
  /** Explicit absent/unsupported/partial result reason; null means available original output. */ readonly unavailableReason:
    | string
    | null;
}
/** Original semantic change; undefined denotes a field absent from that source. */
export interface SemanticDifference {
  /** Stable semantic field name. */ readonly field: string;
  /** Original baseline value, including exact integer/IEEE values. */ readonly baseline: unknown;
  /** Original candidate value, including exact integer/IEEE values. */ readonly candidate: unknown;
}
/** One exact object/time pair or unchanged unmatched observation. */
export interface ComparisonRow {
  /** Original object identity. */ readonly key: string;
  /** Original exact time coordinate. */ readonly time: number;
  /** Exact matching status, independent of scientific comparability. */ readonly status:
    | "matched"
    | "baseline-only"
    | "candidate-only";
  /** Original baseline scalar; null never implies zero. */ readonly baseline: number | null;
  /** Original candidate scalar; null never implies zero. */ readonly candidate: number | null;
  /** Candidate minus baseline only for comparable finite matched values. */ readonly delta:
    | number
    | null;
  /** Arithmetic availability; blocked meanings never produce a difference. */ readonly deltaState:
    | "available"
    | "blocked"
    | "unavailable";
}
/** Complete bounded comparison; no aggregate ranking or scientific promotion. */
export interface ImmutableRunComparison {
  /** All original semantic changes shown before arithmetic. */ readonly semanticDiff: readonly SemanticDifference[];
  /** Every incompatible meaning or unavailable source reason. */ readonly blockers: readonly string[];
  /** Explicit supported alignment; no interpolation or nearest-time substitution. */ readonly alignment: "exact-object-time";
  /** Stable baseline order followed by all unmatched candidate observations. */ readonly rows: readonly ComparisonRow[];
  /** Exact object/time pairs, including pairs whose arithmetic is blocked. */ readonly matched: number;
  /** Original observations present only in baseline. */ readonly baselineOnly: number;
  /** Original observations present only in candidate. */ readonly candidateOnly: number;
}

function identity(observation: ComparisonObservation): string {
  return JSON.stringify([observation.key, rawResultText(observation.time)]);
}
function checkedObservations(
  source: ComparisonSnapshot,
): ReadonlyMap<string, ComparisonObservation> {
  if (source.observations.length > resultLimits.samples)
    throw new Error("Bounded unique finite original observations required");
  const observations = new Map<string, ComparisonObservation>();
  for (const observation of source.observations) {
    if (
      typeof observation.key !== "string" ||
      observation.key.length === 0 ||
      observation.key.length > 2048 ||
      !Number.isFinite(observation.time) ||
      (observation.value !== null && !Number.isFinite(observation.value)) ||
      observations.has(identity(observation))
    )
      throw new Error("Bounded unique finite original observations required");
    observations.set(identity(observation), observation);
  }
  return observations;
}
function same(left: unknown, right: unknown): boolean {
  if (left === undefined || right === undefined) return left === right;
  const a = canonicalBytes("studio.comparison-semantics.v1", left),
    b = canonicalBytes("studio.comparison-semantics.v1", right);
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

/** Compare original immutable metadata and scalar observations with exact object/time alignment.
 *
 * Canonical semantic equality retains integer precision and signed zero. All key
 * meanings must agree before candidate-minus-baseline arithmetic. Every unmatched
 * observation remains visible; unavailable or overflowing differences stay null.
 * Neither source is modified and no solver, estimator or provider is invoked.
 * Throws before producing a comparison for incomplete keys or malformed samples.
 */
export function compareImmutableRuns(
  baseline: ComparisonSnapshot,
  candidate: ComparisonSnapshot,
): ImmutableRunComparison {
  canonicalBytes("studio.comparison-semantics.v1", baseline.semantics);
  canonicalBytes("studio.comparison-semantics.v1", candidate.semantics);
  const left = checkedObservations(baseline),
    right = checkedObservations(candidate);
  const blockers: string[] = [];
  for (const field of comparisonKeyFields) {
    if (
      typeof baseline.key[field] !== "string" ||
      baseline.key[field].length === 0 ||
      typeof candidate.key[field] !== "string" ||
      candidate.key[field].length === 0
    )
      throw new Error("Complete explicit comparison key required");
    if (baseline.key[field] !== candidate.key[field]) blockers.push(`${field} differs`);
  }
  if (baseline.unavailableReason !== null) blockers.push(`Baseline: ${baseline.unavailableReason}`);
  if (candidate.unavailableReason !== null)
    blockers.push(`Candidate: ${candidate.unavailableReason}`);
  const fields = [
    ...new Set([...Object.keys(baseline.semantics), ...Object.keys(candidate.semantics)]),
  ].sort();
  const semanticDiff = fields
    .filter((field) => !same(baseline.semantics[field], candidate.semantics[field]))
    .map((field) =>
      Object.freeze({
        field,
        baseline: baseline.semantics[field],
        candidate: candidate.semantics[field],
      }),
    );
  const rows: ComparisonRow[] = [];
  const row = (
    original: ComparisonObservation,
    a: ComparisonObservation | undefined,
    b: ComparisonObservation | undefined,
  ): ComparisonRow => {
    const observation = original;
    const status =
      a === undefined ? "candidate-only" : b === undefined ? "baseline-only" : "matched";
    const difference =
      blockers.length === 0 &&
      a?.value !== null &&
      b?.value !== null &&
      a !== undefined &&
      b !== undefined
        ? b.value - a.value
        : null;
    const delta = difference !== null && Number.isFinite(difference) ? difference : null;
    return Object.freeze({
      key: observation.key,
      time: observation.time,
      status,
      baseline: a?.value ?? null,
      candidate: b?.value ?? null,
      delta,
      deltaState: blockers.length !== 0 ? "blocked" : delta === null ? "unavailable" : "available",
    });
  };
  for (const [key, observation] of left) rows.push(row(observation, observation, right.get(key)));
  for (const [key, observation] of right)
    if (!left.has(key)) rows.push(row(observation, undefined, observation));
  return Object.freeze({
    semanticDiff: Object.freeze(semanticDiff),
    blockers: Object.freeze(blockers),
    alignment: "exact-object-time",
    rows: Object.freeze(rows),
    matched: rows.filter((entry) => entry.status === "matched").length,
    baselineOnly: rows.filter((entry) => entry.status === "baseline-only").length,
    candidateOnly: rows.filter((entry) => entry.status === "candidate-only").length,
  });
}
