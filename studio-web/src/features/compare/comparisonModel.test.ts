// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable comparison semantics and independent scalar oracle

import { expect, it } from "vitest";
import { compareImmutableRuns, comparisonKeyFields } from "./comparisonModel";
import type { ComparisonSnapshot } from "./comparisonModel";

const baseline: ComparisonSnapshot = Object.freeze({
  revisionHash: "a".repeat(64),
  runHash: "b".repeat(64),
  semantics: Object.freeze({
    program: "original",
    parameters: Object.freeze({ gain: 2n }),
    settings: Object.freeze({ shots: 100n }),
    evidence: ["original"],
  }),
  key: Object.freeze({
    estimand: "R",
    unit: "1",
    model: "classical Kuramoto",
    dataset: "original input",
    backend: "original wasm",
    precision: "float64",
    shotProtocol: "none: deterministic integration",
    calibration: "none: local model",
    uncertainty: "not estimated",
  }),
  observations: Object.freeze([{ key: "order", time: 0, value: 2 }]),
  unavailableReason: null,
});

it("test_immutable_run_comparison_04: reports independently known right-minus-left scalar delta without modifying either source", () => {
  const right = { ...baseline, observations: [{ key: "order", time: 0, value: 5 }] };
  const original = structuredClone([baseline, right]);
  const result = compareImmutableRuns(baseline, right);
  expect(result.blockers).toEqual([]);
  expect(result.rows).toEqual([
    {
      key: "order",
      time: 0,
      status: "matched",
      baseline: 2,
      candidate: 5,
      delta: 3,
      deltaState: "available",
    },
  ]);
  expect(result.matched).toBe(1);
  expect(result.baselineOnly).toBe(0);
  expect(result.candidateOnly).toBe(0);
  expect(result.semanticDiff).toEqual([]);
  expect([baseline, right]).toEqual(original);
  expect(Object.isFrozen(result.rows[0])).toBe(true);
});

it.each(comparisonKeyFields)(
  "test_immutable_run_comparison_01: blocks arithmetic when the declared %s differs",
  (field) => {
    const candidate = {
      ...baseline,
      key: { ...baseline.key, [field]: "incompatible" },
      observations: [{ key: "order", time: 0, value: 5 }],
    };
    const result = compareImmutableRuns(baseline, candidate);
    expect(result.blockers).toEqual([`${field} differs`]);
    expect(result.rows[0]).toMatchObject({
      baseline: 2,
      candidate: 5,
      delta: null,
      deltaState: "blocked",
    });
  },
);

it("test_immutable_run_comparison_02: exact object/time matching preserves every unmatched original observation", () => {
  const left = {
    ...baseline,
    observations: [
      { key: "order", time: 0, value: 2 },
      { key: "order", time: 0.1, value: 3 },
    ],
  };
  const right = {
    ...baseline,
    observations: [
      { key: "order", time: 0, value: 5 },
      { key: "order", time: 0.2, value: 7 },
      { key: "another", time: 0, value: 8 },
    ],
  };
  const result = compareImmutableRuns(left, right);
  expect(
    result.rows.map((row) => [
      row.key,
      row.time,
      row.status,
      row.baseline,
      row.candidate,
      row.delta,
    ]),
  ).toEqual([
    ["order", 0, "matched", 2, 5, 3],
    ["order", 0.1, "baseline-only", 3, null, null],
    ["order", 0.2, "candidate-only", null, 7, null],
    ["another", 0, "candidate-only", null, 8, null],
  ]);
  expect([result.matched, result.baselineOnly, result.candidateOnly]).toEqual([1, 1, 2]);
});

it("exposes exact semantic changes including absent keys, signed zero and lossless integers", () => {
  const left = {
    ...baseline,
    semantics: { program: "old", unchanged: { count: 9007199254740993n }, sign: -0, removed: null },
  };
  const right = {
    ...baseline,
    semantics: { program: "new", unchanged: { count: 9007199254740993n }, sign: 0, added: 100n },
  };
  expect(compareImmutableRuns(left, right).semanticDiff).toEqual([
    { field: "added", baseline: undefined, candidate: 100n },
    { field: "program", baseline: "old", candidate: "new" },
    { field: "removed", baseline: null, candidate: undefined },
    { field: "sign", baseline: -0, candidate: 0 },
  ]);
});

it("preserves signed zero in coordinates and values without treating an adjacent float as an exact match", () => {
  const left = {
    ...baseline,
    observations: [
      { key: "order", time: -0, value: -0 },
      { key: "order", time: 0.1, value: 2 },
    ],
  };
  const right = {
    ...baseline,
    observations: [
      { key: "order", time: 0, value: 0 },
      { key: "order", time: 0.10000000000000002, value: 5 },
    ],
  };
  const result = compareImmutableRuns(left, right);
  expect(result.matched).toBe(0);
  expect(Object.is(result.rows[0]?.time, -0)).toBe(true);
  expect(Object.is(result.rows[0]?.baseline, -0)).toBe(true);
  expect(result.rows.length).toBe(4);
});

it("retains unavailable source values and refuses a finite-input overflowing difference", () => {
  const missing = compareImmutableRuns(
    { ...baseline, observations: [{ key: "order", time: 0, value: null }] },
    baseline,
  );
  expect(missing.rows[0]).toMatchObject({ delta: null, deltaState: "unavailable" });
  const overflow = compareImmutableRuns(
    { ...baseline, observations: [{ key: "order", time: 0, value: -Number.MAX_VALUE }] },
    { ...baseline, observations: [{ key: "order", time: 0, value: Number.MAX_VALUE }] },
  );
  expect(overflow.rows[0]).toMatchObject({ delta: null, deltaState: "unavailable" });
  const unavailable = compareImmutableRuns(
    { ...baseline, unavailableReason: "cancelled attempt", observations: [] },
    { ...baseline, unavailableReason: "partial evidence", observations: [] },
  );
  expect(unavailable.blockers).toEqual([
    "Baseline: cancelled attempt",
    "Candidate: partial evidence",
  ]);
  expect(unavailable.rows).toEqual([]);
});

it.each(
  [
    [
      { key: "order", time: 0, value: 1 },
      { key: "order", time: 0, value: 2 },
    ],
    [{ key: "order", time: Infinity, value: 1 }],
    [{ key: "order", time: 0, value: NaN }],
    [{ key: "", time: 0, value: 1 }],
    Array.from({ length: 65537 }, (_, index) => ({ key: "order", time: index, value: 1 })),
  ].map((observations) => ({ observations })),
)("refuses malformed or unbounded original observations before comparison", ({ observations }) => {
  expect(() => compareImmutableRuns({ ...baseline, observations }, baseline)).toThrow(
    "Bounded unique finite original observations required",
  );
});

it("requires complete explicit comparability metadata and original canonical semantic data", () => {
  expect(() =>
    compareImmutableRuns({ ...baseline, key: { ...baseline.key, unit: "" } }, baseline),
  ).toThrow("Complete explicit comparison key required");
  expect(() =>
    compareImmutableRuns({ ...baseline, semantics: { invalid: () => 1 } }, baseline),
  ).toThrow();
});
