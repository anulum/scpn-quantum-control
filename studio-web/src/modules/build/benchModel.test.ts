// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — live bench view models against independent invariants

// @vitest-environment node
import { describe, expect, it } from "vitest";
import {
  PHASE_BINS,
  RUN_HISTORY_LIMIT,
  SERIES_COLOURS,
  niceTicks,
  orderVector,
  phaseBin,
  runHistory,
  timeToFraction,
} from "./benchModel";
import type { BenchRun } from "./benchModel";

describe("order vector", () => {
  it("is fully coherent for equal phases and keeps their direction", () => {
    const vector = orderVector([0.7, 0.7, 0.7]);
    expect(vector.r).toBeCloseTo(1, 15);
    expect(vector.psi).toBeCloseTo(0.7, 15);
  });

  it("vanishes for phases spread evenly round the circle", () => {
    const n = 7;
    expect(orderVector(Array.from({ length: n }, (_, k) => (2 * Math.PI * k) / n)).r).toBeLessThan(
      1e-15,
    );
  });

  it("matches the magnitude of the mean unit vector and refuses an empty set", () => {
    const phases = [0, Math.PI / 2];
    expect(orderVector(phases)).toMatchObject({ psi: Math.PI / 4 });
    expect(orderVector(phases).r).toBeCloseTo(Math.SQRT1_2, 15);
    expect(() => orderVector([])).toThrow(RangeError);
  });
});

describe("phase bins", () => {
  it("wraps any real phase into one of the cyclic bins", () => {
    expect(phaseBin(0)).toBe(0);
    expect(phaseBin(2 * Math.PI)).toBe(0);
    expect(phaseBin(-2 * Math.PI)).toBe(0);
    expect(phaseBin(Math.PI)).toBe(PHASE_BINS / 2);
    expect(phaseBin(-Math.PI / 2)).toBe((3 * PHASE_BINS) / 4);
    expect(phaseBin((2 * Math.PI * 15.9) / 16)).toBe(0);
  });
});

describe("axis ticks", () => {
  it("covers the interval on a 1–2–5 ladder", () => {
    expect(niceTicks(0, 1)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(0, 15, 3)).toEqual([0, 5, 10, 15]);
    expect(niceTicks(0, 7, 4)).toEqual([0, 2, 4, 6, 8]);
    expect(niceTicks(0.3, 0.95, 4)).toEqual([0.2, 0.4, 0.6, 0.8, 1]);
    expect(niceTicks(0, 9.5, 1)).toEqual([0, 10]);
  });

  it("returns the single value of an empty interval and refuses bad input", () => {
    expect(niceTicks(2, 2)).toEqual([2]);
    expect(() => niceTicks(Number.NaN, 1)).toThrow(RangeError);
    expect(() => niceTicks(0, 1, 0)).toThrow(RangeError);
  });
});

describe("time to a fraction of the final order parameter", () => {
  it("reports the first sample at or above the threshold", () => {
    expect(timeToFraction([0.1, 0.5, 0.85, 0.9, 1], 0.05, 0.9)).toBeCloseTo(0.15, 15);
    expect(timeToFraction([0.4], 0.05, 0.9)).toBe(0);
    expect(timeToFraction([0.1, 0.5, 0.9], 0.05, 1.5)).toBeNull();
  });

  it("has no time for an empty or non-positive final value", () => {
    expect(timeToFraction([], 0.05, 0.9)).toBeNull();
    expect(timeToFraction([0.3, 0], 0.05, 0.9)).toBeNull();
    expect(timeToFraction([0.3, Number.NaN], 0.05, 0.9)).toBeNull();
  });
});

describe("run history", () => {
  const run = (wallMs: number): Omit<BenchRun, "number" | "colour" | "overlaid"> => ({
    controls: { n: 12, coupling: 2.5, spread: 1, steps: 2, mode: "mean-field" },
    orderParameter: Float64Array.of(0.2, 0.6, 0.9),
    dt: 0.05,
    wallMs,
  });

  it("numbers runs, keeps stable colours, newest first, and bounds the history", () => {
    let runs: readonly BenchRun[] = [];
    for (let index = 0; index < RUN_HISTORY_LIMIT + 2; index += 1)
      runs = runHistory(runs, { type: "record", run: run(index) });
    expect(runs).toHaveLength(RUN_HISTORY_LIMIT);
    expect(runs[0]).toMatchObject({ number: RUN_HISTORY_LIMIT + 2, overlaid: false });
    expect(runs[0]?.colour).toBe((RUN_HISTORY_LIMIT + 1) % SERIES_COLOURS);
    expect(runs.at(-1)?.number).toBe(3);
  });

  it("toggles one run's overlay and clears the history", () => {
    let runs = runHistory([], { type: "record", run: run(1) });
    runs = runHistory(runs, { type: "record", run: run(2) });
    runs = runHistory(runs, { type: "toggle", number: 1 });
    expect(runs.map((entry) => [entry.number, entry.overlaid])).toEqual([
      [2, false],
      [1, true],
    ]);
    expect(runHistory(runs, { type: "clear" })).toEqual([]);
  });
});
