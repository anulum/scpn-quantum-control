// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original native comparison admission tests

import { expect, it } from "vitest";
import captured from "../../../../data/studio/method_comparison_cases.json";
import { readJson } from "../../shared/contracts/jsonTransport";
import { parseMethodComparison } from "./methodComparisonArtifact";

it.each(Object.entries(captured.cases))(
  "admits complete original native %s",
  async (_name, text) => {
    const result = await parseMethodComparison(text);
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(result.message);
    expect(result.value.text).toBe(text);
    const native = readJson(text) as { readonly body: unknown };
    expect(result.value.body).toEqual(native.body);
    expect(Object.isFrozen(result.value.body.comparison.runs)).toBe(true);
  },
);

it("keeps native all-run mean, negative values, full cost and every missing row", async () => {
  const admitted = await parseMethodComparison(captured.cases.population);
  if (!admitted.ok) throw new Error(admitted.message);
  expect(admitted.value.body.comparison.methods[0]?.mean).toBe(1);
  expect(admitted.value.body.comparison.methods[1]?.mean).toBe(2);
  expect(admitted.value.body.comparison.methods[0]?.total_cost_ms).toBe(30);
  expect(admitted.value.body.comparison.runs[0]?.value).toBe(-1);
  const missing = await parseMethodComparison(captured.cases["missing-population"]);
  if (!missing.ok) throw new Error(missing.message);
  expect(missing.value.body.comparison.runs).toHaveLength(6);
  expect(missing.value.body.comparison.ranking_allowed).toBe(false);
  expect(missing.value.body.comparison.methods[1]?.missing_run_ids).toEqual(["candidate-1"]);
});

it.each(["model-mismatch", "precision-mismatch", "unknown-cost", "budget-exceeded"] as const)(
  "retains the native comparison refusal for %s",
  async (name) => {
    const result = await parseMethodComparison(captured.cases[name]);
    if (!result.ok) throw new Error(result.message);
    expect(result.value.body.comparison.ranking_allowed).toBe(false);
    expect(result.value.body.comparison.blockers.length).toBeGreaterThan(0);
    expect(result.value.body.comparison.methods[1]?.difference_vs_reference).toBe(null);
  },
);

it("preserves all400 original intervals and separate nominal confidence semantics", async () => {
  const result = await parseMethodComparison(captured.cases.intervals);
  if (!result.ok) throw new Error(result.message);
  expect(result.value.body.comparison.runs).toHaveLength(400);
  expect(result.value.body.comparison.runs[0]?.interval?.coverage).toBe(0.95);
  expect(result.value.body.comparison.runs[0]?.interval?.n_shots).toBe(1000n);
  expect(result.value.sha256).toBe(captured.interval_protocol.artifact_sha256);
  expect(captured.interval_protocol.methods["shot-noise-delta"].observed_coverage).toBe(0.91);
});

it.each([
  "{",
  captured.cases.population.replace("studio.method_comparison.v1", "studio.method_comparison.v2"),
  captured.cases.population.replace('"value":-1.0', '"value":999.0'),
])(
  "refuses malformed/future/modified originals without returning replacement data",
  async (text) => {
    const result = await parseMethodComparison(text);
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error("Invalid replacement admitted");
    expect(result.message).toContain("refused");
  },
);

it.each(["null", "[]", "42"])("refuses a non-object original root %s", async (text) => {
  expect((await parseMethodComparison(text)).ok).toBe(false);
});

it("preserves native negative zero, exact large seeds/resources and persisted display-only selection", async () => {
  const result = await parseMethodComparison(captured.cases["numeric-boundary"]);
  if (!result.ok) throw new Error(result.message);
  const source = result.value.body.comparison;
  expect(source.runs[0]?.seed).toBe(9007199254740993n);
  expect(source.runs[0]?.budget.peak_bytes).toBe(9007199254740993n);
  expect(Object.is(source.runs[0]?.value, -0)).toBe(true);
  expect(source.selected_run_id).toBe("candidate-large-2");
  expect(source.methods[1]?.mean).toBe(2);
  expect(source.runs).toHaveLength(6);
});

it("retains actual original17-oscillator unavailable exact/quantum rows", async () => {
  const result = await parseMethodComparison(captured.cases["original-ode-only"]);
  if (!result.ok) throw new Error(result.message);
  expect(result.value.body.comparison.runs).toHaveLength(3);
  expect(result.value.body.comparison.runs[0]?.status).toBe("missing");
  expect(result.value.body.comparison.runs[0]?.value).toBe(null);
  expect(result.value.body.comparison.runs[0]?.reason).toContain("intentionally not run");
});
