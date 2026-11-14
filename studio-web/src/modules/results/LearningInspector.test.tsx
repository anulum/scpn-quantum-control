// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — learning custody, proxy labels and contradiction refusals

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import committed from "../../../../data/studio/learning_instrument_20261114.json";
import { LearningInspector } from "./LearningInspector";
import { learningInstrument, parseLearningInstrument } from "./learningInstrument";

afterEach(cleanup);

type Raw = ReturnType<typeof copy>;
const copy = () => structuredClone(committed);
const firstDomain = (data: Raw) =>
  data.templates.domain_proxies[0] as (typeof committed.templates.domain_proxies)[number];

function exportText(): string {
  const link = screen.getByRole("link", { name: "Download proxy metadata" });
  return decodeURIComponent((link.getAttribute("href") ?? "").split(",").slice(1).join(","));
}

it("test_learning_domain_instrument_01: held-out records cannot enter training selection", () => {
  render(<LearningInspector instrument={learningInstrument} />);
  const custody = screen.getByLabelText("Held-out custody");
  expect(
    within(custody).getByText("Held-out trajectories in training").nextSibling?.textContent,
  ).toBe("0");
  expect(custody.textContent).toContain("forecaster fitting requires a train batch");
  expect(screen.getByLabelText("QSNN held-out evaluation").textContent).toContain(
    "train/test overlap: held-out samples [0, 1]",
  );
  const data = copy();
  data.templates.forecasting.held_out.overlapping_samples = 1;
  expect(parseLearningInstrument(data)).toEqual({
    ok: false,
    reason: "Held-out records cannot enter training",
  });
  const other = copy();
  other.templates.forecasting.held_out.model_training_batch_digest =
    other.templates.forecasting.dataset.splits[2]?.content_digest ?? "";
  expect(parseLearningInstrument(other).ok).toBe(false);
});

it("test_learning_domain_instrument_02: a reloaded checkpoint keeps seed and optimiser identity", () => {
  render(<LearningInspector instrument={learningInstrument} />);
  const checkpoint = screen.getByLabelText("Checkpoint identity");
  expect(checkpoint.textContent).toContain("qsnn_training_checkpoint.v2");
  expect(
    within(checkpoint).getByText("Resumed run reproduces the uninterrupted run").nextSibling
      ?.textContent,
  ).toBe("yes");
  expect(screen.getByLabelText("QSNN data and optimiser").textContent).toContain("Layer seed11");
  const mismatched = copy();
  mismatched.templates.qsnn.checkpoint.learning_rate = 0.3;
  expect(parseLearningInstrument(mismatched).ok).toBe(false);
  const diverged = copy();
  diverged.templates.qsnn.checkpoint.resumed_matches_uninterrupted = false;
  expect(parseLearningInstrument(diverged)).toEqual({
    ok: false,
    reason: "A resumed checkpoint must reproduce the uninterrupted run",
  });
});

it("test_learning_domain_instrument_03: invalid covariance shows the owner's refusal", () => {
  render(<LearningInspector instrument={learningInstrument} />);
  const refusals = screen.getByRole("list", { name: "Owner covariance refusals" });
  expect(refusals.textContent).toContain(
    "Refused negative covariance: sigma_q must be positive definite",
  );
  expect(refusals.textContent).toContain("sigma_q must have shape (1, 1), got (2, 2)");
  const silent = copy();
  silent.templates.free_energy.refusals = [];
  expect(parseLearningInstrument(silent)).toEqual({
    ok: false,
    reason: "The free-energy owner's covariance refusals are required",
  });
});

it("test_learning_domain_instrument_04: proxy labels stay in the plot and the export", () => {
  render(<LearningInspector instrument={learningInstrument} />);
  expect(
    screen.getByRole("img", { name: "eeg_alpha_plv_8ch proxy metrics, not physical outcomes" })
      .textContent,
  ).toContain("topology_similarity_proxy (proxy):");
  const exported = JSON.parse(exportText());
  expect(exported.evidence_class).toBe("domain proxy metric");
  expect(exported.physical_outcome_in_custody).toBe(false);
  expect(exported.publication_safe).toBe(false);
  expect(exported.claims_forbidden).toContain("clinical validation");
  fireEvent.change(screen.getByLabelText("Domain benchmark"), { target: { value: "2" } });
  expect(JSON.parse(exportText()).claims_forbidden).toContain("real plant control");
  expect(
    screen.getByRole("list", { name: "Claims this benchmark never supports" }).textContent,
  ).toContain("ITER disruption prediction");
  const promoted = copy();
  firstDomain(promoted).kit.publication_safe = true;
  expect(parseLearningInstrument(promoted).ok).toBe(false);
  const physical = copy();
  firstDomain(physical).physical_outcome_in_custody = true;
  expect(parseLearningInstrument(physical)).toEqual({
    ok: false,
    reason: "A domain proxy cannot carry a measured physical outcome",
  });
});

it("test_learning_domain_instrument_05: no global random-state reset is admitted", () => {
  render(<LearningInspector instrument={learningInstrument} />);
  expect(
    screen.getByRole("region", { name: "Learning, forecasting and domain benchmarks" }).textContent,
  ).toContain("global random state was left unchanged");
  const reset = copy();
  reset.global_random_state_unchanged = false;
  expect(parseLearningInstrument(reset)).toEqual({
    ok: false,
    reason: "The build must leave the caller's global random state unchanged",
  });
});

it("shows an absent NQS baseline as unavailable and recovers after a refusal", () => {
  const { rerender } = render(
    <LearningInspector instrument={{ ok: false, reason: "unsupported source schema" }} />,
  );
  expect(screen.getByRole("alert").textContent).toContain("unsupported source schema");
  const absent = copy() as Record<string, unknown> & Raw;
  (absent.templates as Record<string, unknown>)["nqs"] = { status: "unavailable" };
  rerender(<LearningInspector instrument={parseLearningInstrument(absent)} />);
  expect(screen.getByRole("status").textContent).toContain("No committed JAX NQS evidence");
  rerender(<LearningInspector instrument={learningInstrument} />);
  expect(screen.getByRole("img", { name: /Variational energy per iteration/ })).toBeTruthy();
  expect(screen.getByLabelText("NQS evidence").textContent).toContain("seed 7");
});

it("falls back to the first benchmark when refreshed data removes the selected one", () => {
  const { rerender } = render(<LearningInspector instrument={learningInstrument} />);
  fireEvent.change(screen.getByLabelText("Domain benchmark"), { target: { value: "2" } });
  const data = copy();
  data.templates.domain_proxies = [firstDomain(data)];
  rerender(<LearningInspector instrument={parseLearningInstrument(data)} />);
  expect(JSON.parse(exportText()).dataset_id).toBe("eeg_alpha_plv_8ch");
});

it.each([
  (data: Raw) => {
    data.schema = "future";
  },
  (data: Raw) => {
    data.templates.forecasting.dataset.splits.reverse();
  },
  (data: Raw) => {
    data.templates.qsnn.steps[0] = { ...data.templates.qsnn.steps[0], index: 4 } as never;
  },
  (data: Raw) => {
    (data.templates.nqs as Record<string, unknown>)["status"] = "pending";
  },
  (data: Raw) => {
    data.templates.nqs.evidence_sha256 = "bad";
  },
  (data: Raw) => {
    data.templates.nqs.comparison.energy_history = [];
  },
  (data: Raw) => {
    data.templates.nqs.comparison.variational_energy = 0;
  },
  (data: Raw) => {
    firstDomain(data).kit.claims_forbidden = [];
  },
  (data: Raw) => {
    data.templates.domain_proxies.push(firstDomain(data));
  },
  (data: Raw) => {
    data.templates.domain_proxies = [];
  },
  (data: Raw) => {
    firstDomain(data).proxy_metrics = {} as never;
  },
])("refuses malformed or contradictory source data", (mutate) => {
  const data = copy();
  mutate(data);
  expect(parseLearningInstrument(data).ok).toBe(false);
});

it("reports a model that does not beat persistence and a flat energy history as recorded", () => {
  const data = copy();
  data.templates.forecasting.test.lower_mse_than_persistence = false;
  const comparison = data.templates.nqs.comparison;
  comparison.energy_history = [comparison.variational_energy];
  comparison.exact_ground_energy = comparison.variational_energy;
  render(<LearningInspector instrument={parseLearningInstrument(data)} />);
  expect(
    within(screen.getByLabelText("Classical reference on held-out data")).getByText(
      "Lower error than persistence",
    ).nextSibling?.textContent,
  ).toBe("no");
  const plot = screen.getByRole("img", { name: /Variational energy per iteration/ });
  expect(plot.querySelector('[data-series="exact"]')?.getAttribute("y1")).toBe("60");
});

it("keeps a missing held-out loss visible instead of inventing one", () => {
  const data = copy();
  (data.templates.qsnn.held_out as Record<string, unknown>)["in_distribution_loss"] = null;
  render(<LearningInspector instrument={parseLearningInstrument(data)} />);
  expect(screen.getByLabelText("QSNN held-out evaluation").textContent).toContain(
    "no in-distribution sample",
  );
});
