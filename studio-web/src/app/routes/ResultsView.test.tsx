// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original Results evidence guards

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
const faults = vi.hoisted(() => ({ data: false, replay: false }));
vi.mock("../../panel/data", async original => {
  const source = await original<typeof import("../../panel/data")>();
  return {
    ...source,
    get supportMatrix() { return faults.data ? { ok: false, reason: "Matrix source unavailable" } : source.supportMatrix; },
    get gradientPlanExplanations() { return faults.data ? { ok: false, reason: "Gradient source unavailable" } : source.gradientPlanExplanations; },
    get scorecard() { return faults.data ? { ok: false, reason: "Scorecard source unavailable" } : source.scorecard; },
  };
});
vi.mock("../../panel/programAd", async original => {
  const source = await original<typeof import("../../panel/programAd")>();
  return { ...source, get programAdUnit() { return faults.replay ? { ok: false, reason: "Replay source unavailable" } : source.programAdUnit; } };
});
import ResultsView from "./ResultsView";
afterEach(() => { cleanup(); faults.data = false; faults.replay = false; });

it("renders actual original evidence and preserves the viewer's input refusal", () => {
  render(<ResultsView focusInstrument />);
  expect(screen.getByRole("heading", { name: "Results" })).toBeTruthy();
  expect(screen.getByText("Differentiate support explorer")).toBeTruthy();
  expect(screen.getByText("Baseline scorecard")).toBeTruthy();
  expect(document.activeElement?.id).toBe("/results/program-ad-replay");
  fireEvent.change(screen.getByLabelText("Evidence JSON"), { target: { value: "{" } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect snapshot" }));
  expect(screen.getByRole("alert").textContent).toContain("Cannot inspect evidence");
});

it("renders every missing source explicitly while retaining the original evidence input", () => {
  faults.data = true;
  faults.replay = true;
  render(<ResultsView />);
  expect(screen.getAllByRole("alert")).toHaveLength(4);
  for (const reason of ["Matrix source unavailable", "Gradient source unavailable", "Scorecard source unavailable", "Replay source unavailable"]) expect(screen.getAllByRole("alert").some(alert => alert.textContent?.includes(reason))).toBe(true);
  expect(screen.getByLabelText("Evidence JSON")).toBeTruthy();
  expect(screen.queryByText("Baseline scorecard")).toBeNull();
});


it("reaches the original producer result inspector from the production Results route", () => {
  render(<ResultsView />);
  expect(screen.getByLabelText("Result producer JSON")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Inspect producer result" })).toBeTruthy();
});
