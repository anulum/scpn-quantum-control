// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original Results evidence guards

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, expect, it, vi } from "vitest";
const faults = vi.hoisted(() => ({ data: false, replay: false }));
vi.mock("../../shared/browser/data", async (original) => {
  const source = await original<typeof import("../../shared/browser/data")>();
  return {
    ...source,
    get supportMatrix() {
      return faults.data
        ? { ok: false, reason: "Matrix source unavailable" }
        : source.supportMatrix;
    },
    get gradientPlanExplanations() {
      return faults.data
        ? { ok: false, reason: "Gradient source unavailable" }
        : source.gradientPlanExplanations;
    },
    get scorecard() {
      return faults.data ? { ok: false, reason: "Scorecard source unavailable" } : source.scorecard;
    },
  };
});
vi.mock("../../shared/browser/programAd", async (original) => {
  const source = await original<typeof import("../../shared/browser/programAd")>();
  return {
    ...source,
    get programAdUnit() {
      return faults.replay
        ? { ok: false, reason: "Replay source unavailable" }
        : source.programAdUnit;
    },
  };
});
import { Workbench } from "../../app/Workbench";
afterEach(() => {
  cleanup();
  faults.data = false;
  faults.replay = false;
  window.history.replaceState(null, "", "/");
});

it("renders actual original evidence and preserves the viewer's input refusal", async () => {
  window.history.replaceState(null, "", "#/results/program-ad-replay");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
  // The heading can appear before the lazy view's passive focus effect has run.
  await act(async () => {
    await vi.dynamicImportSettled();
  });
  expect(screen.getByRole("heading", { name: "Analyse" })).toBeTruthy();
  expect(screen.getByText("Differentiate support explorer")).toBeTruthy();
  expect(screen.getByText("Baseline scorecard")).toBeTruthy();
  expect(document.activeElement?.id).toBe("/results/program-ad-replay");
  fireEvent.change(screen.getByLabelText("Evidence JSON"), { target: { value: "{" } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect snapshot" }));
  expect(screen.getByRole("alert").textContent).toContain("Cannot inspect evidence");
});

it("focuses the gradient debugger when its deep link opens Analyse", async () => {
  window.history.replaceState(null, "", "#/results/gradient-debugger");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
  // The heading can appear before the lazy view's passive focus effect has run.
  await act(async () => {
    await vi.dynamicImportSettled();
  });
  expect(document.activeElement?.id).toBe("/results/gradient-debugger");
  expect(screen.getByRole("region", { name: "Gradient debugger" })).toBeTruthy();
});

it.each([
  ["/results/qec-resources", "QEC and future resources"],
  ["/results/mitigation", "Mitigation and readout"],
  ["/results/pulses", "Pulse feasibility"],
])("focuses %s when its deep link opens Analyse", async (path, region) => {
  window.history.replaceState(null, "", `#${path}`);
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
  // The heading can appear before the lazy view's passive focus effect has run.
  await act(async () => {
    await vi.dynamicImportSettled();
  });
  expect(document.activeElement?.id).toBe(path);
  expect(screen.getByRole("region", { name: region })).toBeTruthy();
});

it("renders every missing source explicitly while retaining the original evidence input", async () => {
  faults.data = true;
  faults.replay = true;
  window.history.replaceState(null, "", "#/results");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
  // The gradient debugger reads the same plan source, so it reports it unverifiable too.
  expect(screen.getAllByRole("alert")).toHaveLength(5);
  const debuggerSlot = document.getElementById("/results/gradient-debugger") as HTMLElement;
  expect(debuggerSlot.textContent).toContain("Gradient source unavailable");
  for (const reason of [
    "Matrix source unavailable",
    "Gradient source unavailable",
    "Scorecard source unavailable",
    "Replay source unavailable",
  ])
    expect(screen.getAllByRole("alert").some((alert) => alert.textContent?.includes(reason))).toBe(
      true,
    );
  expect(screen.getByLabelText("Evidence JSON")).toBeTruthy();
  expect(screen.queryByText("Baseline scorecard")).toBeNull();
});

it("reaches the original producer result inspector from the production Results route", async () => {
  window.history.replaceState(null, "", "#/results");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
  expect(screen.getByLabelText("Result producer JSON")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Inspect producer result" })).toBeTruthy();
  // Comparison has its own stage in the research loop; Analyse no longer carries it.
  expect(
    screen.queryByRole("heading", { name: "Compare immutable revisions and runs" }),
  ).toBeNull();
  cleanup();
  window.history.replaceState(null, "", "#/compare");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Compare" }, { timeout: 5000 });
  expect(
    await screen.findByRole("heading", { name: "Compare immutable revisions and runs" }),
  ).toBeTruthy();
  expect(screen.getByLabelText("Baseline archive JSON")).toBeTruthy();
});

it.each([false, true])(
  "retains replay focus when Results is remounted with StrictMode=%s",
  async (strict) => {
    window.history.replaceState(null, "", "#/results/program-ad-replay");
    const first = render(<Workbench>{(_context, editors) => editors}</Workbench>);
    await screen.findByRole("heading", { name: "Analyse" }, { timeout: 5000 });
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    expect(document.activeElement?.id).toBe("/results/program-ad-replay");
    first.unmount();

    await act(async () => {
      const workbench = <Workbench>{(_context, editors) => editors}</Workbench>;
      render(strict ? <StrictMode>{workbench}</StrictMode> : workbench);
      await vi.dynamicImportSettled();
    });
    expect(document.activeElement?.id).toBe("/results/program-ad-replay");
    const input = screen.getByLabelText("Evidence JSON");
    input.focus();
    fireEvent.change(input, { target: { value: "{" } });
    expect(document.activeElement).toBe(input);
  },
);
