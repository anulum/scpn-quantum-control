// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — guarded Build projection

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
const faults = vi.hoisted(() => ({ compile: false, scenario: false }));
vi.mock("../../shared/browser/recompute", async (original) => {
  const source = await original<typeof import("../../shared/browser/recompute")>();
  return {
    ...source,
    get recomputeUnit() {
      return faults.compile
        ? { ok: false, reason: "Compile source unavailable" }
        : source.recomputeUnit;
    },
  };
});
vi.mock("../../shared/browser/kuramoto", async (original) => {
  const source = await original<typeof import("../../shared/browser/kuramoto")>();
  return {
    ...source,
    get committedScenario() {
      return faults.scenario
        ? { ok: false, reason: "Scenario source unavailable" }
        : source.committedScenario;
    },
  };
});
import { Workbench } from "../../app/Workbench";
afterEach(() => {
  cleanup();
  faults.compile = false;
  faults.scenario = false;
  window.history.replaceState(null, "", "/");
});

it("renders the actual original instrument owners with their committed inputs", async () => {
  window.history.replaceState(null, "", "#/build/compile-recompute");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Model" });
  expect(screen.getByRole("heading", { name: "Model" })).toBeTruthy();
  // The lazy program editor loads inside its own contribution boundary, after the stage.
  expect(await screen.findByRole("heading", { name: "Program editor" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Compile source" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Compiler trace inspector" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Recompute in browser" })).toBeTruthy();
  expect(screen.getByText(/A requested workspace revision is not substituted/)).toBeTruthy();
  // jsdom serves no WASM kernel, so only the kernel fetch may be reported; no instrument
  // owner, contribution or route is refused.
  expect(
    screen
      .queryAllByRole("alert")
      .every((alert) => alert.textContent === "unverifiable — fetch failed"),
  ).toBe(true);
  expect(document.activeElement?.id).toBe("/build/compile-recompute");
});

it("opens original native compiler metadata through the Build route beside source authoring", async () => {
  window.history.replaceState(null, "", "#/build");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Model" });
  fireEvent.click(screen.getByRole("button", { name: "Open native example" }));
  await screen.findByRole("table", { name: "Qubit mapping" });
  expect(screen.getByRole("region", { name: "Program authoring" })).toBeTruthy();
  expect(screen.getByLabelText("Mapped readout").textContent).toContain(
    "q[0] / c[1] → q[1] / c[1]",
  );
  expect(
    screen.getByRole("button", { name: "Export admitted trace" }).hasAttribute("disabled"),
  ).toBe(false);
});

it("refuses a missing compile source without removing available simulation instruments", async () => {
  faults.compile = true;
  window.history.replaceState(null, "", "#/build");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Model" });
  expect(screen.getByRole("alert").textContent).toContain("Compile source unavailable");
  expect(screen.queryByRole("button", { name: "Recompute in browser" })).toBeNull();
  expect(screen.getByRole("heading", { name: "Kuramoto Play" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "3D Lab" })).toBeTruthy();
});

it("refuses a missing scenario for both original simulation owners", async () => {
  faults.scenario = true;
  window.history.replaceState(null, "", "#/build");
  render(<Workbench>{(_context, editors) => editors}</Workbench>);
  await screen.findByRole("heading", { name: "Model" });
  expect(screen.getByRole("alert").textContent).toContain("Scenario source unavailable");
  expect(screen.getByRole("button", { name: "Recompute in browser" })).toBeTruthy();
  expect(screen.queryByText("Kuramoto Play")).toBeNull();
});
