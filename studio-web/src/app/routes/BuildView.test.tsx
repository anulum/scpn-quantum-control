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
vi.mock("../../panel/recompute", async original => {
  const source = await original<typeof import("../../panel/recompute")>();
  return { ...source, get recomputeUnit() { return faults.compile ? { ok: false, reason: "Compile source unavailable" } : source.recomputeUnit; } };
});
vi.mock("../../panel/kuramoto", async original => {
  const source = await original<typeof import("../../panel/kuramoto")>();
  return { ...source, get committedScenario() { return faults.scenario ? { ok: false, reason: "Scenario source unavailable" } : source.committedScenario; } };
});
import BuildView from "./BuildView";
afterEach(() => { cleanup(); faults.compile = false; faults.scenario = false; });

it("renders the actual original instrument owners with their committed inputs", () => {
  render(<BuildView focusInstrument />);
  expect(screen.getByRole("heading", { name: "Build" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Program editor" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Compile source" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Compiler trace inspector" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Recompute in browser" })).toBeTruthy();
  expect(screen.getByText(/A requested workspace revision is not substituted/)).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
  expect(document.activeElement?.id).toBe("/build/compile-recompute");
});

it("opens original native compiler metadata through the Build route beside source authoring", async () => {
  render(<BuildView />);
  fireEvent.click(screen.getByRole("button", { name: "Open native example" }));
  await screen.findByRole("table", { name: "Qubit mapping" });
  expect(screen.getByRole("region", { name: "Program authoring" })).toBeTruthy();
  expect(screen.getByLabelText("Mapped readout").textContent).toContain("q[0] / c[1] → q[1] / c[1]");
  expect(screen.getByRole("button", { name: "Export admitted trace" }).hasAttribute("disabled")).toBe(false);
});

it("refuses a missing compile source without removing available simulation instruments", () => {
  faults.compile = true;
  render(<BuildView />);
  expect(screen.getByRole("alert").textContent).toContain("Compile source unavailable");
  expect(screen.queryByRole("button", { name: "Recompute in browser" })).toBeNull();
  expect(screen.getByRole("heading", { name: "Kuramoto Play" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "3D Lab" })).toBeTruthy();
});

it("refuses a missing scenario for both original simulation owners", () => {
  faults.scenario = true;
  render(<BuildView />);
  expect(screen.getByRole("alert").textContent).toContain("Scenario source unavailable");
  expect(screen.getByRole("button", { name: "Recompute in browser" })).toBeTruthy();
  expect(screen.queryByText("Kuramoto Play")).toBeNull();
});
