// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — accessible original support matrix and unchanged claim filters

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { SupportMatrixView } from "./data";
import { SupportMatrixGrid } from "./SupportMatrixGrid";

afterEach(cleanup);

/** Independent authored statuses; no science verdict is inferred by the renderer. */
function source(): SupportMatrixView {
  return { artifactId: "accessibility-independent-status-oracle", claimBoundary: "Synthetic structural evidence only; no scientific certification", rows: [
    { rowId: "analytical-row", lane: "native", caseIds: ["a"], evidence: ["analytic_reference"], transformStack: ["grad"], status: "passed", supported: true, residual: 0.002, tolerance: 0.01, blockedReasons: [], notes: ["bounded reference"] },
    { rowId: "blocked-row", lane: "unsupported_boundary", caseIds: ["b"], evidence: ["framework_parity_lane_required"], transformStack: ["vmap"], status: "blocked", supported: false, residual: null, tolerance: 0.01, blockedReasons: ["original policy boundary"], notes: [] },
    { rowId: "unknown-row", lane: "native", caseIds: ["c"], evidence: ["opaque"], transformStack: ["opaque"], status: "unrecognised", supported: false, residual: null, tolerance: 0.01, blockedReasons: [], notes: [] },
  ] };
}

it("makes the original wide table a named keyboard focus target with textual claim statuses", () => {
  const matrix = source();
  const original = JSON.stringify(matrix);
  render(<SupportMatrixGrid matrix={matrix} />);
  const scrolling = screen.getByRole("region", { name: "Support matrix table scrolling" });
  scrolling.focus();
  expect(document.activeElement).toBe(scrolling);
  const table = within(scrolling).getByRole("table");
  expect(within(table).getByText("passed · bounded-model · bounded-model · boundary")).toBeTruthy();
  expect(within(table).getByText("blocked · fail-closed · fail-closed boundary")).toBeTruthy();
  expect(within(table).getByText("unrecognised · unverifiable · unverifiable")).toBeTruthy();
  expect(within(table).getByText("2.000e-3")).toBeTruthy();
  expect(within(table).getAllByText("n/a")).toHaveLength(2);
  expect(within(table).getByText("-")).toBeTruthy();
  expect(JSON.stringify(matrix)).toBe(original);
});

it("retains original row counts and all native filters while exposing an explicit empty result", () => {
  const matrix = source();
  const original = JSON.stringify(matrix);
  render(<SupportMatrixGrid matrix={matrix} />);
  for (const [label, value] of [["Framework", "native-transform"], ["Backend", "analytic-reference"], ["Exactness", "reference-checked"], ["Claim status", "bounded-model"]] as const) {
    const select = screen.getByLabelText(label);
    fireEvent.change(select, { target: { value } });
    expect(screen.getByText("analytical-row")).toBeTruthy();
    fireEvent.change(select, { target: { value: "all" } });
  }
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: "no authored row" } });
  expect(screen.getByText("No committed support row matches these filters.")).toBeTruthy();
  expect(screen.getByText(/3 rows/)).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Operation"), { target: { value: "grad" } });
  expect(screen.getByText("analytical-row")).toBeTruthy();
  expect(JSON.stringify(matrix)).toBe(original);
});
