// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel render tests

import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import QuantumStudioPanel from "./QuantumStudioPanel";
import { ALL_LANES, usePanelStore } from "./panel/store";

beforeEach(() => {
  usePanelStore.setState({ laneFilter: ALL_LANES });
});

describe("QuantumStudioPanel", () => {
  it("renders the boundary banner before any evidence", () => {
    render(<QuantumStudioPanel />);
    expect(
      screen.getByText(/Committed bounded-model evidence rendered at its boundary/),
    ).toBeTruthy();
  });

  it("renders the capability manifest verbatim", () => {
    render(<QuantumStudioPanel />);
    const capabilities = screen.getByText("Capabilities").closest("section");
    expect(capabilities).not.toBeNull();
    if (capabilities !== null) {
      expect(within(capabilities).getByText("differentiate")).toBeTruthy();
      expect(
        within(capabilities).getAllByText("studio.differentiation-evidence.v1").length,
      ).toBeGreaterThanOrEqual(1);
    }
  });

  it("renders all 13 support rows with fail-closed boundaries first-class", () => {
    render(<QuantumStudioPanel />);
    expect(screen.getByText(/13 rows · 9 supported · 4 fail-closed boundaries/)).toBeTruthy();
    expect(screen.getByText("native_grad_vmap")).toBeTruthy();
    expect(screen.getByText("unsupported_nondifferentiable_boundary")).toBeTruthy();
    expect(screen.getAllByText(/blocked · fail-closed boundary/)).toHaveLength(4);
  });

  it("narrows the grid by lane without hiding the boundary counts", () => {
    render(<QuantumStudioPanel />);
    fireEvent.change(screen.getByLabelText(/Lane/), {
      target: { value: "unsupported_boundary" },
    });
    expect(screen.queryByText("native_grad_vmap")).toBeNull();
    expect(screen.getByText("unsupported_complex_valued_objective")).toBeTruthy();
    expect(screen.getByText(/13 rows · 9 supported · 4 fail-closed boundaries/)).toBeTruthy();
  });

  it("renders all 11 scorecard categories verbatim at the boundary", () => {
    render(<QuantumStudioPanel />);
    const badges = screen.getAllByText(/behind_baseline · bounded-model · boundary/);
    expect(badges).toHaveLength(11);
    expect(screen.getByText("jax_native_transforms")).toBeTruthy();
  });

  it("never renders a validated or green grade anywhere", () => {
    const { container } = render(<QuantumStudioPanel />);
    expect(container.textContent).not.toContain("reference-validated");
    expect(container.querySelector(".qsp-badge-validated")).toBeNull();
  });
});
