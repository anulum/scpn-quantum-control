// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel render tests

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import QuantumStudioPanel from "./QuantumStudioPanel";

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
    expect(screen.getByText("frameworks 5")).toBeTruthy();
    expect(screen.getByText("exactness levels 3")).toBeTruthy();
    expect(screen.getByText("native_grad_vmap")).toBeTruthy();
    expect(screen.getByText("unsupported_nondifferentiable_boundary")).toBeTruthy();
    expect(screen.getAllByText(/blocked · fail-closed · fail-closed boundary/)).toHaveLength(4);
  });

  it("narrows the cockpit by framework without hiding the boundary counts", () => {
    render(<QuantumStudioPanel />);
    fireEvent.change(screen.getByLabelText("Framework"), {
      target: { value: "unsupported-boundary" },
    });
    expect(screen.queryByText("native_grad_vmap")).toBeNull();
    expect(screen.getByText("unsupported_complex_valued_objective")).toBeTruthy();
    expect(screen.getByText(/13 rows · 9 supported · 4 fail-closed boundaries/)).toBeTruthy();
  });

  it("narrows the cockpit by backend, exactness, claim status, and operation search", () => {
    render(<QuantumStudioPanel />);
    fireEvent.change(screen.getByLabelText("Backend"), {
      target: { value: "parameter-shift-reference" },
    });
    fireEvent.change(screen.getByLabelText("Exactness"), {
      target: { value: "reference-checked" },
    });
    fireEvent.change(screen.getByLabelText("Claim status"), {
      target: { value: "bounded-model" },
    });
    fireEvent.change(screen.getByLabelText("Operation"), {
      target: { value: "phase_qnode_native_vmap_grad_matches_parameter_shift_reference" },
    });
    expect(screen.getByText("quantum_gradient_native_nesting")).toBeTruthy();
    expect(screen.queryByText("native_grad_vmap")).toBeNull();
  });

  it("renders an explicit empty state when filters match no committed row", () => {
    render(<QuantumStudioPanel />);
    fireEvent.change(screen.getByLabelText("Operation"), {
      target: { value: "not-a-committed-support-row" },
    });
    expect(screen.getByText("No committed support row matches these filters.")).toBeTruthy();
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
