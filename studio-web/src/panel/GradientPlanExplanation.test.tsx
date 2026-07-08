// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — gradient-plan explanation render tests

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { gradientPlanExplanations } from "./data";
import type { GradientPlanExplanationView } from "./data";
import { GradientPlanExplanation } from "./GradientPlanExplanation";

function committedPlans() {
  expect(gradientPlanExplanations.ok).toBe(true);
  if (!gradientPlanExplanations.ok) {
    throw new Error(gradientPlanExplanations.reason);
  }
  return gradientPlanExplanations.value;
}

describe("GradientPlanExplanation", () => {
  it("renders the selected supported planner cell and its method", () => {
    render(<GradientPlanExplanation plans={committedPlans()} />);
    expect(screen.getByText("Gradient-plan explanation")).toBeTruthy();
    expect(screen.getByText(/10 cells/)).toBeTruthy();
    expect(screen.getByText("parameter_shift")).toBeTruthy();
    expect(screen.getByText("deterministic_local")).toBeTruthy();
    expect(screen.getByText(/backend planner selected parameter_shift with 4 evaluations/))
      .toBeTruthy();
  });

  it("switches framework and cell while keeping fail-closed boundaries explicit", () => {
    render(<GradientPlanExplanation plans={committedPlans()} />);
    fireEvent.change(screen.getByLabelText("Planner framework"), {
      target: { value: "native" },
    });
    fireEvent.change(screen.getByLabelText("Planner cell"), {
      target: { value: "ry::pauli_expectation::hardware::grad::native" },
    });
    expect(screen.getAllByText("unsupported")).toHaveLength(2);
    expect(screen.getByText("fail_closed · fail-closed boundary")).toBeTruthy();
    expect(
      screen.getAllByText("hardware gradient execution requires explicit hardware policy approval")
        .length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("statevector_simulator, finite_shot_simulator")).toBeTruthy();
  });

  it("fails closed when an empty planner artefact reaches the component", () => {
    const empty: GradientPlanExplanationView = {
      artifactId: "empty",
      claimBoundary: "empty boundary",
      methodFamilies: [],
      rows: [],
    };
    expect(() => render(<GradientPlanExplanation plans={empty} />)).toThrow(
      "gradient-plan artefact has no rows",
    );
  });
});
