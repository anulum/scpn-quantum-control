// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel fail-closed render tests

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./panel/data", () => ({
  studioManifest: { ok: false, reason: "forced manifest guard failure" },
  supportMatrix: { ok: false, reason: "forced matrix guard failure" },
  gradientPlanExplanations: {
    ok: false,
    reason: "forced gradient-plan guard failure",
  },
  scorecard: { ok: false, reason: "forced scorecard guard failure" },
}));

vi.mock("./panel/recompute", () => ({
  recomputeUnit: { ok: false, reason: "forced recompute guard failure" },
}));

import QuantumStudioPanel from "./QuantumStudioPanel";

describe("QuantumStudioPanel with failing guards", () => {
  it("renders one loud unverifiable block per failed surface", () => {
    render(<QuantumStudioPanel />);
    const alerts = screen.getAllByRole("alert");
    expect(alerts).toHaveLength(5);
    expect(screen.getByText(/forced manifest guard failure/)).toBeTruthy();
    expect(screen.getByText(/forced recompute guard failure/)).toBeTruthy();
    expect(screen.getByText(/forced matrix guard failure/)).toBeTruthy();
    expect(screen.getByText(/forced gradient-plan guard failure/)).toBeTruthy();
    expect(screen.getByText(/forced scorecard guard failure/)).toBeTruthy();
  });
});
