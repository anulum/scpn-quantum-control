// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — evidence catalogue refuses each failed surface in place

import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

vi.mock("../../shared/browser/data", () => ({
  studioManifest: { ok: false, reason: "forced manifest guard failure" },
  supportMatrix: { ok: false, reason: "forced matrix guard failure" },
  gradientPlanExplanations: { ok: false, reason: "forced gradient-plan guard failure" },
  scorecard: { ok: false, reason: "forced scorecard guard failure" },
}));

import EvidenceCatalogueView from "./EvidenceCatalogueView";

it("renders one loud unverifiable block per failed evidence surface", () => {
  render(<EvidenceCatalogueView />);
  expect(screen.getByRole("heading", { name: "Evidence catalogue" })).toBeTruthy();
  expect(screen.getAllByRole("alert")).toHaveLength(4);
  for (const reason of ["manifest", "matrix", "gradient-plan", "scorecard"])
    expect(screen.getByText(new RegExp(`forced ${reason} guard failure`))).toBeTruthy();
});
