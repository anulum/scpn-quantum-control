// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

// @vitest-environment node
import { expect, it } from "vitest";
import { admitLabResources } from "./labResources";
import { browserResourcePolicy } from "./kuramotoResources";

const bounds = { maxOscillators: 128, maxSteps: 4096 };
it("declares Lab capture history and original scene coordinates independently", () => {
  const admission = admitLabResources({ n: 2, steps: 10, mode: "mean-field" }, bounds);
  expect(admission.estimate.components.find(component => component.name === "retained_phase_history")?.bytes).toBe(176n);
  // (2+1)*11 + 7*(48+1)+4 + 2+2 = 384 Vec3 numeric coordinates.
  expect(admission.estimate.components.find(component => component.role === "graph")?.bytes).toBe(9216n);
  expect(admission.estimate.workUnits).toBe(160n);
  expect(admission.policy.source).toContain("SVG string and DOM overhead excluded");
  expect(admitLabResources({ n: 2, steps: 10, mode: "mean-field" }, bounds, { ...browserResourcePolicy(bounds), memoryBytes: 0n }).allowed).toBe(false);
});
it("refuses original Lab bound violations before capture", () => {
  expect(() => admitLabResources({ n: 33, steps: 10, mode: "mean-field" }, bounds)).toThrow("Lab bounds");
  expect(() => admitLabResources({ n: 2, steps: 361, mode: "mean-field" }, bounds)).toThrow("Lab bounds");
});
