// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native original panel input refusal

import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import { QuantumStudioPanel } from "../src/QuantumStudioPanel";
import { conformanceCodecs } from "./workspaceFixture";

/** Render original parser refusals from damaged input files in an isolated source tree. */
export async function runNativePanelRefusalCases(): Promise<Record<string, unknown>> {
  const container = document.createElement("main");
  document.body.append(container);
  const root = createRoot(container);
  const surfaces = [
    "studio_manifest.json", "capability catalogue", "xy_compile_recompute_unit_20260708.json",
    "kuramoto_scenario_meanfield_20260708.json", "program_ad_replay_rational_20260714.json",
    "differentiable_transform_support_matrix_20260708.json", "gradient_plan_explanations_20260709.json",
    "differentiable_baseline_scorecard_20260620.json",
  ];
  try {
    flushSync(() => root.render(<QuantumStudioPanel rawCodecs={conformanceCodecs} />));
    const alerts = [...container.querySelectorAll(".qsp-unverifiable[role=alert]")];
    const observed = alerts.map(alert => ({ surface: alert.querySelector("code")?.textContent, refusal: alert.textContent }));
    if (alerts.length !== 9 || surfaces.some(surface => !observed.some(alert => alert.surface === surface && alert.refusal?.includes("failed its fail-closed guard")))) throw new Error("Original panel omitted a damaged-source refusal");
    if (observed.filter(alert => alert.surface === "kuramoto_scenario_meanfield_20260708.json").length !== 2) throw new Error("Damaged scenario did not refuse both original simulation views");
    if (container.querySelector(".qsp-workspace") === null) throw new Error("Source refusal removed the local workspace");
    if (container.querySelector(".qsp-play-chart, .qsp-lab-figures") !== null) throw new Error("Damaged source produced a numerical view");
    return { observed, localWorkspaceRetained: true, numericalViews: 0, boundary: "Original parsers and panel over isolated damaged source files; no solver or hardware evidence" };
  } finally {
    flushSync(() => root.unmount());
    container.remove();
  }
}
