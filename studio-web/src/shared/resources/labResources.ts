// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

import type { KuramotoBounds } from "../../panel/kuramoto";
import { GUIDE_SEGMENTS } from "../../panel/labGeometry";
import { LAB_MAX_OSCILLATORS, LAB_MAX_STEPS } from "../../panel/phaseTrajectory";
import { checkResourcePlan } from "./admission";
import type { ResourceAdmission, ResourceBuffer, ResourcePolicy } from "./admission";
import { admitKuramotoResources, browserResourcePolicy } from "./kuramotoResources";
import type { KuramotoResourceRequest } from "./kuramotoResources";

/** Declare retained trajectory and scene coordinates before capture or geometry construction. */
export function admitLabResources(
  request: KuramotoResourceRequest, bounds: KuramotoBounds,
  supplied: ResourcePolicy = browserResourcePolicy(bounds),
): ResourceAdmission {
  if (request.n > LAB_MAX_OSCILLATORS || request.steps > LAB_MAX_STEPS) throw new Error("request exceeds declared Lab bounds");
  const base = admitKuramotoResources(request, bounds, supplied);
  const n = BigInt(request.n);
  const samples = BigInt(request.steps) + 1n;
  const guides = 7n * BigInt(GUIDE_SEGMENTS + 1) + 4n;
  const vertices = (n + 1n) * samples + guides + n + 2n;
  const buffers: ResourceBuffer[] = [
    ...base.estimate.components.map(({ name, role, shape, dtype, count }) => ({ name, role, shape, dtype, count })),
    { name: "retained_phase_history", role: "intermediate", shape: [samples, n], dtype: "float64", count: 1n },
    { name: "retained_series_and_scene_series", role: "intermediate", shape: [samples], dtype: "float64", count: 5n },
    { name: "current_phase_copy", role: "transfer", shape: [n], dtype: "float64", count: 1n },
    { name: "scene_numeric_coordinates", role: "graph", shape: [vertices, 3n], dtype: "float64", count: 1n },
  ];
  return checkResourcePlan({
    backend: base.estimate.backend, method: `verified-trajectory-${request.mode}`,
    buffers, concurrency: 1n, workUnits: base.estimate.workUnits! * 2n,
  }, { ...supplied, source: `${supplied.source}; scene object, SVG string and DOM overhead excluded` });
}
