// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

import type { ResourceAdmission } from "./admission";

/** Original declaration and policy verdict rendered by the inspector. */
export interface ResourcePlanInspectorProps {
  /** Complete resource verdict, preserving its source limitations. */
  readonly admission: ResourceAdmission;
}

/** Show exact declared bytes and source limitations without numerical promotion. */
export function ResourcePlanInspector({ admission }: ResourcePlanInspectorProps) {
  return (
    <section aria-label="Resource plan" className="qsp-resource-plan">
      <h4>Declared resource plan</h4>
      <p>{admission.estimate.backend} · {admission.estimate.method}</p>
      <dl>
        {admission.estimate.components.map(component => (
          <div key={component.name}>
            <dt>{component.name} ({component.role}, {component.dtype})</dt>
            <dd>{component.bytes.toString()} bytes</dd>
          </div>
        ))}
        <div><dt>Total declared bytes</dt><dd>{admission.bytesRequired?.toString() ?? "unknown"}</dd></div>
        <div><dt>Memory ceiling</dt><dd>{admission.policy.memoryBytes?.toString() ?? "unknown"}</dd></div>
        <div><dt>Requested wall-clock ceiling (ms)</dt><dd>{admission.requestedWallMs?.toString() ?? "not requested"}</dd></div>
        <div><dt>Declared workload units / ceiling</dt><dd>{admission.estimate.workUnits?.toString() ?? "unknown"} / {admission.policy.workUnits?.toString() ?? "unknown"}</dd></div>
      </dl>
      <p className="qsp-meta">{admission.policy.source}. {admission.claimBoundary}</p>
      {!admission.allowed && <p role="alert">Resource plan refused: {admission.blockers.join(", ")}. Reduce oscillators or steps and inspect the recalculated plan.</p>}
    </section>
  );
}
