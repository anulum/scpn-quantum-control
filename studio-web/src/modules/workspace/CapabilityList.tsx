// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — what this installation can do, every capability shown

import type { CapabilityServices, CapabilityState } from "../../shared/contracts/capabilities";

const tone: Readonly<Record<CapabilityState, string>> = {
  stable: "qsp-status qsp-status-ok",
  experimental: "qsp-status qsp-status-ok",
  degraded: "qsp-status",
  unavailable: "qsp-status qsp-status-refused",
};

/**
 * Every capability the modules declare, with its state and only its unmet needs.
 *
 * The state is written out, so it never depends on colour. The kernels are
 * checked for this list only when the user asks.
 *
 * @param props - Component inputs.
 * @param props.services - Resolved capabilities and the on-demand check.
 */
export function CapabilityList({ services }: { readonly services: CapabilityServices }) {
  return (
    <section className="qsp-panel" aria-label="What this installation can do">
      <h3>What this installation can do</h3>
      <p>Every capability is listed, also the ones that cannot run here, with what is missing.</p>
      <button type="button" onClick={services.checkRuntimes}>
        Check runtimes
      </button>
      <ul>
        {services.capabilities.map((capability) => (
          <li key={capability.id} data-capability={capability.id} data-state={capability.state}>
            <span className={tone[capability.state]}>{capability.state}</span> {capability.label}
            {capability.reasons.length === 0 ? null : (
              <ul aria-label={`What ${capability.label} is missing`}>
                {capability.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
