// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web Module Federation contract

/**
 * The federation contract of the QUANTUM studio remote.
 *
 * Locked by the platform v1 contract: the federation name is
 * `scpn_quantum_control` (underscored form of the studio id), the remote
 * exposes exactly `./QuantumStudioPanel`, and react/react-dom are shared as
 * version-pinned singletons so the Hub never double-mounts a second React.
 * Keep this object additive-only; renaming any field is a breaking change.
 */

export const FEDERATION_NAME = "scpn_quantum_control";
export const PANEL_EXPOSE_KEY = "./QuantumStudioPanel";
export const REACT_VERSION = "19.2.7";

export const moduleFederationConfig = {
  name: FEDERATION_NAME,
  filename: "remoteEntry.js",
  exposes: {
    [PANEL_EXPOSE_KEY]: "./src/QuantumStudioPanel.tsx",
  },
  shared: {
    react: { singleton: true, requiredVersion: REACT_VERSION },
    "react-dom": { singleton: true, requiredVersion: REACT_VERSION },
  },
} as const;
