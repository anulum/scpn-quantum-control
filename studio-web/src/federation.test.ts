// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — federation contract regression guards

import { describe, expect, it } from "vitest";

import {
  FEDERATION_NAME,
  PANEL_EXPOSE_KEY,
  REACT_VERSION,
  moduleFederationConfig,
} from "../module-federation.config";
import packageJson from "../package.json";

describe("module federation contract", () => {
  it("keeps the locked federation name and expose key", () => {
    expect(FEDERATION_NAME).toBe("scpn_quantum_control");
    expect(moduleFederationConfig.name).toBe(FEDERATION_NAME);
    expect(moduleFederationConfig.filename).toBe("remoteEntry.js");
    expect(Object.keys(moduleFederationConfig.exposes)).toEqual([PANEL_EXPOSE_KEY]);
    expect(PANEL_EXPOSE_KEY).toBe("./QuantumStudioPanel");
  });

  it("shares react and react-dom as version-pinned singletons", () => {
    expect(moduleFederationConfig.shared.react).toEqual({
      singleton: true,
      requiredVersion: REACT_VERSION,
    });
    expect(moduleFederationConfig.shared["react-dom"]).toEqual({
      singleton: true,
      requiredVersion: REACT_VERSION,
    });
  });

  it("pins the installed react to the shared singleton version", () => {
    expect(REACT_VERSION).toBe("19.2.7");
    expect(packageJson.dependencies.react).toBe(REACT_VERSION);
    expect(packageJson.dependencies["react-dom"]).toBe(REACT_VERSION);
  });
});
