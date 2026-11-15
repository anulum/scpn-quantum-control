// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — capability services for tests that build the shell services by hand

import type { CapabilityServices, RuntimeObservation } from "../src/shared/contracts/capabilities";
import type { LocalServiceConnection } from "../src/shared/transport/localService";

const unchecked: RuntimeObservation = { status: "unknown", reason: "not checked in this test" };

/** No declared capability and no runtime checked; the check action does nothing. */
export const noCapabilityServices: CapabilityServices = {
  capabilities: [],
  checkRuntimes: () => {},
  runtimes: {
    "browser-kernel": unchecked,
    "program-ad-kernel": unchecked,
    "browser-storage": unchecked,
    "local-service": unchecked,
    hardware: unchecked,
  },
};

/** A connection that was never used; its setters and report do nothing. */
export const idleLocalService: LocalServiceConnection = {
  url: "http://127.0.0.1:8765",
  token: "",
  status: "not-connected",
  reason: "not connected in this test",
  setUrl: () => {},
  setToken: () => {},
  report: () => {},
};
