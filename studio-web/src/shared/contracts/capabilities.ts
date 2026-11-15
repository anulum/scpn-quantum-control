// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — resolved capability states the shell hands to modules

import type { RuntimeNeed } from "./module";

/** What the workbench shows for a capability; none is ever hidden. */
export type CapabilityState = "stable" | "experimental" | "degraded" | "unavailable";

/** What the shell has observed about one runtime. */
export interface RuntimeObservation {
  /** `unknown` until the runtime has been checked. */
  readonly status: "available" | "degraded" | "unavailable" | "unknown";
  /** Why the runtime is not fully available, or what was verified. */
  readonly reason: string;
}

/** One declared capability with its resolved state. */
export interface ResolvedCapability {
  /** Capability identifier. */ readonly id: string;
  /** What the user can do. */ readonly label: string;
  /** Module that declared it. */ readonly module: string;
  /** Resolved state. */ readonly state: CapabilityState;
  /** Only the unmet needs, each with its reason; empty when nothing is missing. */ readonly reasons: readonly string[];
}

/** The capability report and the on-demand runtime check, handed to a stage. */
export interface CapabilityServices {
  /** Every declared capability, resolved. */
  readonly capabilities: readonly ResolvedCapability[];
  /** Load and verify the WASM runtimes now; the report checks nothing before this is called. */
  readonly checkRuntimes: () => void;
  /** Observations per runtime, for display. */
  readonly runtimes: Readonly<Record<RuntimeNeed, RuntimeObservation>>;
}
