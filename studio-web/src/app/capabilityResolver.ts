// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — resolve declared capabilities against runtimes, manifest and host policy

import type { ResolvedCapability, RuntimeObservation } from "../shared/contracts/capabilities";
import type { MissingRequirement, ModuleCapability, RuntimeNeed } from "../shared/contracts/module";

/** A capability together with the module that declared it. */
export interface OwnedCapability {
  /** The declaration. */ readonly capability: ModuleCapability;
  /** Declaring module. */ readonly module: string;
}

/**
 * Resolve every declared capability; none is dropped.
 *
 * A manifest requirement the installation lacks, a need the host refuses or an
 * unavailable need makes the capability unavailable; a degraded or unchecked
 * need makes it degraded; otherwise it keeps its declared maturity. Only the
 * unmet needs are named.
 *
 * @param owned - Declared capabilities with their modules, in registry order.
 * @param runtimes - What the shell has observed per runtime.
 * @param missing - Manifest requirements the installation does not declare.
 * @param refused - Runtimes the host refuses, with the reason.
 * @returns One resolved row per declared capability, in the same order.
 */
export function resolveCapabilities(
  owned: readonly OwnedCapability[],
  runtimes: Readonly<Record<RuntimeNeed, RuntimeObservation>>,
  missing: readonly MissingRequirement[],
  refused: ReadonlyMap<RuntimeNeed, string>,
): readonly ResolvedCapability[] {
  return owned.map(({ capability, module }) => {
    const lacking = missing
      .filter((requirement) => requirement.module === module)
      .map(
        (requirement) =>
          `the installation does not declare the ${requirement.kind} ${requirement.name}`,
      );
    const policy = capability.needs.flatMap((need) => {
      const reason = refused.get(need);
      return reason === undefined ? [] : [`${need}: ${reason}`];
    });
    const observed = (status: RuntimeObservation["status"]) =>
      capability.needs
        .filter((need) => !refused.has(need) && runtimes[need].status === status)
        .map((need) => `${need}: ${runtimes[need].reason}`);
    const blocking = [...lacking, ...policy, ...observed("unavailable")];
    const weak = [...observed("degraded"), ...observed("unknown")];
    const base = { id: capability.id, label: capability.label, module };
    if (blocking.length > 0) return { ...base, state: "unavailable", reasons: blocking };
    if (weak.length > 0) return { ...base, state: "degraded", reasons: weak };
    return { ...base, state: capability.maturity, reasons: [] };
  });
}
