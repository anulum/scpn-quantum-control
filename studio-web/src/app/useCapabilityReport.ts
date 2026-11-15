// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — observe runtimes on demand and resolve every declared capability

import { useCallback, useEffect, useRef, useState } from "react";
import { probeBrowserKernel, probeProgramAdKernel } from "../shared/browser/runtimeProbes";
import type { CapabilityServices, RuntimeObservation } from "../shared/contracts/capabilities";
import type { MissingRequirement, ModuleRegistry, RuntimeNeed } from "../shared/contracts/module";
import type { WorkspacePersistence } from "../shared/storage/workspacePersistence";
import { resolveCapabilities } from "./capabilityResolver";

/** The two runtimes checked by loading and verifying their committed inputs. */
export type KernelNeed = "browser-kernel" | "program-ad-kernel";
/** One bounded check per kernel; it rejects when the kernel cannot be used. */
export type KernelProbes = Readonly<Record<KernelNeed, () => Promise<void>>>;

const shippedProbes: KernelProbes = {
  "browser-kernel": probeBrowserKernel,
  "program-ad-kernel": probeProgramAdKernel,
};

/** Runtimes this host refuses, whatever a module declares. */
export const hostRefusals: ReadonlyMap<RuntimeNeed, string> = new Map([
  [
    "hardware",
    "remote execution is a separate gated service integration, not part of this installation",
  ],
]);

const unchecked: RuntimeObservation = {
  status: "unknown",
  reason: "not checked yet; use Check runtimes",
};

function storage(persistence: WorkspacePersistence | null): RuntimeObservation {
  if (persistence === null)
    return { status: "unknown", reason: "browser storage is still opening" };
  if (persistence === "browser") return { status: "available", reason: "browser cache" };
  return { status: "degraded", reason: "this page session only; export your work to keep it" };
}

function useKernelObservations(probes: KernelProbes) {
  const [kernels, setKernels] = useState<Record<KernelNeed, RuntimeObservation>>({
    "browser-kernel": unchecked,
    "program-ad-kernel": unchecked,
  });
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const check = useCallback(() => {
    for (const need of Object.keys(probes) as KernelNeed[]) {
      const publish = (observation: RuntimeObservation) => {
        if (alive.current) setKernels((previous) => ({ ...previous, [need]: observation }));
      };
      publish({ status: "unknown", reason: "checking" });
      probes[need]().then(
        () => publish({ status: "available", reason: "committed input verified" }),
        (cause: unknown) => publish({ status: "unavailable", reason: (cause as Error).message }),
      );
    }
  }, [probes]);
  return { kernels, check };
}

/**
 * Resolve every capability the registered modules declare.
 *
 * The report checks the kernels only when `checkRuntimes` is called; it does
 * not borrow what other panels observed. The local service is optional and is
 * never contacted here.
 *
 * @param registry - The compiled module registry.
 * @param missing - Manifest requirements the installation does not declare.
 * @param persistence - Where saved work lives, or null while storage opens.
 * @param probes - Kernel checks; tests pass their own.
 * @returns The resolved capabilities, the runtime observations and the check action.
 */
export function useCapabilityReport<Services, DocumentServices>(
  registry: ModuleRegistry<Services, DocumentServices>,
  missing: readonly MissingRequirement[],
  persistence: WorkspacePersistence | null,
  probes: KernelProbes = shippedProbes,
): CapabilityServices {
  const { kernels, check } = useKernelObservations(probes);
  const runtimes: Record<RuntimeNeed, RuntimeObservation> = {
    ...kernels,
    "browser-storage": storage(persistence),
    "local-service": {
      status: "unavailable",
      reason: "optional; start the local job service and connect from Run → Local job service",
    },
    hardware: { status: "unavailable", reason: "not part of this installation" },
  };
  const owned = registry.capabilities.map((capability) => ({
    capability,
    module: registry.contributionOwner("capabilities", capability.id)?.id ?? "unknown",
  }));
  return {
    capabilities: resolveCapabilities(owned, runtimes, missing, hostRefusals),
    checkRuntimes: check,
    runtimes,
  };
}
