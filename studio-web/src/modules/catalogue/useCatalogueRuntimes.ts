// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — useCatalogueRuntimes

import { useEffect, useState } from "react";
import { probeBrowserKernel, probeProgramAdKernel } from "../../shared/browser/runtimeProbes";
import type { RuntimeAvailability } from "./catalogue";

const loaders: Readonly<Record<string, () => Promise<unknown>>> = {
  compile: probeBrowserKernel,
  differentiate: probeProgramAdKernel,
};
/** Load and bind optional WASM runtimes; late resolutions cannot update a disposed panel. */
export function useCatalogueRuntimes(
  probes: Readonly<Record<string, () => Promise<unknown>>> = loaders,
): Readonly<Record<string, RuntimeAvailability>> {
  const [result, setResult] = useState<{
    probes: typeof probes;
    values: Record<string, RuntimeAvailability>;
  }>({ probes, values: {} });
  useEffect(() => {
    let live = true;
    for (const [verb, probe] of Object.entries(probes)) {
      const publish = (availability: RuntimeAvailability) => {
        if (live) {
          setResult((previous) => ({
            probes,
            values: {
              ...(previous.probes === probes ? previous.values : {}),
              [verb]: availability,
            },
          }));
        }
      };
      Promise.resolve()
        .then(probe)
        .then(
          () => publish({ available: true, reason: "Bounded WASM input verified" }),
          (error: unknown) =>
            publish({
              available: false,
              reason: error instanceof Error ? error.message : "Kernel load failed",
            }),
        );
    }
    return () => {
      live = false;
    };
  }, [probes]);
  return result.probes === probes ? result.values : {};
}
