// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — bounded checks that the shipped WASM runtimes load and verify

import { fetchProgramAd, programAdUnit, verifyProgramAdUnit } from "./programAd";
import { fetchKernel, recomputeUnit, verifyRecomputeUnit } from "./recompute";

/**
 * Load the browser kernel and verify its committed XY recompute input.
 *
 * @throws Error When the input failed its guard, the kernel did not load or the result differs.
 */
export async function probeBrowserKernel(): Promise<void> {
  if (!recomputeUnit.ok) throw new Error(recomputeUnit.reason);
  const verdict = verifyRecomputeUnit(recomputeUnit.value, await fetchKernel());
  if (verdict.display !== "match") throw new Error("Committed XY input did not verify");
}

/**
 * Load the program-AD runtime and verify its committed replay input.
 *
 * @throws Error When the input failed its guard, the runtime did not load or the replay differs.
 */
export async function probeProgramAdKernel(): Promise<void> {
  if (!programAdUnit.ok) throw new Error(programAdUnit.reason);
  const verdict = await verifyProgramAdUnit(programAdUnit.value, await fetchProgramAd());
  if (verdict.display !== "match") throw new Error("Committed program-AD input did not verify");
}
