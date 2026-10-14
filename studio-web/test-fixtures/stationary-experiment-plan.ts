// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — independent stationary original-kernel test source

import { readFileSync } from "node:fs";
import { instantiateKuramoto } from "../src/shared/browser/kuramoto";
import {
  createLocalExperiment,
  readLocalExperiment,
} from "../src/shared/experiments/experimentArchive";
import { prepareExperimentPlan } from "../src/shared/experiments/experimentPlan";
import type { ExperimentPlan } from "../src/shared/experiments/experimentPlan";
import { localExperimentCodecs } from "../src/shared/experiments/kuramotoArtifacts";

/**
 * Prepare an independent stationary two-oscillator oracle with the actual compiled WASM.
 *
 * @returns The original immutable source and its genuine numerical admission receipt.
 */
export async function stationaryPlan(): Promise<ExperimentPlan> {
  const path = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
  if (path === undefined) throw new Error("Actual original built WASM required");
  const kernel = await instantiateKuramoto(new Uint8Array(readFileSync(path)));
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.1, steps: 2 },
    "stationary registered-job oracle",
    localExperimentCodecs,
    "actual native worker test",
  );
  return prepareExperimentPlan(
    await readLocalExperiment(archive.json, localExperimentCodecs),
    kernel,
  );
}
