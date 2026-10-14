// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original stationary source with a saved local workflow

import { instantiateKuramoto } from "../src/shared/browser/kuramoto";
import { readLocalExperiment } from "../src/shared/experiments/experimentArchive";
import { prepareExperimentPlan } from "../src/shared/experiments/experimentPlan";
import type { ExperimentPlan } from "../src/shared/experiments/experimentPlan";
import { localExperimentCodecs } from "../src/shared/experiments/kuramotoArtifacts";
import { archiveWorkflow } from "../src/shared/workflows/workflowArchive";
import { parseWorkflow } from "../src/shared/workflows/workflowModel";
import { stationaryPlan } from "./stationary-experiment-plan";

/** Save a one-cell original workflow and bind the real plan to its complete archive. */
export async function stationaryWorkflowPlan(): Promise<ExperimentPlan> {
  const baseline = await stationaryPlan();
  const kernel = await instantiateKuramoto(baseline.wasmBytes);
  const definition = parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "registered-stationary-workflow",
      stages: [
        {
          id: "simulate",
          adapter: "local-kuramoto",
          verb: "simulate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [],
          depends_on: [],
        },
      ],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
    },
    extensions: {},
  });
  const archive = await archiveWorkflow(
    baseline.sourceJson,
    baseline.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  return prepareExperimentPlan(
    await readLocalExperiment(archive.json, localExperimentCodecs),
    kernel,
  );
}
