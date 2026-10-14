// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered original experiment worker job

import type { ModuleJob } from "../../shared/contracts/module";
import type { WorkbenchServices } from "../../shared/contracts/workbench";

/** Existing prepared numerical plan and original worker lifecycle, without self-issued admission. */
export const experimentJob: ModuleJob<WorkbenchServices> = {
  id: "local-experiment",
  label: "Run prepared local experiment",
  runner: "worker",
  stage: "experiments",
  estimate: async (services, document) => {
    const plan = services.experiment.plan;
    if (plan === null || plan.sourceJson !== document.content)
      throw new Error("Prepare the current numerical plan in Experiments first");
    const original = plan.admission.estimate;
    return {
      kind: "declared-resources",
      plan: {
        backend: original.backend,
        method: original.method,
        buffers: original.components.map(({ name, role, shape, dtype, count }) => ({
          name,
          role,
          shape,
          dtype,
          count,
        })),
        concurrency: original.concurrency,
        workUnits: original.workUnits,
      },
    };
  },
  run: async (services, document, signal) => {
    if (signal.aborted || services.experiment.plan?.sourceJson !== document.content)
      throw new Error("Experiment source changed or execution cancelled");
    const cancel = () => {
      void services.experiment.cancel();
    };
    signal.addEventListener("abort", cancel, { once: true });
    try {
      await services.experiment.run();
    } finally {
      signal.removeEventListener("abort", cancel);
    }
  },
};
