// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — test-only module that cannot issue itself job permission

import type { ModuleJob, StudioModule } from "../../src/shared/contracts/module";
import type {
  WorkbenchServices,
  WorkspaceViewServices,
} from "../../src/shared/contracts/workbench";

/** Deliberately carry an unsupported permission flag beside a genuine original plan. */
export function policyProbeModule(
  onInvocation: () => void,
): StudioModule<WorkbenchServices, WorkspaceViewServices> {
  const job: ModuleJob<WorkbenchServices> = Object.assign(
    {
      id: "policy-probe",
      label: "Run policy probe",
      runner: "worker" as const,
      stage: "experiments",
      estimate: async (services: WorkbenchServices) => {
        const plan = services.experiment.plan;
        if (plan === null) throw new Error("Original numerical plan missing");
        const original = plan.admission.estimate;
        return {
          kind: "declared-resources" as const,
          plan: {
            backend: original.backend,
            method: original.method,
            concurrency: original.concurrency,
            workUnits: original.workUnits,
            buffers: original.components.map(({ name, role, shape, dtype, count }) => ({
              name,
              role,
              shape,
              dtype,
              count,
            })),
          },
        };
      },
      run: async (services: WorkbenchServices) => {
        onInvocation();
        await services.experiment.run();
      },
    },
    { allowed: true },
  );
  return {
    id: "policy-probe",
    version: "1",
    requires: { verbs: [], schemas: [] },
    stages: [],
    jobs: [job],
  };
}
