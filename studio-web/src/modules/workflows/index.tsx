// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import { workflowJob } from "./moduleJob";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
const WorkflowRunner = lazy(() =>
  import("./WorkflowRunner").then((module) => ({ default: module.WorkflowRunner })),
);
/** Original workflow traversal and worker settlement contributed to Experiments. */
export const workflowsModule: StudioModule<WorkbenchServices, WorkspaceViewServices, never> = {
  id: "workflows",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "stage",
      id: "workflow-graph",
      label: "Workflow graph",
      stage: "experiments",
      render: (services: WorkbenchServices) => (
        <WorkflowRunner
          workspace={services.workspace}
          rawCodecs={services.codecs}
          run={services.workflow}
          experimentBlocked={
            services.experiment.status === "running" ||
            services.experiment.status === "cancelling" ||
            services.experiment.disposalBlocked
          }
        />
      ),
    },
  ],
  jobs: [workflowJob],
  commands: [
    {
      id: "run-workflow",
      label: "Run selected local workflow",
      action: { kind: "job", job: "local-workflow" },
    },
  ],
};
