// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import { experimentJob } from "./moduleJob";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
const ExperimentRunner = lazy(() => import("./ExperimentRunner"));
/** Original experiment runner shares the work area with independently registered workflow controls. */
export const experimentsModule: StudioModule<
  WorkbenchServices,
  WorkspaceViewServices,
  "experiments"
> = {
  id: "experiments",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "experiments" as const,
      label: "Experiments",
      group: "primary",
      instruments: [],
      render: (services: WorkbenchServices) => (
        <>
          <ExperimentRunner
            workspace={services.workspace}
            rawCodecs={services.codecs}
            run={services.experiment}
            workspaceHref={services.workspaceHref}
          />
          {services.renderEditors("experiments")}
        </>
      ),
    },
  ],
  commands: [
    {
      id: "open-experiments",
      label: "Open Experiments",
      action: { kind: "navigate", stage: "experiments", instrument: null },
    },
    {
      id: "run-experiment",
      label: "Run prepared local experiment",
      action: { kind: "job", job: "local-experiment" },
    },
  ],
  jobs: [experimentJob],
};
