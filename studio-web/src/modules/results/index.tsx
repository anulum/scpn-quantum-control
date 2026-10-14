// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
const ResultsView = lazy(() => import("./ResultsView"));
/** Original results and committed replay retain their source and lifecycle bindings. */
export const resultsModule: StudioModule<WorkbenchServices, WorkspaceViewServices, "results"> = {
  id: "results",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "results" as const,
      label: "Results",
      group: "primary",
      instruments: ["program-ad-replay"],
      render: (services: WorkbenchServices, instrument: string | null) => (
        <ResultsView
          focusInstrument={instrument === "program-ad-replay"}
          plan={services.experiment.plan}
          outcome={services.experiment.outcome}
          editors={services.renderEditors("results")}
        />
      ),
    },
  ],
  commands: [
    {
      id: "open-results",
      label: "Open Results",
      action: { kind: "navigate", stage: "results", instrument: null },
    },
  ],
};
