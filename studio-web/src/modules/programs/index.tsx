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
const ProgramEditor = lazy(() =>
  import("./ProgramEditor").then((module) => ({ default: module.ProgramEditor })),
);
/** The original supported program authoring controls contributed to Build. */
export const programsModule: StudioModule<WorkbenchServices, WorkspaceViewServices, never> = {
  id: "programs",
  apiVersion: "2.0",
  requires: { verbs: ["compile"], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "stage",
      id: "program-source",
      label: "Program source",
      stage: "build",
      render: () => <ProgramEditor />,
    },
  ],
};
