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
const OperationsView = lazy(() => import("./OperationsView"));
/** Source-owned offline profiles, policy metadata and review dossiers on the original stage. */
export const operatorsModule: StudioModule<WorkbenchServices, WorkspaceViewServices, "operations"> =
  {
    id: "operations",
    version: "1",
    requires: { verbs: [], schemas: [] },
    stages: [
      {
        id: "operations" as const,
        label: "Devices & Operations",
        group: "context",
        instruments: [],
        render: () => <OperationsView />,
      },
    ],
    commands: [
      {
        id: "open-operations",
        label: "Open Devices & Operations",
        action: { kind: "navigate", stage: "operations", instrument: null },
      },
    ],
  };
