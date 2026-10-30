// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered evidence catalogue destination

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
const EvidenceCatalogueView = lazy(() => import("./EvidenceCatalogueView"));
/** Committed capability, support, gradient-plan and scorecard surfaces on their own destination. */
export const evidenceModule: StudioModule<WorkbenchServices, WorkspaceViewServices, "evidence"> = {
  id: "evidence",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "evidence" as const,
      label: "Evidence catalogue",
      group: "context",
      instruments: [],
      render: () => <EvidenceCatalogueView />,
    },
  ],
  commands: [
    {
      id: "open-evidence",
      label: "Open Evidence catalogue",
      action: { kind: "navigate", stage: "evidence", instrument: null },
    },
  ],
};
