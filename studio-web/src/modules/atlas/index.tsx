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
const UnavailableView = lazy(() => import("./UnavailableView"));
/** Preserve the truthful unavailable Atlas stage without inventing a snapshot browser. */
export const atlasModule: StudioModule<WorkbenchServices, WorkspaceViewServices, "atlas"> = {
  id: "atlas",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "atlas" as const,
      label: "Atlas",
      group: "primary",
      instruments: [],
      render: (services: WorkbenchServices) => (
        <UnavailableView view="atlas" workspaceHref={services.workspaceHref} />
      ),
    },
  ],
  commands: [
    {
      id: "open-atlas",
      label: "Open Atlas",
      action: { kind: "navigate", stage: "atlas", instrument: null },
    },
  ],
};
