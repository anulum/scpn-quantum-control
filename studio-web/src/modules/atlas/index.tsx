// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { NavigationServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = NavigationServices;

const UnavailableView = lazy(() => import("./UnavailableView"));
/** Preserve the truthful unavailable Atlas stage without inventing a snapshot browser. */
export const atlasModule: StudioModule<Services, WorkspaceViewServices, "atlas"> = {
  id: "atlas",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "atlas" as const,
      label: "Atlas",
      group: "primary",
      instruments: [],
      render: (services: Services) => (
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
