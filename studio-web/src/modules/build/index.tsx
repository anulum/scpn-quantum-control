// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type {
  NavigationServices,
  StageCompositionServices,
  WorkspaceViewServices,
} from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = NavigationServices & StageCompositionServices;

const BuildView = lazy(() => import("./BuildView"));
/** Original compiled instruments and registered authoring editors on their existing addresses. */
export const buildModule: StudioModule<Services, WorkspaceViewServices, "build"> = {
  id: "build",
  apiVersion: "2.0",
  requires: { verbs: ["compile"], schemas: [] },
  stages: [
    {
      id: "build" as const,
      label: "Build",
      group: "primary",
      instruments: ["compile-recompute"],
      render: (services: Services, instrument: string | null) => (
        <BuildView
          focusInstrument={instrument === "compile-recompute"}
          editors={services.renderEditors("build")}
          compareHref={`#/results${services.contextSuffix}`}
        />
      ),
    },
  ],
  commands: [
    {
      id: "open-build",
      label: "Open Build",
      action: { kind: "navigate", stage: "build", instrument: null },
    },
  ],
};
