// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkspaceViewServices } from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = WorkspaceViewServices;

const RunComparison = lazy(() =>
  import("./RunComparison").then((module) => ({ default: module.RunComparison })),
);
const MethodComparison = lazy(() =>
  import("./MethodComparison").then((module) => ({ default: module.MethodComparison })),
);
/** Independently source-bound run and complete method-population comparison controls. */
export const compareModule: StudioModule<Services, WorkspaceViewServices, never> = {
  id: "compare",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "stage",
      id: "run-comparison",
      label: "Run comparison",
      stage: "results",
      render: (services: Services) => (
        <RunComparison sourceJson={services.workspace.draft} rawCodecs={services.codecs} />
      ),
    },
    {
      kind: "stage",
      id: "method-comparison",
      label: "Method comparison and uncertainty",
      stage: "results",
      render: () => <MethodComparison />,
    },
  ],
};
