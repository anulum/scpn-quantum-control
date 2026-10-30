// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { NoServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = NoServices;

const CompilerTrace = lazy(() =>
  import("./CompilerTrace").then((module) => ({ default: module.CompilerTrace })),
);
/** The original compiler trace inspector contributed to Build. */
export const compilerModule: StudioModule<Services, WorkspaceViewServices, never> = {
  id: "compiler",
  apiVersion: "2.0",
  requires: { verbs: ["compile"], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "stage",
      id: "compiler-trace",
      label: "Compiler trace",
      stage: "build",
      render: () => <CompilerTrace />,
    },
  ],
};
