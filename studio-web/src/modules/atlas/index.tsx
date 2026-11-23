// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas module

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type {
  NavigationServices,
  SourceServices,
  WorkspaceViewServices,
} from "../../shared/contracts/workbench";

/** Shell services this module uses; the compiler refuses any other. */
type Services = NavigationServices & SourceServices;

const AtlasView = lazy(() => import("./AtlasView"));
/** Browse the committed Atlas snapshot through deep-linkable lenses. */
export const atlasModule: StudioModule<Services, WorkspaceViewServices, "atlas"> = {
  id: "atlas",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "atlas" as const,
      label: "Atlas",
      group: "context",
      instruments: ["architecture", "claims", "evidence", "workspace"],
      render: (services: Services, instrument: string | null) => {
        // The same admitted workspace the explorer shows is overlaid as private objects.
        const preview = services.workspace.preview ?? services.workspace.saved?.preview ?? null;
        return (
          <AtlasView
            instrument={instrument}
            contextSuffix={services.contextSuffix}
            workspace={preview === null ? null : { json: preview.json, codecs: services.codecs }}
            draftHref={services.workspaceHref}
          />
        );
      },
    },
  ],
  capabilities: [
    {
      id: "browse-atlas",
      label: "Browse the committed Atlas snapshot as a graph and as tables",
      maturity: "stable",
      needs: [],
    },
  ],
  commands: [
    {
      id: "open-atlas",
      label: "Open Atlas",
      action: { kind: "navigate", stage: "atlas", instrument: null },
    },
    {
      id: "open-atlas-claims",
      label: "Open the Atlas claims lens",
      action: { kind: "navigate", stage: "atlas", instrument: "claims" },
    },
  ],
};
