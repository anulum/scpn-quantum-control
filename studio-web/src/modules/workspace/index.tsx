// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workbench module

import type { StudioModule } from "../../shared/contracts/module";
import type {
  SourceServices,
  StageCompositionServices,
  WorkspaceViewServices,
} from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = SourceServices & StageCompositionServices;

/** Original workspace landing stage; persistence remains in the shared shell owner. */
export const workspaceModule: StudioModule<Services, WorkspaceViewServices, "workspace"> = {
  id: "workspace",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "workspace" as const,
      label: "Workspace",
      group: "primary",
      instruments: [],
      render: (services: Services) => services.overview(services.renderEditors("workspace")),
    },
  ],
  explorer: [
    {
      id: "workspace-sources",
      label: "Workspace sources",
      kinds: ["archive", "source"],
      nodes: (services: Services) =>
        services.documents.map((document) => ({
          id: `${document.schema}:${document.sha256}`,
          kind: document.schema === "quantum_workspace_archive.v1" ? "archive" : "source",
          label: `${document.schema} · ${document.sha256.slice(0, 12)}`,
          document,
        })),
    },
  ],
  commands: [
    {
      id: "open-workspace",
      label: "Open Workspace",
      action: { kind: "navigate", stage: "workspace", instrument: null },
    },
  ],
};
