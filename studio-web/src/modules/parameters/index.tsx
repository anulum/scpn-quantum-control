// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered parameter revision editor

import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
import { ParameterWorkspace } from "./ParameterWorkspace";

/** Original parameter editor and save owner behind the workspace document slot. */
export const parametersModule: StudioModule<WorkbenchServices, WorkspaceViewServices, never> = {
  id: "parameters",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "document",
      id: "parameter-revision",
      label: "Parameter revision",
      schemas: ["quantum_workspace_archive.v1"],
      render: ({ workspace, codecs }, document) => {
        const preview =
          workspace.preview ??
          (workspace.saved?.preview.json === workspace.draft ? workspace.saved.preview : null);
        if (
          preview === null ||
          preview.archiveDigest !== document.sha256 ||
          preview.json !== document.content
        ) {
          return (
            <p role="alert">
              Parameter source changed. The original document is retained; select its admitted
              workspace before editing.
            </p>
          );
        }
        return (
          <ParameterWorkspace
            preview={preview}
            rawCodecs={codecs}
            saveArchive={workspace.saveRevision}
          />
        );
      },
    },
  ],
};
