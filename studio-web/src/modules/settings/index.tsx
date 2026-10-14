// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered source-owned settings projection

import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
import { SettingsDocumentInspector } from "./SettingsDocumentInspector";
import { SettingsInspector } from "./SettingsInspector";

/** Display the original resolver's settings without recomputing policy or granting authority. */
export const settingsModule: StudioModule<WorkbenchServices, WorkspaceViewServices, never> = {
  id: "settings",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [],
  inspectors: [
    {
      kind: "document",
      id: "settings-document",
      label: "Settings document",
      schemas: ["resolved_settings.v1"],
      render: (_services, document) => <SettingsDocumentInspector document={document} />,
    },
  ],
  settings: [
    {
      kind: "document",
      id: "resolved-settings",
      label: "Resolved settings",
      schemas: ["quantum_workspace_archive.v1"],
      render: ({ workspace }, document) => {
        const preview = workspace.preview;
        if (
          preview === null ||
          preview.archiveDigest !== document.sha256 ||
          preview.json !== document.content
        ) {
          return (
            <p role="alert">
              Settings source changed. The original document is retained; preview its admitted
              workspace before inspection.
            </p>
          );
        }
        return <SettingsInspector preview={preview} />;
      },
    },
  ],
};
