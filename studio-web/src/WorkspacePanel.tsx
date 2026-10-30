// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — standalone workspace composition over registered modules

import { declaredBy, workbenchRegistry } from "./app/moduleRegistry";
import { studioManifest } from "./shared/browser/data";
import { ModuleDocumentViews } from "./app/moduleViews/ModuleDocumentViews";
import { WorkspaceEditor as WorkspacePresentation } from "./shared/workspace/WorkspaceEditor";
import type { WorkspaceEditorProps, WorkspacePanelProps } from "./shared/workspace/WorkspaceEditor";
import { noWorkspaceProducers, useWorkspace } from "./shared/workspace/useWorkspace";

export type { WorkspaceEditorProps, WorkspacePanelProps } from "./shared/workspace/WorkspaceEditor";
export { downloadWorkspaceArchive } from "./shared/storage/workspaceDownload";

const declared = declaredBy(studioManifest);

/** Compose the original editor with this build's registered document and settings views. */
export function WorkspaceEditor(props: WorkspaceEditorProps) {
  const { workspace, rawCodecs = noWorkspaceProducers } = props;
  const services = { workspace, codecs: rawCodecs };
  const preview = workspace.preview;
  const revision =
    preview ?? (workspace.saved?.preview.json === workspace.draft ? workspace.saved.preview : null);
  return (
    <WorkspacePresentation
      {...props}
      previewViews={
        props.previewViews ??
        (preview === null ? undefined : (
          <ModuleDocumentViews
            registry={workbenchRegistry}
            slot="settings"
            services={services}
            declared={declared}
            document={{
              schema: preview.schema,
              sha256: preview.archiveDigest,
              content: preview.json,
            }}
          />
        ))
      }
      revisionViews={
        props.revisionViews ??
        (revision === null ? undefined : (
          <ModuleDocumentViews
            registry={workbenchRegistry}
            slot="editors"
            services={services}
            declared={declared}
            document={{
              schema: revision.schema,
              sha256: revision.archiveDigest,
              content: revision.json,
            }}
          />
        ))
      }
    />
  );
}

/** Open one original persistence controller and preserve standalone import/export and callbacks. */
export function WorkspacePanel({
  rawCodecs = noWorkspaceProducers,
  onSavedWorkspace,
}: WorkspacePanelProps) {
  const workspace = useWorkspace(rawCodecs);
  return (
    <WorkspaceEditor
      workspace={workspace}
      rawCodecs={rawCodecs}
      {...(onSavedWorkspace === undefined ? {} : { onSavedWorkspace })}
    />
  );
}
