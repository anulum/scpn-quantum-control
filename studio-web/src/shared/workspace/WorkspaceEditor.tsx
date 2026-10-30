// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — local workspace editor

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { RawCodec } from "../contracts";
import { downloadWorkspaceArchive } from "../storage/workspaceDownload";
export { downloadWorkspaceArchive } from "../storage/workspaceDownload";
import type { StoredWorkspace } from "../storage/workspaceStore";
import type { WorkspaceController } from "./workspaceService";
import { PersistenceNotice } from "./PersistenceNotice";

/** Trusted producer registry from the host; imported data cannot install verifiers. */
export interface WorkspacePanelProps {
  /** Existing source-owned codecs; unsupported source formats refuse admission. */
  readonly rawCodecs?: ReadonlyMap<string, RawCodec>;
  /** Observe the original store's admitted head; navigation cannot manufacture one. */
  readonly onSavedWorkspace?: (saved: StoredWorkspace | null) => void;
}

/** Original editor supplied with its single existing controller by the workbench. */
export interface WorkspaceEditorProps extends WorkspacePanelProps {
  /** Existing connection/controller; this presentation never opens a second store. */ readonly workspace: WorkspaceController;
  /** Registered settings views for the current admitted preview. */ readonly previewViews?: ReactNode;
  /** Registered editors for the current admitted revision source. */ readonly revisionViews?: ReactNode;
}

/** Preserve all original editing/import/export controls around the supplied original store owner. */
export function WorkspaceEditor({
  workspace,
  onSavedWorkspace,
  previewViews,
  revisionViews,
}: WorkspaceEditorProps) {
  useEffect(() => {
    onSavedWorkspace?.(workspace.saved);
  }, [onSavedWorkspace, workspace.saved]);
  const [title, setTitle] = useState("Untitled workspace");
  const preview = workspace.preview;
  const saved = workspace.saved;
  const exportPreview = preview === null ? undefined : () => downloadWorkspaceArchive(preview);
  const exportSaved = saved === null ? undefined : () => downloadWorkspaceArchive(saved.preview);
  return (
    <section id="/workspace" className="qsp-workspace" aria-label="Local workspace">
      <h3>Local workspace</h3>
      <p>
        Drafts and immutable revision references stay in this browser. Export is a portable backup;
        browser cache is not server storage.
      </p>
      <p>
        Source formats without an available verifier are refused. This editor grants no execution,
        hardware or provider authority.
      </p>
      <label>
        New project title{" "}
        <input value={title} maxLength={512} onChange={(event) => setTitle(event.target.value)} />
      </label>
      <button
        type="button"
        disabled={workspace.busy}
        onClick={() => {
          void workspace.create(title);
        }}
      >
        Create empty project
      </button>
      <label>
        Workspace archive file{" "}
        <input
          type="file"
          accept="application/json,.json"
          disabled={workspace.busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void workspace.read(file);
            event.target.value = "";
          }}
        />
      </label>
      <label>
        Workspace archive JSON{" "}
        <textarea
          value={workspace.draft}
          disabled={workspace.busy}
          spellCheck={false}
          onChange={(event) => workspace.edit(event.target.value)}
        />
      </label>
      <div className="qsp-workspace-actions">
        <button
          type="button"
          disabled={workspace.busy || workspace.draft === ""}
          onClick={() => {
            void workspace.inspect();
          }}
        >
          Preview archive
        </button>
        <button
          type="button"
          disabled={workspace.busy || !workspace.storageAvailable || workspace.preview === null}
          onClick={() => {
            void workspace.save();
          }}
        >
          Save draft and revision references
        </button>
        <button
          type="button"
          disabled={workspace.busy || !workspace.storageAvailable}
          onClick={() => {
            void workspace.reload();
          }}
        >
          Reload saved workspace
        </button>
        <button
          type="button"
          disabled={workspace.busy || workspace.preview === null}
          onClick={exportPreview}
        >
          Export preview archive
        </button>
        <button
          type="button"
          disabled={workspace.busy || workspace.saved === null}
          onClick={exportSaved}
        >
          Export saved archive
        </button>
      </div>
      <p role="status">{workspace.message}</p>
      <PersistenceNotice persistence={workspace.persistence} />
      {workspace.saved && (
        <dl aria-label="Saved workspace identity">
          <dt>Saved project</dt>
          <dd>{workspace.saved.preview.projectId}</dd>
          <dt>Saved workspace digest</dt>
          <dd>{workspace.saved.preview.workspaceHash}</dd>
          <dt>Saved archive digest</dt>
          <dd>{workspace.saved.preview.archiveDigest}</dd>
          <dt>Durability</dt>
          <dd>Browser cache; independent exported backup required</dd>
        </dl>
      )}
      {workspace.preview && (
        <dl aria-label="Archive preview">
          <dt>Version</dt>
          <dd>{workspace.preview.schema} — no migration required</dd>
          <dt>Project</dt>
          <dd>{workspace.preview.projectId}</dd>
          <dt>Workspace digest</dt>
          <dd>{workspace.preview.workspaceHash}</dd>
          <dt>Immutable document digests</dt>
          <dd>{workspace.preview.documentHashes.join(" · ") || "No revision documents"}</dd>
          <dt>Original source digests</dt>
          <dd>{workspace.preview.rawHashes.join(" · ") || "No source artifacts"}</dd>
          <dt>Archive digest</dt>
          <dd>{workspace.preview.archiveDigest}</dd>
          <dt>Members</dt>
          <dd>{workspace.preview.memberNames.join(" · ") || "Root manifest only"}</dd>
        </dl>
      )}
      {workspace.preview && previewViews}
      {(workspace.preview ||
        (workspace.saved?.preview.json === workspace.draft && workspace.saved.preview)) &&
        revisionViews}
    </section>
  );
}
