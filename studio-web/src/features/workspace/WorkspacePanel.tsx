// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — local workspace editor

import { useState } from "react";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { noWorkspaceProducers, useWorkspace } from "./useWorkspace";

/** Trusted producer registry from the host; imported data cannot install verifiers. */
export interface WorkspacePanelProps {
  /** Existing source-owned codecs; unsupported source formats refuse admission. */
  readonly rawCodecs?: ReadonlyMap<string, RawCodec>;
}

/** Original panel's local workspace editor, atomic save and portable archive recovery. */
export function WorkspacePanel({ rawCodecs = noWorkspaceProducers }: WorkspacePanelProps) {
  const workspace = useWorkspace(rawCodecs);
  const [title, setTitle] = useState("Untitled workspace");
  const exportArchive = (portable: WorkspaceArchivePreview) => {
    const blob = new Blob([portable.json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    try {
      anchor.href = url;
      anchor.download = `workspace-${portable.projectId}-${portable.archiveDigest}.json`;
      anchor.click();
    } finally { URL.revokeObjectURL(url); }
  };
  const preview = workspace.preview;
  const saved = workspace.saved;
  const exportPreview = preview === null ? undefined : () => exportArchive(preview);
  const exportSaved = saved === null ? undefined : () => exportArchive(saved.preview);
  return (
    <section id="/workspace" className="qsp-workspace" aria-label="Local workspace">
      <h3>Local workspace</h3>
      <p>Drafts and immutable revision references stay in this browser. Export is a portable backup; browser cache is not server storage.</p>
      <p>Source formats without an available verifier are refused. This editor grants no execution, hardware or provider authority.</p>
      <label>New project title <input value={title} maxLength={512} onChange={event => setTitle(event.target.value)} /></label>
      <button type="button" disabled={workspace.busy} onClick={() => { void workspace.create(title); }}>Create empty project</button>
      <label>Workspace archive file <input type="file" accept="application/json,.json" disabled={workspace.busy} onChange={event => {
        const file = event.target.files?.[0];
        if (file) void workspace.read(file);
        event.target.value = "";
      }} /></label>
      <label>Workspace archive JSON <textarea value={workspace.draft} disabled={workspace.busy} spellCheck={false} onChange={event => workspace.edit(event.target.value)} /></label>
      <div className="qsp-workspace-actions">
        <button type="button" disabled={workspace.busy || workspace.draft === ""} onClick={() => { void workspace.inspect(); }}>Preview archive</button>
        <button type="button" disabled={workspace.busy || !workspace.storageAvailable || workspace.preview === null} onClick={() => { void workspace.save(); }}>Save draft and revision references</button>
        <button type="button" disabled={workspace.busy || !workspace.storageAvailable} onClick={() => { void workspace.reload(); }}>Reload saved workspace</button>
        <button type="button" disabled={workspace.busy || workspace.preview === null} onClick={exportPreview}>Export preview archive</button>
        <button type="button" disabled={workspace.busy || workspace.saved === null} onClick={exportSaved}>Export saved archive</button>
      </div>
      <p role="status">{workspace.message}</p>
      {workspace.saved && <dl aria-label="Saved workspace identity">
        <dt>Saved project</dt><dd>{workspace.saved.preview.projectId}</dd>
        <dt>Saved workspace digest</dt><dd>{workspace.saved.preview.workspaceHash}</dd>
        <dt>Saved archive digest</dt><dd>{workspace.saved.preview.archiveDigest}</dd>
        <dt>Durability</dt><dd>Browser cache; independent exported backup required</dd>
      </dl>}
      {workspace.preview && <dl aria-label="Archive preview">
        <dt>Version</dt><dd>{workspace.preview.schema} — no migration required</dd>
        <dt>Project</dt><dd>{workspace.preview.projectId}</dd>
        <dt>Workspace digest</dt><dd>{workspace.preview.workspaceHash}</dd>
        <dt>Immutable document digests</dt><dd>{workspace.preview.documentHashes.join(" · ") || "No revision documents"}</dd>
        <dt>Original source digests</dt><dd>{workspace.preview.rawHashes.join(" · ") || "No source artifacts"}</dd>
        <dt>Archive digest</dt><dd>{workspace.preview.archiveDigest}</dd>
        <dt>Members</dt><dd>{workspace.preview.memberNames.join(" · ") || "Root manifest only"}</dd>
      </dl>}
    </section>
  );
}
