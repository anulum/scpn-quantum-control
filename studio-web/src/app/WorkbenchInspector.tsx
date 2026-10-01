// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — requested versus admitted identity

import type { StoredWorkspace } from "../shared/storage/workspaceStore";
import type { WorkbenchContext } from "./routing";

/** Shared projection over requested URL identity and the original storage owner's admitted head. */
export function WorkbenchInspector({ context, saved }: {
  context: WorkbenchContext;
  saved: StoredWorkspace | null;
}) {
  return (
    <aside className="qsp-workbench-inspector" aria-label="Workbench inspector">
      <h3>Workbench inspector</h3>
      <dl>
        <dt>Requested project</dt><dd>{context.project ?? "Not selected"}</dd>
        <dt>Requested revision</dt><dd>{context.revision ?? "Not selected"}</dd>
        <dt>Requested snapshot</dt><dd>{context.snapshot ?? "Not selected"}</dd>
      </dl>
      <p>URL selections are not loaded or admitted by navigation. Open and validate an archive in Workspace.</p>
      {saved === null ? <p>No admitted saved workspace is available.</p> : (
        <dl>
          <dt>Admitted saved project</dt><dd>{saved.preview.projectId}</dd>
          <dt>Admitted workspace digest</dt><dd>{saved.preview.workspaceHash}</dd>
          <dt>Admitted archive digest</dt><dd>{saved.preview.archiveDigest}</dd>
          <dt>Immutable document digests</dt><dd>{saved.preview.documentHashes.join(" · ") || "No revision documents"}</dd>
          <dt>Durability</dt><dd>Browser cache; exported backup required</dd>
        </dl>
      )}
    </aside>
  );
}
