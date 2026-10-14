// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — portable workspace archive download

import type { WorkspaceArchivePreview } from "./workspaceArchive";

/** Download the original admitted portable archive with its unchanged identity and exact text. */
export function downloadWorkspaceArchive(portable: WorkspaceArchivePreview): void {
  const blob = new Blob([portable.json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  try {
    anchor.href = url;
    anchor.download = `workspace-${portable.projectId}-${portable.archiveDigest}.json`;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}
