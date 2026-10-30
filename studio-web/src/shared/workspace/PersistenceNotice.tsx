// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — say when saved work lives only in this page session

import type { WorkspacePersistence } from "../storage/workspacePersistence";

/**
 * A standing notice while workspaces are saved to this page session only.
 *
 * Browser cache needs no notice beyond the export reminder the editor already
 * gives; session storage loses everything on reload, so it is said every time.
 *
 * @param props - Where saved workspaces live, or null before a store opens.
 * @returns The notice, or nothing.
 */
export function PersistenceNotice({
  persistence,
}: {
  /** Browser cache, this page session only, or null before a store opens. */
  readonly persistence: WorkspacePersistence | null;
}) {
  if (persistence !== "session") return null;
  return (
    <p role="note" className="qsp-boundary" aria-label="Session-only storage">
      Session only: this page cannot use browser storage. Saved work disappears when the page closes
      or reloads; export the archive to keep it.
    </p>
  );
}
