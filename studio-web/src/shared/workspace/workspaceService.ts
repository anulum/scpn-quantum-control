// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workspace service contract

import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";
import type { StoredWorkspace } from "../storage/workspaceStore";

/** Workspace controller's complete observable draft and saved-state boundary. */
export interface WorkspaceController {
  /** Exact editable archive text; changing it invalidates the preview only. */
  readonly draft: string;
  /** Last immutable committed browser archive; editing never rebinds it. */
  readonly saved: StoredWorkspace | null;
  /** Complete current import/save preview, or null on refusal/edit. */
  readonly preview: WorkspaceArchivePreview | null;
  /** In-progress operation; no successful storage claim is made until commit. */
  readonly busy: boolean;
  /** Whether the native storage connection is currently available. */
  readonly storageAvailable: boolean;
  /** User-visible state or explicit refusal. */
  readonly message: string;
  /** Replace an unsaved draft without modifying saved data. */
  edit(text: string): void;
  /** Create a genuinely empty project, with no synthetic revision or evidence. */
  create(title: string): Promise<void>;
  /** Read bounded local archive bytes without automatic import or mutation. */
  read(file: File): Promise<void>;
  /** Validate the complete draft and show the expected prior archive identity. */
  inspect(): Promise<void>;
  /** Commit the exact current preview and its immutable references atomically. */
  save(): Promise<void>;
  /** Commit a validated child archive only while its original source remains selected. */
  saveRevision(
    archive: WorkspaceArchivePreview,
    priorJson: string,
    signal: AbortSignal,
  ): Promise<void>;
  /** Reload current browser selection, preserving the editor on corrupt state. */
  reload(): Promise<void>;
}
