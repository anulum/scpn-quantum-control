// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — choose browser or session workspace storage

import type { RawCodec } from "../contracts/graph";
import { createSessionWorkspaceStore } from "./sessionWorkspaceStore";
import { openWorkspaceStore } from "./workspaceStore";
import type { WorkspaceStore } from "./workspaceStore";

/** Where saved workspaces live: browser cache, or this page session only. */
export type WorkspacePersistence = "browser" | "session";

/** An opened workspace store and what kind it is. */
export interface OpenedWorkspaceStore {
  /** The store every save and load goes through. */
  readonly store: WorkspaceStore;
  /** Browser cache, or this page session only. */
  readonly persistence: WorkspacePersistence;
  /** What to say when the store holds no saved project yet. */
  readonly emptyMessage: string;
}

const browserEmpty =
  "No saved project found. Browser cache may have been evicted; restore an exported copy or create an empty project.";
const sessionEmpty =
  "Browser storage is not available here. Saved work is kept for this page session only; export it to keep it.";

/**
 * Whether an opening failure means this page has no IndexedDB at all.
 *
 * A missing IndexedDB, or a page whose origin may not use it (a sandboxed,
 * opaque-origin frame throws SecurityError), is a property of the page. A
 * blocked or timed-out open, or a failing database, is not: those stay refusals.
 *
 * @param cause - What opening the browser store threw.
 * @returns True when the page cannot have browser storage.
 */
export function isBrowserStorageUnavailable(cause: unknown): boolean {
  if (cause instanceof DOMException) return cause.name === "SecurityError";
  return cause instanceof Error && cause.message.startsWith("IndexedDB unavailable");
}

/**
 * Open browser storage, or session storage when the page cannot have any.
 *
 * @param rawCodecs - Trusted source verifiers used to admit archives.
 * @param openBrowserStore - How the browser store is opened.
 * @returns The store, its kind and its empty-state message.
 * @throws Whatever opening the browser store threw, unless the page has no browser storage.
 */
export async function openWorkspacePersistence(
  rawCodecs: ReadonlyMap<string, RawCodec>,
  openBrowserStore: (
    codecs: ReadonlyMap<string, RawCodec>,
  ) => Promise<WorkspaceStore> = openWorkspaceStore,
): Promise<OpenedWorkspaceStore> {
  try {
    const store = await openBrowserStore(rawCodecs);
    return { store, persistence: "browser", emptyMessage: browserEmpty };
  } catch (cause: unknown) {
    if (!isBrowserStorageUnavailable(cause)) throw cause;
    return {
      store: createSessionWorkspaceStore(rawCodecs),
      persistence: "session",
      emptyMessage: sessionEmpty,
    };
  }
}
