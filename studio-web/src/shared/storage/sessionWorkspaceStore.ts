// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — session-only workspace storage for pages without IndexedDB

import type { RawCodec } from "../contracts/graph";
import { previewWorkspaceArchive } from "./workspaceArchive";
import { requireCanonicalProjectId } from "./workspaceStore";
import type { StoredWorkspace, WorkspaceStore } from "./workspaceStore";

const changedElsewhere =
  "Workspace changed in another tab or was evicted; reload and explicitly reconcile before saving";

/**
 * Keep workspaces in the memory of this page when the browser offers no IndexedDB.
 *
 * Saving follows the browser store's rules: the whole archive is admitted
 * first and the expected prior archive identity must match the current head.
 * An archive's identity is the digest of its exact text, so one identity always
 * names the same bytes; the browser store's check against stored bytes that
 * changed underneath has nothing to guard here, since this memory has no other
 * writer. Nothing
 * survives closing or reloading the page, so every stored workspace is marked
 * session-only and export remains the backup.
 *
 * @param rawCodecs - Trusted source verifiers used to admit archives.
 * @returns A store with the same contract as the browser store.
 */
export function createSessionWorkspaceStore(
  rawCodecs: ReadonlyMap<string, RawCodec> = new Map(),
): WorkspaceStore {
  const codecs = new Map(rawCodecs);
  const archives = new Map<string, string>();
  const heads = new Map<string, string>();
  let selected: string | null = null;
  let open = true;

  function stored(preview: StoredWorkspace["preview"]): StoredWorkspace {
    return Object.freeze({ preview, durability: "session-only-export-required" });
  }

  function requireOpen(): void {
    if (!open) throw new Error("Workspace session store closed; reopen the workspace");
  }

  async function load(projectId: string): Promise<StoredWorkspace | null> {
    requireOpen();
    requireCanonicalProjectId(projectId);
    const digest = heads.get(projectId);
    if (digest === undefined) return null;
    const json = archives.get(`${projectId}/${digest}`) as string;
    return stored(await previewWorkspaceArchive(json, codecs));
  }

  return Object.freeze({
    async save(
      json: string,
      expectedArchiveDigest: string | null,
      signal?: AbortSignal,
    ): Promise<StoredWorkspace> {
      requireOpen();
      if (signal?.aborted)
        throw new Error("Workspace save cancelled before transaction; prior state retained");
      const preview = await previewWorkspaceArchive(json, codecs);
      if (expectedArchiveDigest !== null && !/^[0-9a-f]{64}$/.test(expectedArchiveDigest))
        throw new Error("expected prior archive digest invalid");
      if (signal?.aborted)
        throw new Error("Workspace save cancelled before transaction; prior state retained");
      if ((heads.get(preview.projectId) ?? null) !== expectedArchiveDigest)
        throw new Error(changedElsewhere);
      archives.set(`${preview.projectId}/${preview.archiveDigest}`, preview.json);
      heads.set(preview.projectId, preview.archiveDigest);
      selected = preview.projectId;
      return stored(preview);
    },
    load,
    async selectedProject(): Promise<string | null> {
      requireOpen();
      return selected;
    },
    close(): void {
      open = false;
    },
  });
}
