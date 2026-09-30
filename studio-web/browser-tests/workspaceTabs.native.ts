// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — independent browser tab head reconciliation

import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import type { WorkspaceStore } from "../src/shared/storage/workspaceStore";
import type { WorkspaceArchivePreview } from "../src/shared/storage/workspaceArchive";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

interface PreparedTab {
  readonly connection: WorkspaceStore;
  readonly initial: WorkspaceArchivePreview;
  readonly edited: WorkspaceArchivePreview;
}
let prepared: PreparedTab | null = null;

/** Prepare an independent tab's connection and exact stale head before either tab writes. */
export async function prepareNativeTab(corpusText: string, databaseName: string, initialize: boolean): Promise<Record<string, unknown>> {
  if (prepared !== null) throw new Error("Native tab is already prepared");
  const initial = await conformanceArchive(corpusText, false);
  const edited = await conformanceArchive(corpusText, true);
  const connection = await openWorkspaceStore(conformanceCodecs, databaseName);
  try {
    if (initialize) await connection.save(initial.json, null);
    const loaded = await connection.load(initial.projectId);
    if (loaded?.preview.json !== initial.json || loaded.preview.archiveDigest !== initial.archiveDigest) throw new Error("Independent tab did not read the same prior committed head");
    prepared = { connection, initial, edited };
    return { projectId: initial.projectId, priorArchiveDigest: loaded.preview.archiveDigest, candidateArchiveDigest: edited.archiveDigest, rawHashes: initial.rawHashes };
  } catch (cause: unknown) { connection.close(); throw cause; }
}

/** Commit from the first tab or require the independently prepared second tab to refuse its stale head. */
export async function savePreparedNativeTab(expectConflict: boolean): Promise<Record<string, unknown>> {
  if (prepared === null) throw new Error("Native tab has no independently observed prior head");
  const { connection, initial, edited } = prepared;
  try {
    let conflict: string | null = null;
    try { await connection.save(edited.json, initial.archiveDigest); }
    catch (cause: unknown) {
      if (!expectConflict || !(cause instanceof Error) || !cause.message.includes("changed in another tab")) throw cause;
      conflict = cause.message;
    }
    if (expectConflict && conflict === null) throw new Error("Stale independent tab overwrote the first tab's saved head");
    const actual = await connection.load(initial.projectId);
    if (actual?.preview.json !== edited.json || actual.preview.archiveDigest !== edited.archiveDigest || actual.preview.rawHashes.join(",") !== initial.rawHashes.join(",")) throw new Error("Independent-tab reconciliation changed saved revision or original evidence");
    return { conflict, archiveDigest: actual.preview.archiveDigest, workspaceHash: actual.preview.workspaceHash, documentHashes: actual.preview.documentHashes, rawHashes: actual.preview.rawHashes, boundary: "Two actual page-owned native connections; synthetic metadata only" };
  } finally { connection.close(); prepared = null; }
}

window.addEventListener("pagehide", () => { prepared?.connection.close(); prepared = null; }, { once: true });
