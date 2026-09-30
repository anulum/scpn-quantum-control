// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact draft formatting snapshot recovery

import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

/** Save a formatting-only draft without rewriting original revisions and refuse its stale predecessor. */
export async function runNativeFormatting(corpusText: string): Promise<Record<string, unknown>> {
  const initial = await conformanceArchive(corpusText, false);
  const formatted = "\n" + initial.json + "\n";
  const store = await openWorkspaceStore(conformanceCodecs, `workspace-formatting-${crypto.randomUUID()}`);
  try {
    await store.save(initial.json, null);
    const saved = await store.save(formatted, initial.archiveDigest);
    if (saved.preview.archiveDigest === initial.archiveDigest || saved.preview.workspaceHash !== initial.workspaceHash || saved.preview.documentHashes.join(",") !== initial.documentHashes.join(",") || saved.preview.rawHashes.join(",") !== initial.rawHashes.join(",")) throw new Error("Formatting snapshot changed original revision/evidence identity or reused its predecessor");
    const reloaded = await store.load(initial.projectId);
    if (reloaded?.preview.json !== formatted || reloaded.preview.archiveDigest !== saved.preview.archiveDigest) throw new Error("Native reload lost exact formatting-only draft");
    let stale: string | null = null;
    try { await store.save(initial.json, initial.archiveDigest); }
    catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("changed in another tab")) throw cause;
      stale = cause.message;
    }
    if (stale === null || (await store.load(initial.projectId))?.preview.json !== formatted) throw new Error("Stale formatting predecessor overwrote the exact saved draft");
    return { priorArchive: initial.json, archive: saved.preview.json, priorArchiveDigest: initial.archiveDigest, archiveDigest: saved.preview.archiveDigest, workspaceHash: saved.preview.workspaceHash, documentHashes: saved.preview.documentHashes, rawHashes: saved.preview.rawHashes, stale, boundary: "Actual native formatting snapshot; no numerical or hardware claim" };
  } finally { store.close(); }
}
