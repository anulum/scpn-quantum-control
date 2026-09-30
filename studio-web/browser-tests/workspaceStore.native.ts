// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native browser workspace conformance

import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { previewWorkspaceArchive } from "../src/shared/storage/workspaceArchive";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

function equal(actual: unknown, expected: unknown, message: string): void {
  if (actual !== expected) throw new Error(message);
}

/** Observe actual native transaction abort, immutable reload and stale-head refusal. */
export async function runNativeWorkspaceCases(corpusText: string): Promise<Record<string, unknown>> {
  const initial = await conformanceArchive(corpusText, false);
  const edited = await conformanceArchive(corpusText, true);
  const databaseName = `workspace-native-conformance-${crypto.randomUUID()}`;
  const first = await openWorkspaceStore(conformanceCodecs, databaseName);
  try { await first.save(initial.json, null); }
  finally { first.close(); }
  const store = await openWorkspaceStore(conformanceCodecs, databaseName);
  let observedNativeTransactions = 0;
  const cancellation = new AbortController();
  const nativeTransaction = IDBDatabase.prototype.transaction;
  try {
    equal(await store.selectedProject(), initial.projectId, "Selected project was not committed atomically");
    const restored = await store.load(initial.projectId);
    equal(restored?.preview.json, initial.json, "Native reload changed exact source text");
    equal(restored?.preview.workspaceHash, initial.workspaceHash, "Native reload changed workspace identity");
    // Forward the real constructor and deliver a real cancellation after it starts.
    // No native result or error is replaced; the public AbortSignal drives rollback.
    IDBDatabase.prototype.transaction = function(this: IDBDatabase, names: string | string[], mode?: IDBTransactionMode, options?: IDBTransactionOptions): IDBTransaction {
      const transaction = nativeTransaction.call(this, names, mode, options);
      if (this.name === databaseName && mode === "readwrite") {
        observedNativeTransactions++;
        queueMicrotask(() => cancellation.abort());
      }
      return transaction;
    };
    let interrupted = false;
    try { await store.save(edited.json, initial.archiveDigest, cancellation.signal); }
    catch (cause: unknown) {
      interrupted = cause instanceof Error && cause.message.includes("interrupted");
    } finally { IDBDatabase.prototype.transaction = nativeTransaction; }
    equal(observedNativeTransactions, 1, "Cancellation never reached an actual native write transaction");
    equal(interrupted, true, "Interrupted native transaction did not report failure");
    equal((await store.load(initial.projectId))?.preview.json, initial.json, "Interrupted native write replaced the prior revision");
    const committed = await store.save(edited.json, initial.archiveDigest);
    equal(committed.preview.archiveDigest, edited.archiveDigest, "Edited draft did not commit exactly");
    let staleRefused = false;
    try { await store.save(initial.json, initial.archiveDigest); }
    catch (cause: unknown) { staleRefused = cause instanceof Error && cause.message.includes("changed in another tab"); }
    equal(staleRefused, true, "Stale prior head overwrote the committed draft");
    equal((await store.load(initial.projectId))?.preview.json, edited.json, "Conflict refusal changed current saved draft");
    const roundtrip = await previewWorkspaceArchive(edited.json, conformanceCodecs);
    equal(roundtrip.workspaceHash, edited.workspaceHash, "Export changed workspace digest");
    equal(roundtrip.documentHashes.join(","), edited.documentHashes.join(","), "Export changed saved revision digests");
    equal(roundtrip.rawHashes.join(","), initial.rawHashes.join(","), "Editing rebound or changed original evidence identities");
    return { databaseName, projectId: edited.projectId, archive: edited.json,
      archiveDigest: edited.archiveDigest, workspaceHash: edited.workspaceHash,
      documentHashes: edited.documentHashes, rawHashes: edited.rawHashes,
      interruptedNativeTransactions: observedNativeTransactions,
      staleRefused, priorRetainedAfterInterrupt: true,
      boundary: "Real IndexedDB and original synthetic metadata corpus; no numerical, provider or hardware support claim" };
  } finally { IDBDatabase.prototype.transaction = nativeTransaction; store.close(); }
}

/** Import the exact export in a fresh browser context and compare every admitted identity. */
export async function importNativeWorkspace(json: string): Promise<Record<string, unknown>> {
  const preview = await previewWorkspaceArchive(json, conformanceCodecs);
  const store = await openWorkspaceStore(conformanceCodecs, `workspace-fresh-conformance-${crypto.randomUUID()}`);
  try {
    equal(await store.selectedProject(), null, "Fresh browser context already has selected state");
    equal(await store.load(preview.projectId), null, "Fresh browser context already has this project");
    await store.save(json, null);
    const restored = await store.load(preview.projectId);
    equal(restored?.preview.json, json, "Clean-context import changed exact exported source");
    return { projectId: restored?.preview.projectId, archiveDigest: restored?.preview.archiveDigest,
      workspaceHash: restored?.preview.workspaceHash, documentHashes: restored?.preview.documentHashes,
      rawHashes: restored?.preview.rawHashes };
  } finally { store.close(); }
}
