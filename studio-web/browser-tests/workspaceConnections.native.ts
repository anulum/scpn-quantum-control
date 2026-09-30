// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native workspace connection lifecycle

import { openWorkspaceStore, workspaceOpenTimeoutMilliseconds } from "../src/shared/storage/workspaceStore";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

async function nativeRefusal(action: () => Promise<unknown>, name: string): Promise<string> {
  try { await action(); }
  catch (cause: unknown) {
    if (!(cause instanceof DOMException) || cause.name !== name) throw cause;
    return cause.name;
  }
  throw new Error(`Expected actual native ${name}`);
}

/** Reopen exact saved state, observe real version-change closure, and refuse unsupported cache versions. */
export async function runNativeConnectionCases(corpusText: string): Promise<Record<string, unknown>> {
  const initial = await conformanceArchive(corpusText, false);
  const databaseName = `workspace-connections-${crypto.randomUUID()}`;
  const invalidNames: string[] = [];
  for (const name of ["   ", "x".repeat(257)]) {
    try {
      const unexpected = await openWorkspaceStore(conformanceCodecs, name);
      unexpected.close();
      throw new Error("Unbounded or empty database name admitted");
    } catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("bounded database name required")) throw cause;
      invalidNames.push(cause.message);
    }
  }
  const first = await openWorkspaceStore(conformanceCodecs, databaseName);
  let cancelled: string | null = null;
  const interruptedReads: Record<string, string> = {};
  const nativeTransaction = IDBDatabase.prototype.transaction;
  try {
    await first.save(initial.json, null);
    const repeated = await first.save(initial.json, initial.archiveDigest);
    if (repeated.preview.json !== initial.json || repeated.preview.archiveDigest !== initial.archiveDigest) throw new Error("Repeated save changed the immutable native archive");
    for (const operation of ["selection", "archive"] as const) {
      let observed = 0;
      IDBDatabase.prototype.transaction = function(this: IDBDatabase, names: string | string[], mode?: IDBTransactionMode, options?: IDBTransactionOptions): IDBTransaction {
        const transaction = nativeTransaction.call(this, names, mode, options);
        if (this.name === databaseName && mode === "readonly") {
          observed++;
          queueMicrotask(() => transaction.abort());
        }
        return transaction;
      };
      try {
        try { await (operation === "selection" ? first.selectedProject() : first.load(initial.projectId)); }
        catch (cause: unknown) {
          if (!(cause instanceof Error) || !cause.message.includes("transaction interrupted")) throw cause;
          interruptedReads[operation] = cause.message;
        }
      } finally { IDBDatabase.prototype.transaction = nativeTransaction; }
      if (observed !== 1 || interruptedReads[operation] === undefined) throw new Error(`Actual ${operation} read did not abort through its native transaction`);
      if ((await first.load(initial.projectId))?.preview.json !== initial.json || await first.selectedProject() !== initial.projectId) throw new Error("Interrupted native read changed original saved state");
    }
    const pendingCancellation = new AbortController();
    const pendingSave = first.save(initial.json, initial.archiveDigest, pendingCancellation.signal);
    pendingCancellation.abort();
    try { await pendingSave; throw new Error("Save admitted cancellation during real preview"); }
    catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("cancelled before transaction")) throw cause;
      interruptedReads["save-preview"] = cause.message;
    }
    const cancellation = new AbortController();
    cancellation.abort();
    try { await first.save(initial.json, initial.archiveDigest, cancellation.signal); }
    catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("cancelled before transaction")) throw cause;
      cancelled = cause.message;
    }
    if (cancelled === null || (await first.load(initial.projectId))?.preview.json !== initial.json || await first.selectedProject() !== initial.projectId) throw new Error("Pre-cancelled save admitted or changed actual stored state");
  }
  finally { IDBDatabase.prototype.transaction = nativeTransaction; first.close(); }
  const closed = await nativeRefusal(() => first.selectedProject(), "InvalidStateError");
  const reopened = await openWorkspaceStore(conformanceCodecs, databaseName);
  try {
    const restored = await reopened.load(initial.projectId);
    if (restored?.preview.json !== initial.json || restored.preview.archiveDigest !== initial.archiveDigest || await reopened.selectedProject() !== initial.projectId) throw new Error("Connection reopen changed exact saved snapshot or selection");
    const upgraded = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(databaseName, 2);
      let refused = false;
      request.onblocked = () => {
        refused = true;
        reject(new Error("Actual version change was blocked; original workspace connection did not close"));
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        if (refused) request.result.close();
        else resolve(request.result);
      };
    });
    try {
      const changed = await nativeRefusal(() => reopened.load(initial.projectId), "InvalidStateError");
      const future = await nativeRefusal(() => openWorkspaceStore(conformanceCodecs, databaseName), "VersionError");
      const retained = await new Promise<{ archive: unknown; head: unknown; selection: unknown }>((resolve, reject) => {
        const transaction = upgraded.transaction(["archives", "heads"], "readonly");
        const archive = transaction.objectStore("archives").get(`${initial.projectId}/${initial.archiveDigest}`);
        const head = transaction.objectStore("heads").get(initial.projectId);
        const selection = transaction.objectStore("heads").get("$selected");
        transaction.oncomplete = () => resolve({ archive: archive.result as unknown, head: head.result as unknown, selection: selection.result as unknown });
        transaction.onabort = () => reject(transaction.error);
      });
      if (retained.archive !== initial.json || retained.selection !== initial.projectId || JSON.stringify(retained.head) !== JSON.stringify({ archiveDigest: initial.archiveDigest, workspaceHash: initial.workspaceHash })) throw new Error("Unsupported version refusal changed actual stored archive or head");
      return { invalidNames, interruptedReads, cancelled, closed, changed, future, archiveDigest: initial.archiveDigest, workspaceHash: initial.workspaceHash, retainedSelection: retained.selection, boundary: "Actual native connection/version lifecycle in an owned test database; no migration or science claim" };
    } finally { upgraded.close(); }
  } finally { reopened.close(); }
}


/** Refuse a real queued open behind deletion and close the original late native connection. */
export async function runNativeBlockedOpenCases(): Promise<Record<string, unknown>> {
  const name = `workspace-blocked-open-${crypto.randomUUID()}`;
  for (const milliseconds of [0, -1, 1.5, NaN, Infinity, workspaceOpenTimeoutMilliseconds + 1]) {
    try { await openWorkspaceStore(conformanceCodecs, name, milliseconds); throw new Error("Invalid opening timeout admitted"); }
    catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("open timeout must")) throw cause;
    }
  }
  const held = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(name, 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  let blockedDeletionObserved = false;
  const deletion = new Promise<void>((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onblocked = () => { blockedDeletionObserved = true; };
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  let refusal: string | null = null;
  try {
    try {
      const unexpected = await openWorkspaceStore(conformanceCodecs, name, 50);
      unexpected.close();
      throw new Error("Native queued open ignored its declared deadline");
    } catch (cause: unknown) {
      if (!(cause instanceof Error) || !cause.message.includes("open blocked or timed out")) throw cause;
      refusal = cause.message;
    }
  } finally { held.close(); }
  await deletion;
  const subsequent = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(name, 2);
    request.onblocked = () => reject(new Error("Timed-out open leaked its actual late native connection"));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    if (!blockedDeletionObserved || refusal === null) throw new Error("Actual blocked deletion or explicit timeout was not observed");
    return { blockedDeletionObserved, timeoutMilliseconds: 50, refusal, lateConnectionClosed: true, subsequentVersion: subsequent.version };
  } finally { subsequent.close(); }
}
