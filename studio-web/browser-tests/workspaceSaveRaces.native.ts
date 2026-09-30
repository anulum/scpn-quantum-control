// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual native mutation between admission and save

import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

/** Inject actual committed cache changes after the native prior snapshot and require atomic refusal. */
export async function runNativeSaveRaces(corpusText: string): Promise<Record<string, unknown>> {
  const initial = await conformanceArchive(corpusText, false);
  const edited = await conformanceArchive(corpusText, true);
  const results: Record<string, unknown> = {};
  const nativeTransaction = IDBDatabase.prototype.transaction;
  for (const kind of ["head", "head_fields", "head_revision", "archive", "immutable_archive"] as const) {
    const databaseName = `workspace-save-race-${crypto.randomUUID()}`;
    const store = await openWorkspaceStore(conformanceCodecs, databaseName);
    const faultState: { injected: boolean; mutation: Promise<void> | null } = { injected: false, mutation: null };
    try {
      await store.save(initial.json, null);
      IDBDatabase.prototype.transaction = function(this: IDBDatabase, names: string | string[], mode?: IDBTransactionMode, options?: IDBTransactionOptions): IDBTransaction {
        const transaction = nativeTransaction.call(this, names, mode, options);
        if (this.name === databaseName && mode === "readonly" && !faultState.injected) {
          transaction.addEventListener("complete", () => {
            faultState.injected = true;
            faultState.mutation = new Promise<void>((resolve, reject) => {
              const fault = nativeTransaction.call(this, ["heads", "archives"], "readwrite");
              fault.oncomplete = () => resolve();
              fault.onabort = () => reject(fault.error);
              if (kind === "head") fault.objectStore("heads").put({ archiveDigest: initial.archiveDigest, workspaceHash: "0".repeat(64) }, initial.projectId);
              else if (kind === "head_fields") fault.objectStore("heads").put({ archiveDigest: initial.archiveDigest, workspaceHash: initial.workspaceHash, unexpected: true }, initial.projectId);
              else if (kind === "head_revision") {
                fault.objectStore("archives").put(edited.json, `${initial.projectId}/${edited.archiveDigest}`);
                fault.objectStore("heads").put({ archiveDigest: edited.archiveDigest, workspaceHash: edited.workspaceHash }, initial.projectId);
              }
              else if (kind === "immutable_archive") fault.objectStore("archives").put("conflicting exact archive", `${initial.projectId}/${edited.archiveDigest}`);
              else fault.objectStore("archives").put(initial.json + " ", `${initial.projectId}/${initial.archiveDigest}`);
            });
          }, { once: true });
        }
        return transaction;
      };
      let refusal: string | null = null;
      try { await store.save(edited.json, initial.archiveDigest); }
      catch (cause: unknown) {
        if (!(cause instanceof Error)) throw cause;
        refusal = cause.message;
      } finally { IDBDatabase.prototype.transaction = nativeTransaction; }
      if (!faultState.injected || faultState.mutation === null) throw new Error("Race never reached the actual native prior snapshot");
      await faultState.mutation;
      const required = {
        head: "head changed or became corrupt",
        head_fields: "head fields corrupt",
        head_revision: "changed in another tab",
        archive: "archive changed or became corrupt",
        immutable_archive: "Immutable archive identity conflict",
      }[kind];
      if (refusal === null || !refusal.includes(required)) throw new Error(`Actual ${kind} change was not refused before writes: ${refusal}`);
      if (kind === "head_revision") {
        const retained = await store.load(initial.projectId);
        if (retained?.preview.json !== edited.json || retained.preview.archiveDigest !== edited.archiveDigest) throw new Error("Stale save replaced the concurrently committed revision");
      } else if (kind === "immutable_archive") {
        const retained = await store.load(initial.projectId);
        if (retained?.preview.json !== initial.json || retained.preview.archiveDigest !== initial.archiveDigest) throw new Error("Immutable collision replaced the prior saved state");
        const conflicting = await new Promise<unknown>((resolve, reject) => {
          const request = indexedDB.open(databaseName, 1);
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction("archives", "readonly");
            const archive = transaction.objectStore("archives").get(`${initial.projectId}/${edited.archiveDigest}`);
            transaction.oncomplete = () => { db.close(); resolve(archive.result as unknown); };
            transaction.onabort = () => { db.close(); reject(transaction.error); };
          };
        });
        if (conflicting !== "conflicting exact archive") throw new Error("Immutable collision overwrote the existing native archive");
      } else {
        let corruptStateRetained = false;
        try { await store.load(initial.projectId); }
        catch (cause: unknown) { corruptStateRetained = cause instanceof Error && cause.message.includes(kind === "head_fields" ? "head fields corrupt" : "identity mismatch"); }
        if (!corruptStateRetained) throw new Error("Refused save replaced externally corrupted state");
      }
      if (await store.selectedProject() !== initial.projectId) throw new Error("Refused save changed project selection");
      results[kind] = { databaseName, actualMutationCommitted: faultState.injected, refusal, priorArchiveDigest: initial.archiveDigest };
    } finally { IDBDatabase.prototype.transaction = nativeTransaction; store.close(); }
  }
  return results;
}
