// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native cache corruption and import recovery

import { readJson, writeJson } from "../src/shared/contracts";
import { maxArchiveBytes } from "../src/shared/storage/workspaceArchive";
import { openWorkspaceStore } from "../src/shared/storage/workspaceStore";
import { conformanceArchive, conformanceCodecs } from "./workspaceFixture";

async function refuse(operation: () => Promise<unknown>, message: string): Promise<string> {
  try { await operation(); }
  catch (cause: unknown) {
    if (!(cause instanceof Error) || !cause.message.includes(message)) throw cause;
    return cause.message;
  }
  throw new Error(`Expected native public API refusal: ${message}`);
}

async function mutateCache(databaseName: string, mutation: (transaction: IDBTransaction) => void): Promise<void> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(["archives", "heads"], "readwrite");
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(transaction.error);
      mutation(transaction);
    });
  } finally { database.close(); }
}

/** Exercise actual cache corruption, eviction and rejected imports without replacing native behavior. */
export async function runNativeRecoveryCases(corpusText: string): Promise<Record<string, unknown>> {
  const initial = await conformanceArchive(corpusText, false);
  const edited = await conformanceArchive(corpusText, true);
  const databaseName = `workspace-native-recovery-${crypto.randomUUID()}`;
  const store = await openWorkspaceStore(conformanceCodecs, databaseName);
  const failures: Record<string, string> = {};
  try {
    await store.save(initial.json, null);
    for (const selection of [42, "not-a-project-id"] as const) {
      await mutateCache(databaseName, transaction => transaction.objectStore("heads").put(selection, "$selected"));
      failures[`corrupt_selection_${typeof selection}`] = await refuse(() => store.selectedProject(), typeof selection === "number" ? "selection corrupt" : "canonical project UUID");
      if ((await store.load(initial.projectId))?.preview.json !== initial.json) throw new Error("Corrupt selection changed the saved archive");
    }
    await mutateCache(databaseName, transaction => transaction.objectStore("heads").put(initial.projectId, "$selected"));
    if (await store.selectedProject() !== initial.projectId) throw new Error("Explicit original selection restore failed");
    const pathEscape = readJson(initial.json) as { members: { name: string }[] };
    const first = pathEscape.members[0];
    if (first === undefined) throw new Error("Original recovery corpus has no members");
    first.name = "../outside.json";
    const future = readJson(initial.json) as { schema: string };
    future.schema = "quantum_workspace_archive.v2";
    const corrupt = readJson(initial.json) as { members: { kind: string; sha256: string }[] };
    const raw = corrupt.members.find(member => member.kind === "raw");
    if (raw === undefined) throw new Error("Original recovery corpus has no evidence");
    raw.sha256 = "0".repeat(64);
    for (const [name, json, message] of [
      ["path_escape", writeJson(pathEscape), "unsafe member name"],
      ["future_major", writeJson(future), "Unsupported archive schema"],
      ["corrupt_identity", writeJson(corrupt), "raw producer identity mismatch"],
      ["oversized", " ".repeat(maxArchiveBytes + 1), "encoded archive limit exceeded"],
    ] as const) {
      failures[name] = await refuse(() => store.save(json, initial.archiveDigest), message);
      const retained = await store.load(initial.projectId);
      if (retained?.preview.json !== initial.json || await store.selectedProject() !== initial.projectId) {
        throw new Error(`Rejected ${name} changed saved state`);
      }
    }
    const key = `${initial.projectId}/${initial.archiveDigest}`;
    for (const [name, head, message] of [
      ["null_head", null, "head corrupt"],
      ["array_head", [], "head corrupt"],
      ["scalar_head", 42, "head corrupt"],
      ["nonstring_archive_digest", { archiveDigest: 42, workspaceHash: initial.workspaceHash }, "head identity corrupt"],
      ["short_archive_digest", { archiveDigest: "a".repeat(63), workspaceHash: initial.workspaceHash }, "head identity corrupt"],
      ["uppercase_archive_digest", { archiveDigest: "A".repeat(64), workspaceHash: initial.workspaceHash }, "head identity corrupt"],
      ["nonstring_workspace_digest", { archiveDigest: initial.archiveDigest, workspaceHash: null }, "head identity corrupt"],
      ["short_workspace_digest", { archiveDigest: initial.archiveDigest, workspaceHash: "a".repeat(63) }, "head identity corrupt"],
      ["uppercase_workspace_digest", { archiveDigest: initial.archiveDigest, workspaceHash: "A".repeat(64) }, "head identity corrupt"],
    ] as const) {
      await mutateCache(databaseName, transaction => transaction.objectStore("heads").put(head, initial.projectId));
      failures[`${name}_reload`] = await refuse(() => store.load(initial.projectId), message);
      failures[`${name}_save`] = await refuse(() => store.save(edited.json, null), message);
      await mutateCache(databaseName, transaction => transaction.objectStore("heads").put({ archiveDigest: initial.archiveDigest, workspaceHash: initial.workspaceHash }, initial.projectId));
      if ((await store.load(initial.projectId))?.preview.json !== initial.json || await store.selectedProject() !== initial.projectId) throw new Error(`Corrupt ${name} refusal changed the original archive or selection`);
    }
    for (const digest of ["", "a".repeat(63), "A".repeat(64)]) {
      failures[`invalid_expected_${digest.length}`] = await refuse(() => store.save(edited.json, digest), "expected prior archive digest invalid");
      if ((await store.load(initial.projectId))?.preview.json !== initial.json) throw new Error("Invalid expected digest changed the native saved archive");
    }
    await mutateCache(databaseName, transaction => transaction.objectStore("archives").delete(key));
    failures["missing_archive"] = await refuse(() => store.load(initial.projectId), "missing or corrupt");
    failures["missing_prior_save"] = await refuse(() => store.save(edited.json, initial.archiveDigest), "missing or corrupt");
    await refuse(() => store.load(initial.projectId), "missing or corrupt");
    await mutateCache(databaseName, transaction => transaction.objectStore("archives").put(initial.json, key));
    await mutateCache(databaseName, transaction => transaction.objectStore("heads").put({ archiveDigest: initial.archiveDigest, workspaceHash: "0".repeat(64) }, initial.projectId));
    failures["mismatched_prior_save"] = await refuse(() => store.save(edited.json, initial.archiveDigest), "identity mismatch");
    await refuse(() => store.load(initial.projectId), "identity mismatch");
    await mutateCache(databaseName, transaction => transaction.objectStore("heads").put({ archiveDigest: initial.archiveDigest }, initial.projectId));
    failures["corrupt_head_reload"] = await refuse(() => store.load(initial.projectId), "head fields corrupt");
    failures["corrupt_head_save"] = await refuse(() => store.save(initial.json, null), "head fields corrupt");
    await mutateCache(databaseName, transaction => transaction.objectStore("heads").put({ archiveDigest: initial.archiveDigest, workspaceHash: initial.workspaceHash }, initial.projectId));
    if ((await store.load(initial.projectId))?.preview.json !== initial.json) throw new Error("Head corruption changed immutable archive");
    await mutateCache(databaseName, transaction => {
      transaction.objectStore("heads").clear();
      transaction.objectStore("archives").clear();
    });
    if (await store.load(initial.projectId) !== null || await store.selectedProject() !== null) throw new Error("Evicted cache invented saved state");
    failures["evicted_stale_save"] = await refuse(() => store.save(initial.json, initial.archiveDigest), "evicted");
    if (await store.load(initial.projectId) !== null || await store.selectedProject() !== null) throw new Error("Stale save rewrote evicted cache");
    await store.save(initial.json, null);
    const restored = await store.load(initial.projectId);
    if (restored?.preview.json !== initial.json || restored.preview.rawHashes.join(",") !== initial.rawHashes.join(",")) throw new Error("Explicit exported-copy restore changed original evidence");
    return { databaseName, failures, archiveDigest: restored.preview.archiveDigest, rawHashes: restored.preview.rawHashes, boundary: "Actual native cache faults and original synthetic metadata; no hardware claim" };
  } finally { store.close(); }
}
