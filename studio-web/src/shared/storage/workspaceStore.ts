// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — transactional browser workspace storage

import type { RawCodec } from "../contracts/graph";
import { previewWorkspaceArchive } from "./workspaceArchive";
import type { WorkspaceArchivePreview } from "./workspaceArchive";

/** A saved archive plus its complete admission preview. Browser cache is not durable backup. */
export interface StoredWorkspace {
  /** Fully admitted, exact archive to export independently of this browser cache. */
  readonly preview: WorkspaceArchivePreview;
  /** Literal browser-local durability boundary. */
  readonly durability: "browser-cache-export-required";
}

/** IndexedDB owner; each save atomically records an immutable archive and project head. */
export interface WorkspaceStore {
  /** Save only after full preview; expected prior archive identity provides concurrency control. */
  save(json: string, expectedArchiveDigest: string | null, signal?: AbortSignal): Promise<StoredWorkspace>;
  /** Revalidate the stored exact archive and its head identity on reload. Missing cache returns null. */
  load(projectId: string): Promise<StoredWorkspace | null>;
  /** Read the last project selected by a committed save, or null after cache eviction. */
  selectedProject(): Promise<string | null>;
  /** Release this connection; a caller owns and closes it at disposal. */
  close(): void;
}

interface Head {
  readonly archiveDigest: string;
  readonly workspaceHash: string;
}
function stored(preview: WorkspaceArchivePreview): StoredWorkspace {
  return Object.freeze({ preview, durability: "browser-cache-export-required" });
}
function readHead(value: unknown): Head {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Saved workspace head corrupt; restore an exported copy");
  const record = value as Record<string, unknown>;
  if (Object.keys(record).length !== 2 || !Object.hasOwn(record, "archiveDigest") || !Object.hasOwn(record, "workspaceHash")) throw new Error("Saved workspace head fields corrupt; restore an exported copy");
  const archiveDigest = record["archiveDigest"];
  const workspaceHash = record["workspaceHash"];
  if (typeof archiveDigest !== "string" || archiveDigest.length !== 64 || !/^[0-9a-f]{64}$/.test(archiveDigest) || typeof workspaceHash !== "string" || workspaceHash.length !== 64 || !/^[0-9a-f]{64}$/.test(workspaceHash)) throw new Error("Saved workspace head identity corrupt; restore an exported copy");
  return { archiveDigest, workspaceHash };
}
function project(value: string): void {
  if (typeof value !== "string" || value.length !== 36 || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value)) throw new Error("canonical project UUID required");
}
function failure(transaction: IDBTransaction): Error | DOMException {
  return transaction.error ?? new Error("Workspace transaction interrupted; previous committed state retained");
}

/** Maximum allowed wait for a native cache open; this is a product timeout, not a browser deadline. */
export const workspaceOpenTimeoutMilliseconds = 10_000;

/** Open native IndexedDB without fallback; callers may lower the bounded opening timeout. */
export async function openWorkspaceStore(
  rawCodecs: ReadonlyMap<string, RawCodec> = new Map(),
  databaseName = "scpn-quantum-workspace-v1",
  openTimeoutMilliseconds = workspaceOpenTimeoutMilliseconds,
): Promise<WorkspaceStore> {
  if (typeof indexedDB === "undefined") throw new Error("IndexedDB unavailable; browser persistence is unsupported");
  if (!databaseName.trim() || databaseName.length > 256) throw new Error("bounded database name required");
  if (!Number.isSafeInteger(openTimeoutMilliseconds) || openTimeoutMilliseconds <= 0 || openTimeoutMilliseconds > workspaceOpenTimeoutMilliseconds) throw new Error("open timeout must be a positive integer within the product bound");
  const codecs = new Map(rawCodecs);
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(databaseName, 1);
    let refused = false;
    request.onupgradeneeded = () => {
      request.result.createObjectStore("archives");
      request.result.createObjectStore("heads");
    };
    const refuse = () => { refused = true; reject(new Error("Workspace database open blocked or timed out; close other connections before retrying")); };
    const timeout = setTimeout(refuse, openTimeoutMilliseconds);
    request.onblocked = refuse;
    request.onerror = () => { clearTimeout(timeout); reject(request.error); };
    request.onsuccess = () => {
      clearTimeout(timeout);
      if (refused) request.result.close();
      else resolve(request.result);
    };
  });
  db.onversionchange = () => db.close();
  async function load(projectId: string): Promise<StoredWorkspace | null> {
    project(projectId);
    const entry = await new Promise<{ head: Head; json: string } | null>((resolve, reject) => {
      const transaction = db.transaction(["heads", "archives"], "readonly");
      let result: { head: Head; json: string } | null = null;
      let error: Error | null = null;
      transaction.oncomplete = () => error ? reject(error) : resolve(result);
      transaction.onabort = () => reject(failure(transaction));
      const head = transaction.objectStore("heads").get(projectId);
      head.onsuccess = () => {
        if (head.result === undefined) return;
        let value: Head;
        try { value = readHead(head.result as unknown); }
        catch (cause: unknown) { error = cause as Error; return; }
        const archive = transaction.objectStore("archives").get(`${projectId}/${value.archiveDigest}`);
        archive.onsuccess = () => {
          if (typeof archive.result !== "string") error = new Error("Saved archive missing or corrupt; restore an exported copy");
          else result = { head: value, json: archive.result as string };
        };
      };
    });
    if (entry === null) return null;
    const preview = await previewWorkspaceArchive(entry.json, codecs);
    if (preview.projectId !== projectId || preview.archiveDigest !== entry.head.archiveDigest || preview.workspaceHash !== entry.head.workspaceHash) throw new Error("Saved archive identity mismatch; restore an exported copy");
    return stored(preview);
  }
  return Object.freeze({
    async save(json: string, expectedArchiveDigest: string | null, signal?: AbortSignal): Promise<StoredWorkspace> {
      if (signal?.aborted) throw new Error("Workspace save cancelled before transaction; prior state retained");
      // All digest and graph awaits precede the native readwrite transaction.
      const preview = await previewWorkspaceArchive(json, codecs);
      if (expectedArchiveDigest !== null && (expectedArchiveDigest.length !== 64 || !/^[0-9a-f]{64}$/.test(expectedArchiveDigest))) throw new Error("expected prior archive digest invalid");
      const validatedPrior = expectedArchiveDigest === null ? null : await load(preview.projectId);
      if (expectedArchiveDigest !== null && (validatedPrior === null || validatedPrior.preview.archiveDigest !== expectedArchiveDigest)) throw new Error("Workspace changed in another tab or was evicted; reload and explicitly reconcile before saving");
      if (signal?.aborted) throw new Error("Workspace save cancelled before transaction; prior state retained");
      await new Promise<void>((resolve, reject) => {
        const transaction = db.transaction(["archives", "heads"], "readwrite");
        const heads = transaction.objectStore("heads");
        const archives = transaction.objectStore("archives");
        let error: Error | null = null;
        const abort = () => {
          try { transaction.abort(); }
          catch {
            // Native abort refuses a committing/finished transaction; its terminal event owns the outcome.
          }
        };
        const cleanup = () => signal?.removeEventListener("abort", abort);
        transaction.oncomplete = () => { cleanup(); resolve(); };
        transaction.onabort = () => { cleanup(); reject(error ?? failure(transaction)); };
        signal?.addEventListener("abort", abort, { once: true });
        const current = heads.get(preview.projectId);
        current.onsuccess = () => {
          let prior: Head | undefined;
          try { prior = current.result === undefined ? undefined : readHead(current.result as unknown); }
          catch (cause: unknown) {
            error = cause as Error;
            transaction.abort();
            return;
          }
          if ((prior?.archiveDigest ?? null) !== expectedArchiveDigest) {
            error = new Error("Workspace changed in another tab or was evicted; reload and explicitly reconcile before saving");
            transaction.abort();
            return;
          }
          if (validatedPrior !== null && prior?.workspaceHash !== validatedPrior.preview.workspaceHash) {
            error = new Error("Saved workspace head changed or became corrupt; reload before saving");
            transaction.abort();
            return;
          }
          const write = () => {
            const key = `${preview.projectId}/${preview.archiveDigest}`;
            const existing = archives.get(key);
            existing.onsuccess = () => {
              if (existing.result !== undefined && existing.result !== preview.json) {
                error = new Error("Immutable archive identity conflict; previous state retained");
                transaction.abort();
                return;
              }
              if (existing.result === undefined) archives.add(preview.json, key);
              heads.put({ archiveDigest: preview.archiveDigest, workspaceHash: preview.workspaceHash }, preview.projectId);
              heads.put(preview.projectId, "$selected");
            };
          };
          if (validatedPrior === null) write();
          else {
            const priorArchive = archives.get(`${preview.projectId}/${validatedPrior.preview.archiveDigest}`);
            priorArchive.onsuccess = () => {
              if (priorArchive.result !== validatedPrior.preview.json) {
                error = new Error("Saved archive changed or became corrupt; previous head retained");
                transaction.abort();
                return;
              }
              write();
            };
          }
        };
      });
      return stored(preview);
    },
    load,
    async selectedProject(): Promise<string | null> {
      return new Promise<string | null>((resolve, reject) => {
        const transaction = db.transaction("heads", "readonly");
        let value: string | null = null;
        let error: Error | null = null;
        transaction.oncomplete = () => error ? reject(error) : resolve(value);
        transaction.onabort = () => reject(failure(transaction));
        const request = transaction.objectStore("heads").get("$selected");
        request.onsuccess = () => {
          if (request.result === undefined) return;
          try {
            if (typeof request.result !== "string") throw new Error("Saved project selection corrupt; restore an exported copy");
            project(request.result);
            value = request.result;
          } catch (cause: unknown) {
            error = cause as Error;
          }
        };
      });
    },
    close(): void { db.close(); },
  });
}
