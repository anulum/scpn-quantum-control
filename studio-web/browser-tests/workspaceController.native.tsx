// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual controller file and cache recovery

import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import { useWorkspace } from "../src/features/workspace/useWorkspace";
import type { WorkspaceController } from "../src/features/workspace/useWorkspace";
import { conformanceCodecs } from "./workspaceFixture";

async function tick(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0));
}

/** Unmount during native connection and restore completion; require each real connection to close. */
export async function runNativeControllerUnmountCases(): Promise<Record<string, unknown>> {
  const receipts: Record<string, unknown> = {};
  for (const phase of ["open", "restore"] as const) {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const nativeClose = IDBDatabase.prototype.close;
    const nativeTransaction = IDBDatabase.prototype.transaction;
    let unmounted = false;
    let closedConnections = 0;
    let completedReads = 0;
    const unmount = () => {
      if (!unmounted) {
        unmounted = true;
        flushSync(() => root.unmount());
      }
    };
    function Host() {
      useWorkspace(conformanceCodecs);
      return null;
    }
    try {
      IDBDatabase.prototype.close = function(this: IDBDatabase): void {
        if (this.name === "scpn-quantum-workspace-v1") closedConnections++;
        nativeClose.call(this);
      };
      if (phase === "restore") {
        IDBDatabase.prototype.transaction = function(this: IDBDatabase, names: string | string[], mode?: IDBTransactionMode, options?: IDBTransactionOptions): IDBTransaction {
          const transaction = nativeTransaction.call(this, names, mode, options);
          if (this.name === "scpn-quantum-workspace-v1" && mode === "readonly") {
            transaction.addEventListener("complete", () => {
              completedReads++;
              unmount();
            }, { once: true });
          }
          return transaction;
        };
      }
      flushSync(() => root.render(<Host />));
      if (phase === "open") unmount();
      const deadline = performance.now() + 10_000;
      while (closedConnections === 0) {
        if (performance.now() > deadline) throw new Error(`Unmount during ${phase} leaked its actual database connection`);
        await tick();
      }
      await tick();
      if (!unmounted || closedConnections !== 1 || completedReads !== (phase === "restore" ? 1 : 0)) throw new Error(`Unmount during ${phase} changed the native lifecycle`);
      receipts[phase] = { unmounted, closedConnections, completedReads };
    } finally {
      IDBDatabase.prototype.close = nativeClose;
      IDBDatabase.prototype.transaction = nativeTransaction;
      unmount();
      container.remove();
    }
  }
  return receipts;
}

/** Exercise the public React controller against native File and IndexedDB, including superseded operations. */
export async function runNativeControllerCases(): Promise<Record<string, unknown>> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  const state: { current: WorkspaceController | null } = { current: null };
  function Host() {
    state.current = useWorkspace(conformanceCodecs);
    return null;
  }
  const read = (): WorkspaceController => {
    if (state.current === null) throw new Error("Actual controller was not mounted");
    return state.current;
  };
  const perform = async (action: () => Promise<void>): Promise<void> => {
    await flushSync(action);
    await tick();
  };
  try {
    flushSync(() => root.render(<Host />));
    const deadline = performance.now() + 10_000;
    while (read().busy) {
      if (performance.now() > deadline) throw new Error("Native controller connection did not settle");
      await tick();
    }
    if (!read().storageAvailable) throw new Error(`Native persistence unavailable: ${read().message}`);
    await perform(() => read().create("Native controller recovery"));
    const original = read().draft;
    await perform(() => read().save());
    if (!read().message.includes("complete preview is required") || read().draft !== original) throw new Error("Preview-less save changed or admitted a draft");
    await perform(() => read().inspect());
    const preview = read().preview;
    if (preview === null) throw new Error("Actual empty project preview missing");
    await perform(() => read().save());
    if (read().saved?.preview.json !== original) throw new Error("Native controller did not commit exact draft");
    const savedDigest = read().saved?.preview.archiveDigest;
    await perform(() => read().read(new File([new Uint8Array([0xff])], "invalid-utf8.json")));
    const invalidUtf8 = read().message;
    if (!invalidUtf8.includes("valid UTF-8") || read().draft !== original || read().saved?.preview.archiveDigest !== savedDigest) throw new Error("Invalid native File replaced draft or saved archive");

    const formatted = "\n" + original;
    await perform(() => read().read(new File([formatted], "portable.json")));
    if (read().draft !== formatted || read().preview !== null || read().saved?.preview.archiveDigest !== savedDigest) throw new Error("Native File read rewrote saved state or changed exact source");

    await perform(async () => {
      const pending = read().read(new File([original], "superseded.json"));
      read().edit("Newer editor after file selection");
      await pending;
    });
    if (read().draft !== "Newer editor after file selection" || read().preview !== null || read().saved?.preview.archiveDigest !== savedDigest) throw new Error("Superseded native File read replaced newer editor or saved archive");

    await perform(async () => {
      const pending = read().reload();
      read().edit("Newer editor during reload");
      await pending;
    });
    if (read().draft !== "Newer editor during reload" || read().saved?.preview.archiveDigest !== savedDigest) throw new Error("Superseded native reload overwrote newer editor");

    flushSync(() => read().edit(formatted));
    await perform(() => read().inspect());
    const completedPreview = read().preview;
    if (completedPreview === null) throw new Error("Completion-race preview missing");
    const nativeTransaction = IDBDatabase.prototype.transaction;
    let completedTransactions = 0;
    try {
      IDBDatabase.prototype.transaction = function(this: IDBDatabase, names: string | string[], mode?: IDBTransactionMode, options?: IDBTransactionOptions): IDBTransaction {
        const transaction = nativeTransaction.call(this, names, mode, options);
        if (this.name === "scpn-quantum-workspace-v1" && mode === "readwrite") {
          transaction.addEventListener("complete", () => {
            completedTransactions++;
            flushSync(() => read().edit("Newer editor at native commit completion"));
          }, { once: true });
        }
        return transaction;
      };
      await perform(() => read().save());
    } finally { IDBDatabase.prototype.transaction = nativeTransaction; }
    if (completedTransactions !== 1 || read().draft !== "Newer editor at native commit completion" || read().preview !== null) throw new Error("Completed native save overwrote the newer editor");
    await perform(() => read().reload());
    if (read().draft !== formatted || read().saved?.preview.archiveDigest !== completedPreview.archiveDigest) throw new Error("Save completion race did not retain the actually committed archive");
    flushSync(() => read().edit("Retained editor after cache eviction"));
    const completedDigest = read().saved?.preview.archiveDigest;

    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("scpn-quantum-workspace-v1", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    try {
      await new Promise<void>((resolve, reject) => {
        const transaction = database.transaction("heads", "readwrite");
        transaction.objectStore("heads").delete(preview.projectId);
        transaction.objectStore("heads").delete("$selected");
        transaction.oncomplete = () => resolve();
        transaction.onabort = () => reject(transaction.error);
      });
    } finally { database.close(); }
    await perform(() => read().reload());
    const missingCache = read().message;
    if (!missingCache.includes("missing or evicted") || read().draft !== "Retained editor after cache eviction" || read().saved?.preview.archiveDigest !== completedDigest) throw new Error("Evicted cache reload rewrote current editor or invented a new saved revision");
    const unmountRecovery = await runNativeControllerUnmountCases();
    return { projectId: preview.projectId, savedDigest, invalidUtf8, missingCache, unmountRecovery,
      nativeFileReadExact: true, supersededFileRetainedEditor: true, supersededReloadRetainedEditor: true,
      completedTransactions, completedDigest,
      boundary: "Real public controller and native File/IndexedDB in isolated browser context; no execution or producer qualification" };
  } finally {
    flushSync(() => root.unmount());
    container.remove();
  }
}


/** Observe real unsupported cache opening both while mounted and after immediate disposal. */
export async function runNativeControllerFailedOpenCases(): Promise<Record<string, unknown>> {
  const upgraded = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("scpn-quantum-workspace-v1", 2);
    request.onblocked = () => reject(new Error("Original native workspace connections did not close for version change"));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  const observations: Record<string, unknown> = {};
  try {
    for (const phase of ["mounted", "disposed"] as const) {
      const container = document.createElement("div");
      document.body.append(container);
      const root = createRoot(container);
      const state: { current: WorkspaceController | null } = { current: null };
      function Host() { state.current = useWorkspace(conformanceCodecs); return null; }
      let disposed = false;
      try {
        flushSync(() => root.render(<Host />));
        if (phase === "disposed") { flushSync(() => root.unmount()); disposed = true; }
        const deadline = performance.now() + 10_000;
        while (phase === "mounted" && state.current?.busy !== false) {
          if (performance.now() > deadline) throw new Error("Native unsupported cache open did not settle");
          await tick();
        }
        await new Promise<void>(resolve => setTimeout(resolve, 20));
        const current = state.current;
        if (current === null || current.storageAvailable || current.saved !== null || current.preview !== null) throw new Error("Unsupported cache opening fabricated workspace state");
        if (phase === "mounted" && !current.message.includes("version")) throw new Error(`Actual native version refusal omitted its diagnosis: ${current.message}`);
        observations[phase] = { disposed, storageAvailable: current.storageAvailable, saved: current.saved, message: current.message };
      } finally {
        if (!disposed) flushSync(() => root.unmount());
        container.remove();
      }
    }
    return observations;
  } finally { upgraded.close(); }
}
