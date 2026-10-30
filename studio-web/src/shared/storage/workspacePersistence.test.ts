// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser storage first, session storage only when the page has none

// @vitest-environment node
import { describe, expect, it } from "vitest";
import { isBrowserStorageUnavailable, openWorkspacePersistence } from "./workspacePersistence";
import type { WorkspaceStore } from "./workspaceStore";

const browserStore: WorkspaceStore = {
  save: async () => {
    throw new Error("unused");
  },
  load: async () => null,
  selectedProject: async () => null,
  close: () => undefined,
};

describe("workspace persistence", () => {
  it("uses the browser store whenever it opens", async () => {
    const opened = await openWorkspacePersistence(new Map(), async () => browserStore);
    expect(opened.store).toBe(browserStore);
    expect(opened.persistence).toBe("browser");
    expect(opened.emptyMessage).toContain("Browser cache may have been evicted");
  });

  it("falls back to session storage when the page has no IndexedDB", async () => {
    for (const cause of [
      new Error("IndexedDB unavailable; browser persistence is unsupported"),
      new DOMException("The operation is insecure.", "SecurityError"),
    ]) {
      const opened = await openWorkspacePersistence(new Map(), async () => {
        throw cause;
      });
      expect(opened.persistence).toBe("session");
      expect(opened.emptyMessage).toContain("kept for this page session only");
      expect(await opened.store.selectedProject()).toBeNull();
    }
  });

  it("keeps refusing a blocked, timed-out or failing browser store", async () => {
    for (const cause of [
      new Error(
        "Workspace database open blocked or timed out; close other connections before retrying",
      ),
      new DOMException("Version change", "VersionError"),
      new DOMException("Quota", "QuotaExceededError"),
      "not an error",
    ]) {
      await expect(
        openWorkspacePersistence(new Map(), async () => {
          throw cause;
        }),
      ).rejects.toBe(cause);
      expect(isBrowserStorageUnavailable(cause)).toBe(false);
    }
  });

  it("uses the real browser store by default and falls back in a page without IndexedDB", async () => {
    // This test environment has no IndexedDB, as an opaque-origin frame would not.
    expect(typeof indexedDB).toBe("undefined");
    const opened = await openWorkspacePersistence(new Map());
    expect(opened.persistence).toBe("session");
  });
});
