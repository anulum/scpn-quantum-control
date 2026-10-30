// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — session workspace store keeps the browser store's rules

// @vitest-environment node
import { describe, expect, it } from "vitest";
import { parseWorkspaceManifest } from "../contracts";
import { createSessionWorkspaceStore } from "./sessionWorkspaceStore";
import { createWorkspaceArchive } from "./workspaceArchive";

const projectId = "0f8fad5b-d9cb-469f-a165-70867728950e";

async function archive(title: string, id = projectId): Promise<string> {
  const parsed = parseWorkspaceManifest({
    schema: "quantum_workspace.v1",
    body: {
      project_id: id,
      revision_refs: [],
      draft_ref: null,
      created_at: "2026-10-30T00:00:00.000Z",
      updated_at: "2026-10-30T00:00:00.000Z",
      artefact_refs: [],
    },
    extensions: { title },
  });
  if (!parsed.ok) throw new Error(parsed.message);
  return (await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map())).json;
}

describe("session workspace store", () => {
  it("starts empty, saves a new project and restores its exact archive as session-only", async () => {
    const store = createSessionWorkspaceStore();
    expect(await store.selectedProject()).toBeNull();
    expect(await store.load(projectId)).toBeNull();
    const json = await archive("first");
    const saved = await store.save(json, null);
    expect(saved.durability).toBe("session-only-export-required");
    expect(saved.preview.json).toBe(json);
    expect(await store.selectedProject()).toBe(projectId);
    const loaded = await store.load(projectId);
    expect(loaded?.preview.json).toBe(json);
    expect(loaded?.preview.archiveDigest).toBe(saved.preview.archiveDigest);
    expect(loaded?.durability).toBe("session-only-export-required");
  });

  it("requires the current head as the expected prior archive, like the browser store", async () => {
    const store = createSessionWorkspaceStore();
    const first = await store.save(await archive("first"), null);
    const second = await archive("second");
    await expect(store.save(second, null)).rejects.toThrow("Workspace changed in another tab");
    await expect(store.save(second, "0".repeat(64))).rejects.toThrow(
      "Workspace changed in another tab",
    );
    await expect(store.save(second, "not-a-digest")).rejects.toThrow(
      "expected prior archive digest invalid",
    );
    const updated = await store.save(second, first.preview.archiveDigest);
    expect((await store.load(projectId))?.preview.json).toBe(updated.preview.json);
    // Saving the same bytes again under the same head is idempotent.
    await store.save(second, updated.preview.archiveDigest);
    expect((await store.load(projectId))?.preview.archiveDigest).toBe(
      updated.preview.archiveDigest,
    );
  });

  it("gives other bytes another identity and keeps every saved archive exact", async () => {
    const store = createSessionWorkspaceStore();
    const json = await archive("first");
    const saved = await store.save(json, null);
    const respaced = await store.save(`${json}\n`, saved.preview.archiveDigest);
    expect(respaced.preview.archiveDigest).not.toBe(saved.preview.archiveDigest);
    expect((await store.load(projectId))?.preview.json).toBe(`${json}\n`);
    // Returning to the first bytes restores the first identity and its exact text.
    const back = await store.save(json, respaced.preview.archiveDigest);
    expect(back.preview.archiveDigest).toBe(saved.preview.archiveDigest);
    expect((await store.load(projectId))?.preview.json).toBe(json);
  });

  it("refuses an invalid archive, a cancelled save, an invalid project and a closed store", async () => {
    const store = createSessionWorkspaceStore();
    await expect(store.save("{}", null)).rejects.toThrow();
    const cancelled = new AbortController();
    cancelled.abort();
    await expect(store.save(await archive("first"), null, cancelled.signal)).rejects.toThrow(
      "cancelled before transaction",
    );
    expect(await store.selectedProject()).toBeNull();
    await expect(store.load("not-a-project")).rejects.toThrow("canonical project UUID required");
    store.close();
    await expect(store.selectedProject()).rejects.toThrow("session store closed");
    await expect(store.load(projectId)).rejects.toThrow("session store closed");
    await expect(store.save(await archive("first"), null)).rejects.toThrow("session store closed");
  });

  it("refuses a save cancelled while its archive is admitted", async () => {
    const store = createSessionWorkspaceStore();
    const cancelled = new AbortController();
    const saving = store.save(await archive("first"), null, cancelled.signal);
    cancelled.abort();
    await expect(saving).rejects.toThrow("cancelled before transaction");
    expect(await store.load(projectId)).toBeNull();
  });
});
