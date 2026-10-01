// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workspace controller refusal tests

import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";
import { useWorkspace } from "./useWorkspace";
import { maxArchiveBytes, previewWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";

it("refuses a parameter child request when native persistence is unavailable and retains its admitted preview", async () => {
  const archive = await conformanceArchive(corpusText, false);
  const hook = renderHook(() => useWorkspace(conformanceCodecs));
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  expect(hook.result.current.storageAvailable).toBe(false);
  act(() => hook.result.current.edit(archive.json));
  await act(async () => { await hook.result.current.inspect(); });
  expect(hook.result.current.preview?.archiveDigest).toBe(archive.archiveDigest);
  await act(async () => {
    await expect(hook.result.current.saveRevision(archive, archive.json, new AbortController().signal)).rejects.toThrow("Browser persistence unavailable; export the archive");
  });
  expect(hook.result.current.saved).toBeNull();
  expect(hook.result.current.draft).toBe(archive.json);
  expect(hook.result.current.preview?.archiveDigest).toBe(archive.archiveDigest);
  expect(hook.result.current.busy).toBe(false);
  hook.unmount();
});

it("retains an edited draft through unavailable cache reload", async () => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  act(() => hook.result.current.edit("exact unsaved text"));
  await act(async () => { await hook.result.current.reload(); });
  expect(hook.result.current.draft).toBe("exact unsaved text");
  expect(hook.result.current.saved).toBeNull();
  expect(hook.result.current.message).toContain("Browser persistence unavailable");
  hook.unmount();
});

it("refuses an actual oversized File before replacing the current editor", async () => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  act(() => hook.result.current.edit("retained unsaved draft"));
  const file = new File([new Uint8Array(maxArchiveBytes + 1)], "oversized.json", { type: "application/json" });
  await act(async () => { await hook.result.current.read(file); });
  expect(hook.result.current.message).toContain("64 MiB import bound");
  expect(hook.result.current.draft).toBe("retained unsaved draft");
  expect(hook.result.current.preview).toBeNull();
  expect(hook.result.current.saved).toBeNull();
  expect(hook.result.current.busy).toBe(false);
  hook.unmount();
});

it("clears a refused preview across edits and keeps unsupported persistence explicit", async () => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  act(() => hook.result.current.edit("{}"));
  await act(async () => { await hook.result.current.inspect(); });
  expect(hook.result.current.preview).toBeNull();
  expect(hook.result.current.message).toContain("archive fields");
  act(() => hook.result.current.edit("another exact draft"));
  await act(async () => { await hook.result.current.save(); });
  expect(hook.result.current.draft).toBe("another exact draft");
  expect(hook.result.current.message).toContain("Browser persistence unavailable");
  hook.unmount();
});

it.each(["Portable recovery project", "x".repeat(512)])("creates and previews a portable draft while refusing unavailable browser persistence [%#]", async title => {
  const { result } = renderHook(() => useWorkspace());
  await waitFor(() => expect(result.current.busy).toBe(false));
  expect(result.current.storageAvailable).toBe(false);
  expect(result.current.message).toContain("IndexedDB unavailable");

  await act(async () => { await result.current.create(title); });
  const draft = result.current.draft;
  const original = await previewWorkspaceArchive(draft);
  expect(original.documentHashes).toEqual([]);
  expect(original.rawHashes).toEqual([]);
  expect(JSON.parse(draft).manifest.extensions.title).toBe(title);
  expect(result.current.saved).toBeNull();

  await act(async () => { await result.current.inspect(); });
  expect(result.current.preview?.archiveDigest).toBe(original.archiveDigest);
  expect(result.current.message).toContain("No existing project head will be replaced");

  await act(async () => { await result.current.save(); });
  expect(result.current.message).toContain("Browser persistence unavailable; export the preview");
  expect(result.current.saved).toBeNull();
  expect(result.current.draft).toBe(draft);
  expect(result.current.preview?.archiveDigest).toBe(original.archiveDigest);

  await act(async () => { await result.current.reload(); });
  expect(result.current.message).toBe("Browser persistence unavailable");
  expect(result.current.draft).toBe(draft);
  expect(result.current.preview?.archiveDigest).toBe(original.archiveDigest);

  act(() => { result.current.edit("\n" + draft); });
  expect(result.current.preview).toBeNull();
  expect(result.current.saved).toBeNull();
  await act(async () => { await result.current.inspect(); });
  expect(result.current.preview?.workspaceHash).toBe(original.workspaceHash);
  expect(result.current.preview?.archiveDigest).not.toBe(original.archiveDigest);
});

it.each(["", " \t\n", "x".repeat(513)])("retains the exact editor and preview when a new project title is invalid: %j", async title => {
  const { result } = renderHook(() => useWorkspace());
  await waitFor(() => expect(result.current.busy).toBe(false));
  await act(async () => { await result.current.create("Existing portable draft"); });
  await act(async () => { await result.current.inspect(); });
  const draft = result.current.draft;
  const digest = result.current.preview?.archiveDigest;
  expect(digest).toMatch(/^[a-f0-9]{64}$/);
  await act(async () => { await result.current.create(title); });
  expect(result.current.message).toBe("Project title must contain 1–512 characters");
  expect(result.current.draft).toBe(draft);
  expect(result.current.preview?.archiveDigest).toBe(digest);
  expect(result.current.saved).toBeNull();
  expect(result.current.busy).toBe(false);
});

it("keeps a newer editor value when project creation finishes after an edit", async () => {
  const { result } = renderHook(() => useWorkspace());
  await waitFor(() => expect(result.current.busy).toBe(false));
  await act(async () => {
    const creating = result.current.create("Superseded project");
    result.current.edit("Newer unsaved editor value");
    await creating;
  });
  expect(result.current.draft).toBe("Newer unsaved editor value");
  expect(result.current.preview).toBeNull();
  expect(result.current.saved).toBeNull();
  expect(result.current.message).toContain("Unsaved draft");
  expect(result.current.busy).toBe(false);
});

it("refuses to publish a preview whose archive was edited during validation", async () => {
  const { result } = renderHook(() => useWorkspace());
  await waitFor(() => expect(result.current.busy).toBe(false));
  await act(async () => { await result.current.create("Original project"); });
  const newer = "\n" + result.current.draft;
  await act(async () => {
    const inspecting = result.current.inspect();
    result.current.edit(newer);
    await inspecting;
  });
  expect(result.current.draft).toBe(newer);
  expect(result.current.preview).toBeNull();
  expect(result.current.saved).toBeNull();
  expect(result.current.message).toContain("Unsaved draft");
  expect(result.current.busy).toBe(false);
});


it.each(["\ud800", "\udfff"])("refuses a project title containing a non-scalar Unicode value [%#]", async title => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  act(() => hook.result.current.edit("Retained original draft"));
  await act(async () => { await hook.result.current.create(title); });
  expect(hook.result.current.message).toContain("invalid Unicode scalar");
  expect(hook.result.current.draft).toBe("Retained original draft");
  expect(hook.result.current.preview).toBeNull();
  expect(hook.result.current.saved).toBeNull();
  expect(hook.result.current.busy).toBe(false);
  hook.unmount();
});


it("keeps the draft when a hostile dynamically supplied title throws a non-Error value", async () => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  act(() => hook.result.current.edit("Exact retained draft"));
  const hostile = Object.create(null) as object;
  Object.defineProperty(hostile, "trim", { get() { throw "untrusted caller refused"; } });
  await act(async () => { await hook.result.current.create(hostile as unknown as string); });
  expect(hook.result.current.message).toBe("Workspace operation refused");
  expect(hook.result.current.draft).toBe("Exact retained draft");
  expect(hook.result.current.saved).toBeNull();
  expect(hook.result.current.preview).toBeNull();
  hook.unmount();
});

it("does not replace a newer editor message when an earlier rejected operation settles", async () => {
  const hook = renderHook(() => useWorkspace());
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  await act(async () => {
    const rejected = hook.result.current.create(" ");
    hook.result.current.edit("Newer exact draft");
    await rejected;
  });
  expect(hook.result.current.draft).toBe("Newer exact draft");
  expect(hook.result.current.message).toContain("Unsaved draft");
  expect(hook.result.current.preview).toBeNull();
  expect(hook.result.current.saved).toBeNull();
  hook.unmount();
});
