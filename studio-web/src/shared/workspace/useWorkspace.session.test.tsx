// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — a page without IndexedDB saves to its own session

import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import { useWorkspace } from "./useWorkspace";

it("saves, revises and restores within the session when the page has no IndexedDB", async () => {
  // This test environment has no IndexedDB, as an opaque-origin embedding frame would not.
  expect(typeof indexedDB).toBe("undefined");
  const archive = await conformanceArchive(corpusText, false);
  const hook = renderHook(() => useWorkspace(conformanceCodecs));
  await waitFor(() => expect(hook.result.current.busy).toBe(false));
  expect(hook.result.current.storageAvailable).toBe(true);
  expect(hook.result.current.persistence).toBe("session");
  expect(hook.result.current.message).toContain("kept for this page session only");
  expect(hook.result.current.saved).toBeNull();

  act(() => hook.result.current.edit(archive.json));
  await act(async () => {
    await hook.result.current.inspect();
  });
  await act(async () => {
    await hook.result.current.save();
  });
  expect(hook.result.current.saved?.preview.archiveDigest).toBe(archive.archiveDigest);
  expect(hook.result.current.saved?.durability).toBe("session-only-export-required");

  await act(async () => {
    await hook.result.current.saveRevision(archive, archive.json, new AbortController().signal);
  });
  expect(hook.result.current.saved?.preview.archiveDigest).toBe(archive.archiveDigest);

  act(() => hook.result.current.edit("unsaved edit"));
  await act(async () => {
    await hook.result.current.reload();
  });
  expect(hook.result.current.draft).toBe(archive.json);
  expect(hook.result.current.saved?.durability).toBe("session-only-export-required");
  hook.unmount();
});
