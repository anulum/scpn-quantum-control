// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — parameter workspace production integration tests

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import { ParameterWorkspace } from "./ParameterWorkspace";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { parseWorkspaceManifest, readJson, writeJson } from "../../shared/contracts";
import { archiveSnapshotDomain, createWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import { appendParameterRevision, parameterSourceFromArchive } from "./parameterRevision";
import { createParameterDraft, parameterDraftReducer } from "./parameterDraft";

afterEach(cleanup);

it("builds a completely admitted child and passes exact prior identity to the original save owner", async () => {
  const original = await conformanceArchive(corpusText, false);
  const saved: WorkspaceArchivePreview[] = [];
  render(<ParameterWorkspace preview={original} rawCodecs={conformanceCodecs} saveArchive={async (archive, priorJson, signal) => {
    expect(priorJson).toBe(original.json);
    expect(signal.aborted).toBe(false);
    saved.push(archive);
  }} />);
  await waitFor(() => expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-0"));
  fireEvent.change(screen.getByLabelText("Selected value"), { target: { value: "-3" } });
  fireEvent.click(screen.getByRole("button", { name: "Apply value" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
  await waitFor(() => expect(screen.getByText(/Parameter revision saved/)).toBeTruthy());
  expect(saved).toHaveLength(1);
  expect(saved[0]!.documentHashes).toHaveLength(original.documentHashes.length + 1);
  expect(saved[0]!.rawHashes).toEqual(original.rawHashes);
});

it("re-admits unsupported archive versions before showing controls or invoking storage", async () => {
  const original = await conformanceArchive(corpusText, false);
  const future = readJson(original.json) as Record<string, unknown>;
  future["schema"] = "quantum_workspace_archive.v2";
  let saves = 0;
  render(<ParameterWorkspace preview={{ ...original, json: writeJson(future) }} rawCodecs={conformanceCodecs} saveArchive={async () => { saves++; }} />);
  await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("source unavailable"));
  expect(screen.queryByLabelText("Selected value")).toBeNull();
  expect(saves).toBe(0);
});

it("retains an empty project as empty and never synthesises a runnable experiment", async () => {
  const parsed = parseWorkspaceManifest({ schema: "quantum_workspace.v1", body: {
    project_id: "00000000-0000-4000-8000-000000000001", revision_refs: [], draft_ref: null,
    created_at: "2026-09-29T00:00:00Z", updated_at: "2026-09-29T00:00:00Z", artefact_refs: [],
  }, extensions: {} });
  if (!parsed.ok) throw new Error(parsed.message);
  const empty = await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map());
  render(<ParameterWorkspace preview={empty} rawCodecs={new Map()} saveArchive={async () => { throw new Error("No revision should reach storage"); }} />);
  await waitFor(() => expect(screen.getByText(/No experiment revision to edit/)).toBeTruthy());
  expect(screen.queryByRole("button", { name: "Save parameter revision" })).toBeNull();
});

it("hides the prior editor immediately when the trusted host verifier registry changes", async () => {
  const original = await conformanceArchive(corpusText, false);
  let saves = 0;
  const saveArchive = async () => { saves++; };
  const view = render(<ParameterWorkspace preview={original} rawCodecs={conformanceCodecs} saveArchive={saveArchive} />);
  await waitFor(() => expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-0"));
  view.rerender(<ParameterWorkspace preview={original} rawCodecs={new Map()} saveArchive={saveArchive} />);
  expect(screen.queryByLabelText("Selected value")).toBeNull();
  await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("source unavailable"));
  expect(saves).toBe(0);
});

it.each(["resolved", "refused"] as const)("retains the new source after an older producer verification is %s", async outcome => {
  const original = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(original.json, conformanceCodecs);
  if (!source) throw new Error("Missing original source");
  const state = parameterDraftReducer(createParameterDraft(source), { type: "value", key: "theta", index: 0, text: "-3", unit: "rad" });
  const child = await appendParameterRevision(original.json, source, state.snapshot, conformanceCodecs, "2026-10-01T12:00:00Z");
  const gate: { resolve: (() => void) | null; reject: ((error: Error) => void) | null } = { resolve: null, reject: null };
  const pending = new Promise<void>((resolve, reject) => { gate.resolve = resolve; gate.reject = reject; });
  const slowCodecs = new Map([...conformanceCodecs].map(([schema, verify]) => [schema, async (bytes: Uint8Array) => {
    const identity = await verify(bytes);
    await pending;
    return identity;
  }] as const));
  let saves = 0;
  const saveArchive = async () => { saves++; };
  const view = render(<ParameterWorkspace preview={original} rawCodecs={slowCodecs} saveArchive={saveArchive} />);
  view.rerender(<ParameterWorkspace preview={child.archive} rawCodecs={conformanceCodecs} saveArchive={saveArchive} />);
  await waitFor(() => expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-3"));
  await act(async () => {
    if (outcome === "resolved") gate.resolve?.();
    else gate.reject?.(new Error("Delayed original producer verification interrupted"));
    await parameterSourceFromArchive(original.json, slowCodecs).catch(() => null);
  });
  expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-3");
  expect(screen.queryByText(/Parameter source unavailable/)).toBeNull();
  expect(saves).toBe(0);
});

it("aborts a disposed child admission before invoking the host transaction", async () => {
  const original = await conformanceArchive(corpusText, false);
  const gate: { resolve: (() => void) | null; started: (() => void) | null } = { resolve: null, started: null };
  const pending = new Promise<void>(resolve => { gate.resolve = resolve; });
  const started = new Promise<void>(resolve => { gate.started = resolve; });
  let hold = false;
  const codecs = new Map([...conformanceCodecs].map(([schema, verify]) => [schema, async (bytes: Uint8Array) => {
    const identity = await verify(bytes);
    if (hold) { gate.started?.(); await pending; }
    return identity;
  }] as const));
  let saves = 0;
  const view = render(<ParameterWorkspace preview={original} rawCodecs={codecs} saveArchive={async () => { saves++; }} />);
  await waitFor(() => expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty("disabled", false));
  const subtle = crypto.subtle;
  const nativeDigest = subtle.digest.bind(subtle);
  const descriptor = Object.getOwnPropertyDescriptor(subtle, "digest");
  let archiveDigests = 0;
  const completed: { resolve: (() => void) | null } = { resolve: null };
  const childHashed = new Promise<void>(resolve => { completed.resolve = resolve; });
  Object.defineProperty(subtle, "digest", { configurable: true, value: async (algorithm: AlgorithmIdentifier, data: BufferSource) => {
    const bytes = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    const archiveIdentity = new TextDecoder().decode(bytes).startsWith(`${archiveSnapshotDomain}\n`);
    const result = await nativeDigest(algorithm, data);
    if (archiveIdentity && ++archiveDigests === 2) completed.resolve?.();
    return result;
  } });
  try {
    hold = true;
    fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
    await started;
    view.unmount();
    await act(async () => {
      gate.resolve?.();
      await childHashed;
    });
    expect(archiveDigests).toBe(2);
    expect(saves).toBe(0);
    expect(screen.queryByText(/Parameter revision saved/)).toBeNull();
  } finally {
    if (descriptor) Object.defineProperty(subtle, "digest", descriptor);
    else Reflect.deleteProperty(subtle, "digest");
  }
});
