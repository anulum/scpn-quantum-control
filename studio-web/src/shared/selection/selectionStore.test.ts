// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — shared selection store tests

import { act, renderHook } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { describe, expect, it } from "vitest";
import casesText from "../../../../data/studio/atlas_workspace_overlay_cases.json?raw";
import {
  overlayAtlasWorkspace,
  parseAtlasWorkspaceView,
} from "../../modules/atlas/workspaceOverlay";
import { type AtlasSnapshot, atlasSnapshotDigest, parseAtlasSnapshot } from "../contracts/atlas";
import {
  type SelectedObject,
  type SelectionOrigin,
  createSelectionStore,
  SelectionProvider,
  resolveSelection,
  sameObject,
  selectedObjectOf,
  selectedWorkspaceDocument,
  sharedSelectionId,
  useSelectionState,
  useSelectionStore,
  workspaceObjectId,
} from "./selectionStore";

const committed = JSON.parse(casesText) as {
  readonly public: unknown;
  readonly cases: readonly {
    readonly workspace: {
      readonly project_id: string;
      readonly documents: Record<
        string,
        { readonly schema: string; readonly body: Record<string, unknown> }
      >;
    };
  }[];
};
function base(): AtlasSnapshot {
  const parsed = parseAtlasSnapshot(committed.public);
  if (!parsed.ok) throw new Error(parsed.code);
  return parsed.value;
}
function chosen(objectId: string, revisionId: string | null = null): SelectedObject {
  return { snapshotDigest: "a".repeat(64), objectId, revisionId, access: "private" };
}

describe("Selection store", () => {
  it("counts each change once and leaves an identical selection silent", () => {
    const store = createSelectionStore();
    let calls = 0;
    const stop = store.subscribe(() => calls++);
    expect(store.current()).toEqual({ selected: null, origin: null, sequence: 0 });
    expect(store.select(chosen("x"), "atlas")).toBe(true);
    expect(store.select({ ...chosen("x"), snapshotDigest: "b".repeat(64) }, "source")).toBe(false);
    expect(store.select(chosen("x", "1".repeat(64)), "source")).toBe(true);
    expect(store.current()).toMatchObject({ origin: "source", sequence: 2 });
    expect(store.select(null, "atlas")).toBe(true);
    expect(store.select(null, "parameters")).toBe(false);
    stop();
    store.select(chosen("y"), "atlas");
    expect(calls).toBe(3);
  });

  it("test_atlas_studio_selection_02: bidirectional updates stop after one state change", () => {
    const store = createSelectionStore();
    const views: SelectionOrigin[] = ["atlas", "parameters", "source"];
    const applied: string[] = [];
    // Each view follows the others' selections and announces what it now shows.
    for (const view of views)
      store.subscribe(() => {
        const { selected, origin } = store.current();
        if (origin === view || selected === null) return;
        applied.push(`${view}:${selected.objectId}`);
        store.select(selected, view);
      });
    store.select(chosen("revision:sha256:1"), "atlas");
    expect(store.current().sequence).toBe(1);
    expect(applied).toEqual(["parameters:revision:sha256:1", "source:revision:sha256:1"]);
  });

  it("is read by a component and re-renders on change only", () => {
    const store = createSelectionStore();
    let renders = 0;
    const { result } = renderHook(() => {
      renders++;
      return useSelectionState(store);
    });
    expect(result.current.selected).toBeNull();
    act(() => {
      store.select(chosen("x"), "atlas");
    });
    act(() => {
      store.select(chosen("x"), "parameters");
    });
    expect(result.current).toMatchObject({ origin: "atlas", sequence: 1 });
    expect(renders).toBe(2);
    expect(renderHook(() => useSelectionState(null)).result.current).toEqual({
      selected: null,
      origin: null,
      sequence: 0,
    });
  });

  it("treats null selections and content changes as identity", () => {
    expect(sameObject(null, null)).toBe(true);
    expect(sameObject(chosen("x"), null)).toBe(false);
    expect(sameObject(null, chosen("x"))).toBe(false);
    expect(sameObject(chosen("x", "1"), chosen("x", "2"))).toBe(false);
    expect(sameObject(chosen("x"), chosen("y"))).toBe(false);
  });
});

describe("Selection against a snapshot", () => {
  it("test_atlas_studio_selection_03: a deleted node gives its tombstone", async () => {
    const first = await overlayAtlasWorkspace(
      base(),
      parseAtlasWorkspaceView(committed.cases[0]?.workspace),
    );
    const replaced = await overlayAtlasWorkspace(
      base(),
      parseAtlasWorkspaceView(committed.cases[2]?.workspace),
      first,
    );
    const firstDigest = await atlasSnapshotDigest(first);
    const run = first.nodes.find((node) => node.kind === "run");
    if (run === undefined) throw new Error("expected a run");
    const selected = selectedObjectOf(firstDigest, run);
    expect(resolveSelection(selected, first, firstDigest)).toEqual({ state: "present", node: run });
    const resolution = resolveSelection(selected, replaced, await atlasSnapshotDigest(replaced));
    expect(resolution.state).toBe("tombstone");
    expect(resolution).toMatchObject({
      explanation: "Removed at 2026-11-23T11:00:00Z: no longer in the admitted workspace.",
    });
  });

  it("explains a changed object and one the snapshot does not know", () => {
    const snapshot = base();
    const node = snapshot.nodes[1];
    if (node === undefined) throw new Error("expected a node");
    expect(resolveSelection(chosen(node.id), snapshot, "c".repeat(64)).state).toBe("present");
    const stale = resolveSelection(chosen(node.id, "1".repeat(64)), snapshot, "c".repeat(64));
    expect(stale).toMatchObject({
      state: "stale",
      explanation: `It changed since it was selected: its content was ${"1".repeat(64)} and is now ${"d".repeat(64)}.`,
    });
    const unrecorded = snapshot.nodes[0];
    if (unrecorded === undefined) throw new Error("expected a node");
    expect(resolveSelection(chosen(unrecorded.id, "1".repeat(64)), snapshot, "c")).toMatchObject({
      explanation: expect.stringContaining("is now unrecorded"),
    });
    expect(
      resolveSelection(chosen("definition:repo/src:gone.py"), snapshot, "c".repeat(64)),
    ).toEqual({
      state: "absent",
      explanation: `Snapshot ${"c".repeat(64)} neither holds it nor records its removal; it was selected in snapshot ${"a".repeat(64)}.`,
    });
  });

  it("test_atlas_studio_selection_04: private payload excluded from URL", async () => {
    const overlay = await overlayAtlasWorkspace(
      base(),
      parseAtlasWorkspaceView(committed.cases[0]?.workspace),
    );
    const digest = await atlasSnapshotDigest(overlay);
    for (const node of overlay.nodes) {
      const shared = sharedSelectionId(selectedObjectOf(digest, node));
      expect(shared).toBe(node.access === "public" ? node.id : null);
    }
    expect(sharedSelectionId(null)).toBeNull();
  });
});

describe("Selection of workspace sources", () => {
  it("supplies one store through the workbench context and none outside it", () => {
    const store = createSelectionStore();
    const inside = renderHook(() => useSelectionStore(), {
      wrapper: ({ children }: { readonly children: ReactNode }) =>
        createElement(SelectionProvider, { value: store }, children),
    });
    expect(inside.result.current).toBe(store);
    expect(renderHook(() => useSelectionStore()).result.current).toBeNull();
  });

  it("names each workspace document exactly as the overlay does", async () => {
    const workspace = committed.cases[0]?.workspace;
    if (workspace === undefined) throw new Error("expected a workspace");
    const overlay = await overlayAtlasWorkspace(base(), parseAtlasWorkspaceView(workspace));
    const ids = new Set(overlay.nodes.map((node) => node.id));
    let named = 0;
    for (const [digest, document] of Object.entries(workspace.documents)) {
      const id = workspaceObjectId(workspace.project_id, document.schema, digest, document.body);
      if (id === null) continue;
      named++;
      expect(ids.has(id), id).toBe(true);
    }
    expect(named).toBe(7);
    expect(workspaceObjectId(workspace.project_id, "resolved_settings.v1", "c", {})).toBeNull();
    expect(
      workspaceObjectId(workspace.project_id, "parameter_spec.v1", "c", { key: 7 }),
    ).toBeNull();
  });

  it("takes a private selection from a source record, or none it cannot name", () => {
    const content = (schema: string, body: unknown) => JSON.stringify({ schema, body });
    const project = "0f0e0d0c-0b0a-4908-8706-050403020100";
    const hash = "a".repeat(64);
    const revision = { schema: "experiment_revision.v1", sha256: "3".repeat(64) };
    expect(
      selectedWorkspaceDocument(project, hash, {
        ...revision,
        content: content(revision.schema, {}),
      }),
    ).toEqual({
      snapshotDigest: hash,
      objectId: `revision:sha256:${"3".repeat(64)}`,
      revisionId: "3".repeat(64),
      access: "private",
    });
    const spec = { schema: "parameter_spec.v1", sha256: "1".repeat(64) };
    expect(
      selectedWorkspaceDocument(project, hash, {
        ...spec,
        content: content(spec.schema, { key: "omega" }),
      }),
    ).toMatchObject({ objectId: `parameter:workspace.${project}/omega`, revisionId: null });
    expect(selectedWorkspaceDocument(project, hash, { ...spec, content: "{not json" })).toBeNull();
    expect(
      selectedWorkspaceDocument(project, hash, { ...spec, content: content(spec.schema, 3) }),
    ).toBeNull();
    expect(
      selectedWorkspaceDocument(project, hash, {
        schema: "local_run_record.v1",
        sha256: "8".repeat(64),
        content: content("local_run_record.v1", { run_id: "run-1", attempt_id: "a" }),
      }),
    ).toBeNull();
  });
});
