// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas selection linked to Studio drafts

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";
import pinText from "../../../../data/studio/atlas_snapshot_20261122.json?raw";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import { requirePresent } from "../../../test-support/requirePresent";
import type { RawCodec } from "../../shared/contracts";
import {
  type SelectionStore,
  SelectionProvider,
  createSelectionStore,
} from "../../shared/selection/selectionStore";
import {
  type WorkspaceArchivePreview,
  admitWorkspaceArchive,
} from "../../shared/storage/workspaceArchive";
import { ParameterWorkspace } from "../parameters/ParameterWorkspace";
import type { WorkspaceController } from "../../shared/workspace/workspaceService";
import AtlasView, { type AtlasWorkspaceSource, parameterEditorAdmits } from "./AtlasView";
import { admitAtlasSnapshot } from "./atlasSnapshot";
import { openSnapshotSession } from "./atlasSession";
import { atlasModule } from "./index";

afterEach(cleanup);

function within_(store: SelectionStore, view: ReactNode) {
  return <SelectionProvider value={store}>{view}</SelectionProvider>;
}
async function workspace(): Promise<{
  readonly preview: WorkspaceArchivePreview;
  readonly source: AtlasWorkspaceSource;
  readonly draft: string;
}> {
  const preview = await conformanceArchive(corpusText, false);
  const archive = await admitWorkspaceArchive(preview.json, conformanceCodecs);
  // The project has no draft reference; the editor opens its latest listed revision.
  const refs = archive.manifest.body["revision_refs"] as readonly { readonly sha256: string }[];
  const draft = requirePresent(refs.at(-1));
  return {
    preview,
    source: { json: preview.json, codecs: conformanceCodecs },
    draft: draft.sha256,
  };
}
function revisionLabel(hash: string): string {
  return `Experiment revision ${hash.slice(0, 12)}`;
}
async function row(label: string): Promise<HTMLElement> {
  return within(
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 }),
  ).findByRole("button", { name: label }, { timeout: 15_000 });
}

describe("Atlas selection linked to Studio drafts", () => {
  it("test_atlas_studio_selection_01: Atlas->Studio->Atlas preserves identity", async () => {
    const { preview, source, draft } = await workspace();
    const originalJson = preview.json;
    const store = createSelectionStore();
    render(
      within_(
        store,
        <AtlasView
          instrument="workspace"
          contextSuffix=""
          workspace={source}
          draftHref="#/workspace"
        />,
      ),
    );
    fireEvent.click(await row(revisionLabel(draft)));
    expect(store.current()).toMatchObject({
      origin: "atlas",
      selected: {
        objectId: `revision:sha256:${draft}`,
        revisionId: draft,
        access: "private",
      },
    });
    const opening = await screen.findByRole("link", { name: "Open as a new parameter draft" });
    expect(opening.getAttribute("href")).toBe("#/workspace");
    expect(
      screen.getByRole("link", { name: "Link to this view" }).getAttribute("href"),
    ).not.toContain(draft);
    cleanup();

    const saved: WorkspaceArchivePreview[] = [];
    render(
      within_(
        store,
        <ParameterWorkspace
          preview={preview}
          rawCodecs={conformanceCodecs}
          saveArchive={async (archive) => {
            saved.push(archive);
          }}
        />,
      ),
    );
    expect(
      await screen.findByText(
        `Draft from revision ${draft.slice(0, 12)}, selected in atlas. Saving appends a child revision; that revision stays as it is.`,
      ),
    ).toBeTruthy();
    await waitFor(() =>
      expect(screen.getByLabelText("Selected value")).toHaveProperty("value", "-0"),
    );
    fireEvent.change(screen.getByLabelText("Selected value"), { target: { value: "-3" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply value" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save parameter revision" })).toHaveProperty(
        "disabled",
        false,
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save parameter revision" }));
    await waitFor(() => expect(saved).toHaveLength(1));
    const child = requirePresent(store.current().selected);
    expect(store.current().origin).toBe("parameters");
    expect(child.objectId).not.toBe(`revision:sha256:${draft}`);
    cleanup();

    const reopened = await admitWorkspaceArchive(requirePresent(saved[0]).json, conformanceCodecs);
    const before = await admitWorkspaceArchive(originalJson, conformanceCodecs);
    expect(reopened.documents[draft]).toEqual(before.documents[draft]);
    render(
      within_(
        store,
        <AtlasView
          instrument="workspace"
          contextSuffix=""
          workspace={{ json: requirePresent(saved[0]).json, codecs: conformanceCodecs }}
        />,
      ),
    );
    await row(revisionLabel(draft));
    await waitFor(() =>
      expect(
        within(screen.getByRole("region", { name: "Shared selection" })).getByRole("status")
          .textContent,
      ).toBe(`Selected in parameters: ${revisionLabel(child.revisionId as string)} (revision).`),
    );
    const relations = screen.getByRole("table", { name: "Atlas relations table" });
    expect(relations.textContent).toContain(
      `${child.objectId}derives_fromrevision:sha256:${draft}`,
    );
    expect(preview.json).toBe(originalJson);
  }, 60_000);

  it("test_atlas_studio_selection_05: unsupported configuration opens read-only", async () => {
    const { source, draft } = await workspace();
    const store = createSelectionStore();
    render(
      within_(
        store,
        <AtlasView
          instrument="workspace"
          contextSuffix=""
          workspace={source}
          draftHref="#/workspace"
          checkDraft={async () => "fixture refusal"}
        />,
      ),
    );
    fireEvent.click(await row(revisionLabel(draft)));
    const panel = await screen.findByRole("region", { name: "Revision draft" });
    expect(await within(panel).findByRole("note")).toHaveProperty(
      "textContent",
      "Read-only: the parameter editor refuses this revision's configuration (fixture refusal). Its record is shown as committed to the workspace.",
    );
    expect(within(panel).queryByRole("link")).toBeNull();
  }, 30_000);

  it("asks the parameter editor's own admission", async () => {
    const { source, draft } = await workspace();
    expect(await parameterEditorAdmits(source, draft)).toBeNull();
    expect(await parameterEditorAdmits(source, "0".repeat(64))).toContain("must be listed");
    expect(await parameterEditorAdmits({ ...source, json: "{" }, draft)).toEqual(
      expect.any(String),
    );
  });
});

describe("Atlas selection in shared links and other views", () => {
  it("links a public selection and restores it from the link", async () => {
    const store = createSelectionStore();
    render(within_(store, <AtlasView instrument="architecture" contextSuffix="" />));
    const table = await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    fireEvent.click(requirePresent(within(table).getAllByRole("button")[0]));
    const selected = requirePresent(store.current().selected);
    expect(selected.access).toBe("public");
    const link = requirePresent(
      screen.getByRole("link", { name: "Link to this view" }).getAttribute("href"),
    );
    expect(decodeURIComponent(decodeURIComponent(link))).toContain(`sel=${selected.objectId}`);
    cleanup();

    const restored = createSelectionStore();
    render(
      within_(
        restored,
        <AtlasView instrument="architecture" contextSuffix={link.slice(link.indexOf("?"))} />,
      ),
    );
    await waitFor(() => expect(restored.current().selected?.objectId).toBe(selected.objectId));
    expect(restored.current().origin).toBe("atlas");
  }, 30_000);

  it("explains a selection the open snapshot neither holds nor buried", async () => {
    const store = createSelectionStore();
    store.select(
      {
        snapshotDigest: "f".repeat(64),
        objectId: "definition:repo/src:gone.py",
        revisionId: null,
        access: "public",
      },
      "source",
    );
    render(within_(store, <AtlasView instrument="architecture" contextSuffix="" />));
    const region = await screen.findByRole(
      "region",
      { name: "Shared selection" },
      { timeout: 15_000 },
    );
    expect(within(region).getByRole("status").getAttribute("data-selection")).toBe("absent");
    expect(region.textContent).toContain("neither holds it nor records its removal");
  }, 30_000);

  it("follows another view's selection and offers the lens that shows it", async () => {
    const { draft, source } = await workspace();
    const store = createSelectionStore();
    render(
      within_(store, <AtlasView instrument="architecture" contextSuffix="" workspace={source} />),
    );
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    await waitFor(() =>
      expect(screen.getByRole("region", { name: "Atlas snapshot" }).textContent).toContain(
        "with your workspace",
      ),
    );
    store.select(
      {
        snapshotDigest: "a".repeat(64),
        objectId: `revision:sha256:${draft}`,
        revisionId: draft,
        access: "private",
      },
      "source",
    );
    const lens = await screen.findByRole("link", { name: "Show it in the Workspace lens" });
    expect(lens.getAttribute("href")).toContain("#/atlas/workspace?");
  }, 30_000);

  it("reports a refused workspace and returns to the committed snapshot without one", async () => {
    const { source } = await workspace();
    const store = createSelectionStore();
    const { rerender } = render(
      within_(
        store,
        <AtlasView instrument="workspace" contextSuffix="" workspace={{ ...source, json: "{}" }} />,
      ),
    );
    expect((await screen.findByRole("alert", {}, { timeout: 15_000 })).textContent).toContain(
      "Your workspace is not shown in the Atlas:",
    );
    rerender(
      within_(store, <AtlasView instrument="workspace" contextSuffix="" workspace={source} />),
    );
    await waitFor(() =>
      expect(screen.getByRole("region", { name: "Atlas snapshot" }).textContent).toContain(
        "with your workspace",
      ),
    );
    rerender(
      within_(store, <AtlasView instrument="workspace" contextSuffix="" workspace={null} />),
    );
    await waitFor(() =>
      expect(screen.getByRole("region", { name: "Atlas snapshot" }).textContent).not.toContain(
        "with your workspace",
      ),
    );
  }, 30_000);
});

describe("Atlas selection boundaries", () => {
  it("states an admitted revision without a draft address", async () => {
    const { source, draft } = await workspace();
    render(
      within_(
        createSelectionStore(),
        <AtlasView instrument="workspace" contextSuffix="" workspace={source} />,
      ),
    );
    fireEvent.click(await row(revisionLabel(draft)));
    const panel = await screen.findByRole("region", { name: "Revision draft" });
    expect(
      await within(panel).findByText(/The parameter editor admits this revision/),
    ).toBeTruthy();
    expect(within(panel).queryByRole("link")).toBeNull();
  }, 30_000);

  it("discards an overlay or a refusal that completes after its workspace changed", async () => {
    const { source } = await workspace();
    const store = createSelectionStore();
    // Verifiers that answer after 300 ms, so each admission is still running when withdrawn.
    const slow = (refuse: boolean): ReadonlyMap<string, RawCodec> =>
      new Map(
        [...source.codecs].map(([schema, codec]) => [
          schema,
          async (content: Uint8Array) => {
            await new Promise((resolve) => setTimeout(resolve, 300));
            if (refuse) throw new Error("slow refusal");
            return codec(content);
          },
        ]),
      );
    const view = (codecs: ReadonlyMap<string, RawCodec> | null) => (
      <AtlasView
        instrument="workspace"
        contextSuffix=""
        workspace={codecs === null ? null : { json: source.json, codecs }}
      />
    );
    const { rerender } = render(within_(store, view(slow(false))));
    await screen.findByText(/ · committed with this build · /, {}, { timeout: 15_000 });
    rerender(within_(store, view(slow(true))));
    rerender(within_(store, view(null)));
    await new Promise((resolve) => setTimeout(resolve, 1500));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("region", { name: "Atlas snapshot" }).textContent).not.toContain(
      "with your workspace",
    );
  }, 30_000);

  it("keeps an opened pin when the workspace arrives or leaves", async () => {
    const { source } = await workspace();
    const store = createSelectionStore();
    const { rerender } = render(
      within_(store, <AtlasView instrument="architecture" contextSuffix="" />),
    );
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    fireEvent.change(screen.getByLabelText("Open a snapshot pin"), {
      target: { files: [new File([pinText], "other.json", { type: "application/json" })] },
    });
    await screen.findByText(/ · other\.json · /, {}, { timeout: 15_000 });
    rerender(
      within_(store, <AtlasView instrument="architecture" contextSuffix="" workspace={source} />),
    );
    await new Promise((resolve) => setTimeout(resolve, 1500));
    rerender(
      within_(store, <AtlasView instrument="architecture" contextSuffix="" workspace={null} />),
    );
    expect(screen.getByText(/ · other\.json · /)).toBeTruthy();
    expect(screen.getByRole("region", { name: "Atlas snapshot" }).textContent).not.toContain(
      "with your workspace",
    );
  }, 30_000);

  it("restores no selection a link names but the snapshot does not hold publicly", async () => {
    const store = createSelectionStore();
    const filter = encodeURIComponent("sel=definition:repo/none.py");
    render(
      within_(store, <AtlasView instrument="architecture" contextSuffix={`?filter=${filter}`} />),
    );
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(store.current().selected).toBeNull();
  }, 30_000);

  it("shares nothing from a session that holds no snapshot on the main thread", async () => {
    const admission = await admitAtlasSnapshot(pinText);
    if (!admission.ok) throw new Error(admission.reason);
    const session = {
      ...openSnapshotSession(admission.snapshot, admission.digest, "worker fixture"),
      snapshot: null,
    };
    const store = createSelectionStore();
    render(
      within_(
        store,
        <AtlasView
          instrument="architecture"
          contextSuffix=""
          open={() => ({
            tier: "main thread",
            result: Promise.resolve({ ok: true, session }),
            cancel: () => {},
          })}
        />,
      ),
    );
    const table = await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    fireEvent.click(requirePresent(within(table).getAllByRole("button")[0]));
    expect(store.current().selected).toBeNull();
  }, 30_000);

  it("overlays a saved workspace when no preview is open", async () => {
    const { preview, draft } = await workspace();
    const stage = requirePresent(atlasModule.stages[0]);
    const services = {
      contextSuffix: "",
      workspaceHref: "#/workspace",
      codecs: conformanceCodecs,
      documents: [],
      workspace: { preview: null, saved: { preview } } as unknown as WorkspaceController,
    };
    render(within_(createSelectionStore(), stage.render(services, "workspace")));
    expect(await row(revisionLabel(draft))).toBeTruthy();
  }, 30_000);

  it("opens a revision of another workspace as this workspace's draft revision", async () => {
    const { preview } = await workspace();
    const store = createSelectionStore();
    store.select(
      {
        snapshotDigest: "a".repeat(64),
        objectId: `revision:sha256:${"0".repeat(64)}`,
        revisionId: "0".repeat(64),
        access: "private",
      },
      "atlas",
    );
    render(
      within_(
        store,
        <ParameterWorkspace
          preview={preview}
          rawCodecs={conformanceCodecs}
          saveArchive={async () => {}}
        />,
      ),
    );
    expect(
      await screen.findByText(
        "The revision selected in atlas is not in this workspace; its draft revision opens instead.",
      ),
    ).toBeTruthy();
  }, 30_000);
});
