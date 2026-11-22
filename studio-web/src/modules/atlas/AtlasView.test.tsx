// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas view tests

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { InThreadAtlasWorker } from "../../../test-support/atlasWorker";
import { atlasSnapshotDigest, parseAtlasSnapshot } from "../../shared/contracts/atlas";
import {
  type AtlasOpening,
  type AtlasOpeningTask,
  atlasWorkerThreshold,
  openAtlasSession,
} from "./atlasSession";
import { atlasSnapshotPinSchema } from "./atlasSnapshot";
import AtlasView from "./AtlasView";
import { atlasNodeZoomLimit } from "./atlasViewModel";

beforeEach(() => {
  InThreadAtlasWorker.started = [];
});
afterEach(cleanup);

type Wire = Record<string, unknown>;

function node(name: string, kind = "definition"): Wire {
  return {
    id: `${kind}:fixture/${name}`,
    kind,
    label: `Node ${name}`,
    source_ref: `fixture/${name}`,
    source_digest: null,
    validity_domain: null,
    as_of: null,
    access: "public",
    status: "current",
    evidence_refs: [],
    null_reasons: { source_digest: "fixture", validity_domain: "fixture", as_of: "fixture" },
  };
}
function edge(source: string, target: string): Wire {
  return {
    source: `definition:fixture/${source}`,
    kind: "requires",
    target: `definition:fixture/${target}`,
    relation_version: "atlas_relation.v1",
    derivation_method: "declared",
    source_ref: "fixture/edges",
    evidence_digest: null,
    assumptions: [],
    mapping: null,
    as_of: null,
    access: "public",
  };
}
/** A snapshot pin with its recomputed digest, padded to a size when asked. */
async function pinOf(
  nodes: Wire[],
  edges: Wire[],
  size = 0,
): Promise<{ text: string; digest: string }> {
  const snapshot = {
    schema: "quantum_atlas_snapshot.v1",
    extractor_version: "fixture",
    source_commit: "0".repeat(40),
    evidence_set_hash: "a".repeat(64),
    access_scope: "public",
    nodes,
    edges,
    tombstones: [],
    exclusions: [],
  };
  const parsed = parseAtlasSnapshot(snapshot);
  if (!parsed.ok) throw new Error(`${parsed.code} at ${parsed.path}`);
  const digest = await atlasSnapshotDigest(parsed.value);
  const text = JSON.stringify({
    schema: atlasSnapshotPinSchema,
    snapshot,
    snapshot_digest: digest,
  });
  return { text: text.padEnd(size, " "), digest };
}
const diamond = await pinOf(
  ["a", "b", "c", "d", "e"].map((name) => node(name)),
  [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d"), edge("d", "e")],
);
function opener(text: string): () => AtlasOpeningTask {
  return () => openAtlasSession(text, "fixture", () => new InThreadAtlasWorker());
}
async function openDiamond(contextSuffix = ""): Promise<void> {
  render(
    <AtlasView
      instrument="architecture"
      contextSuffix={contextSuffix}
      open={opener(diamond.text)}
    />,
  );
  await screen.findByRole("table", { name: "Atlas nodes" });
}
function graphButton(label: string): HTMLElement {
  return within(screen.getByRole("region", { name: "Atlas graph" })).getByRole("button", {
    name: label,
  });
}
function summary(): string | null {
  return screen.getByRole("status", { name: "Atlas summary" }).textContent;
}
function pinFile(text: string, name: string): File {
  return new File([text], name, { type: "application/json" });
}

describe("Atlas view", () => {
  it("test_atlas_graph_navigation_01: graph and tables show the same selected node and relation", async () => {
    await openDiamond();
    fireEvent.click(
      within(screen.getByRole("table", { name: "Atlas nodes" })).getByRole("button", {
        name: "Node b",
      }),
    );
    expect(graphButton("Node b").getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("row", { selected: true }).textContent).toContain("Node b");
    expect(
      within(screen.getByRole("region", { name: "Atlas inspector" })).getByText(
        "definition:fixture/b",
      ),
    ).toBeTruthy();
    fireEvent.click(graphButton("Node d"));
    fireEvent.click(
      screen.getByRole("button", { name: "definition:fixture/b requires definition:fixture/d" }),
    );
    const relations = screen.getByRole("table", { name: "Atlas relations table" });
    expect(within(relations).getByRole("row", { selected: true }).textContent).toContain(
      "definition:fixture/b",
    );
    expect(
      document
        .querySelector('[data-edge="definition:fixture/b requires definition:fixture/d"]')
        ?.getAttribute("data-selected"),
    ).toBe("true");
    expect(
      within(screen.getByRole("navigation", { name: "Atlas breadcrumbs" }))
        .getAllByRole("button")
        .map((item) => item.textContent),
    ).toEqual(["definition:fixture/b", "definition:fixture/d"]);
  });

  it("test_atlas_graph_navigation_02: the keyboard follows directed edges", async () => {
    await openDiamond();
    fireEvent.click(graphButton("Node a"));
    fireEvent.keyDown(graphButton("Node a"), { key: "ArrowRight" });
    expect(graphButton("Node b").getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(graphButton("Node b"), { key: "ArrowRight" });
    expect(graphButton("Node d").getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(graphButton("Node d"), { key: "ArrowLeft" });
    expect(graphButton("Node b").getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(graphButton("Node b"), { key: "Enter" });
    await screen.findByRole("button", { name: "Leave the neighbourhood of definition:fixture/b" });
    await waitFor(() => expect(summary()).toBe("Showing 3 of 3 nodes."));
    fireEvent.keyDown(graphButton("Node b"), { key: "Escape" });
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /Leave the neighbourhood/ })).toBeNull(),
    );
  });

  it("test_atlas_graph_navigation_03: a deep link restores the snapshot, the lens and the filter", async () => {
    const filter = encodeURIComponent("focus=definition:fixture/b&hops=2");
    render(
      <AtlasView
        instrument="architecture"
        contextSuffix={`?project=p&snapshot=${diamond.digest}&filter=${filter}`}
        open={opener(diamond.text)}
      />,
    );
    await screen.findByRole("button", { name: "Leave the neighbourhood of definition:fixture/b" });
    await waitFor(() => expect(summary()).toBe("Showing 5 of 5 nodes."));
    expect((screen.getByLabelText("Hops") as HTMLSelectElement).value).toBe("2");
    const link = screen.getByRole("link", { name: "Link to this view" }).getAttribute("href");
    expect(link).toBe(
      `#/atlas/architecture?project=p&snapshot=${diamond.digest}&filter=focus%3Ddefinition%253Afixture%252Fb%26hops%3D2`,
    );
    const lenses = screen.getByRole("navigation", { name: "Atlas lenses" });
    expect(
      within(lenses).getByRole("link", { name: "Architecture" }).getAttribute("aria-current"),
    ).toBe("page");
    expect(within(lenses).getByRole("link", { name: "Evidence" }).getAttribute("href")).toBe(
      link?.replace("/architecture?", "/evidence?"),
    );
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "x".repeat(300) } });
    expect(screen.getByText("This search is too long to link.")).toBeTruthy();
    expect(within(lenses).getByRole("link", { name: "Claims" }).getAttribute("href")).toBe(
      `#/atlas/claims?project=p&snapshot=${diamond.digest}`,
    );
    cleanup();
    render(
      <AtlasView instrument="claims" contextSuffix="?snapshot=other" open={opener(diamond.text)} />,
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "the requested snapshot other is not the open snapshot",
    );
    expect(screen.getByRole("heading", { name: "Atlas" })).toBeTruthy();
    cleanup();
    render(
      <AtlasView
        instrument="claims"
        contextSuffix="?filter=hops%3D9"
        open={opener(diamond.text)}
      />,
    );
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The requested Atlas filter was refused because its hops field must be 1 to 6; the view opens without it.",
    );
  });

  it("follows a changed deep link and refuses a focus the snapshot does not hold", async () => {
    const view = (contextSuffix: string) => (
      <AtlasView
        instrument="architecture"
        contextSuffix={contextSuffix}
        open={opener(diamond.text)}
      />
    );
    const { rerender } = render(view(""));
    await waitFor(() => expect(summary()).toBe("Showing 5 of 5 nodes."));
    rerender(view("?filter=q%3Dnode%2Be"));
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."));
    expect((screen.getByLabelText("Search") as HTMLInputElement).value).toBe("node e");
    rerender(view("?filter=focus%3Ddefinition%3Afixture%2Fzz"));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "the node is not in the admitted snapshot",
    );
  });

  it("test_atlas_graph_navigation_04: a 10,000-node source shows a bounded subset and its omission, and its worker query can be cancelled", async () => {
    const large = await pinOf(
      Array.from({ length: 10_000 }, (_unused, index) =>
        node(`n${String(index).padStart(5, "0")}`),
      ),
      [],
    );
    expect(large.text.length).toBeGreaterThan(atlasWorkerThreshold);
    render(<AtlasView instrument="architecture" contextSuffix="" open={opener(large.text)} />);
    expect(
      within(await screen.findByRole("status", { name: "Atlas progress" })).getByRole("button", {
        name: "Cancel",
      }),
    ).toBeTruthy();
    await waitFor(
      () => expect(screen.queryByRole("status", { name: "Atlas summary" })).toBeTruthy(),
      {
        timeout: 30_000,
      },
    );
    expect(summary()).toBe("Showing 500 of 10000 nodes; 9500 more are not shown.");
    expect(
      within(screen.getByRole("table", { name: "Atlas nodes" })).getAllByRole("row"),
    ).toHaveLength(101);
    expect(screen.getByRole("navigation", { name: "Atlas node rows" }).textContent).toContain(
      "Rows 1–100 of 500",
    );
    expect(screen.getByText(/queried in a worker/)).toBeTruthy();
    expect(screen.getByText("Detail: kinds")).toBeTruthy();
    expect(screen.getByRole("list", { name: "Atlas relations by kind" }).textContent).toBe("");
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "n0999" } });
    const progress = await screen.findByRole("status", { name: "Atlas progress" });
    expect(progress.textContent).toContain("Querying the snapshot in a worker…");
    fireEvent.click(within(progress).getByRole("button", { name: "Cancel" }));
    expect(
      (
        await screen.findByText("The query was cancelled; the previous answer stays shown.")
      ).getAttribute("role"),
    ).toBe("alert");
    expect(summary()).toBe("Showing 500 of 10000 nodes; 9500 more are not shown.");
    expect(InThreadAtlasWorker.started[0]?.terminated).toBe(true);
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "n09999" } });
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."), { timeout: 30_000 });
    expect(InThreadAtlasWorker.started).toHaveLength(2);
  }, 60_000);

  it("test_atlas_graph_navigation_05: layout changes do not alter the graph digest", async () => {
    await openDiamond();
    fireEvent.click(graphButton("Node c"));
    const circle = (): Element | null =>
      document.querySelector('[data-node="definition:fixture/c"] circle');
    const position = circle()?.getAttribute("cy");
    fireEvent.click(screen.getByRole("button", { name: "Pin its position" }));
    await waitFor(() => expect(circle()?.getAttribute("cy")).not.toBe(position));
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(screen.getByText("Detail: details")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Zoom in" }).hasAttribute("disabled")).toBe(true);
    expect(document.querySelector('[data-node="definition:fixture/c"] text')?.textContent).toBe(
      "definition · current",
    );
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(screen.getByRole("button", { name: "Zoom out" }).hasAttribute("disabled")).toBe(true);
    expect(document.querySelector('[data-group="definition"]')?.textContent).toBe("definition: 5");
    expect(screen.getByRole("list", { name: "Atlas relations by kind" }).textContent).toBe(
      "definition requires definition: 5",
    );
    fireEvent.click(screen.getByRole("button", { name: "definition: 5" }));
    expect(screen.getByText("Detail: nodes")).toBeTruthy();
    const reopened = await pinOf(
      ["a", "b", "c", "d", "e"].map((name) => node(name)),
      [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("c", "d"), edge("d", "e")],
    );
    expect(reopened.digest).toBe(diamond.digest);
    expect(graphButton("Node c").getAttribute("aria-pressed")).toBe("true");
  });

  it("pages the tables, opens the selection's page and keeps it for a new answer", async () => {
    const names = Array.from(
      { length: 150 },
      (_unused, index) => `p${String(index).padStart(3, "0")}`,
    );
    const paged = await pinOf(
      names.map((name) => node(name)),
      names.slice(1).map((name, index) => edge(names[index] as string, name)),
    );
    render(<AtlasView instrument="architecture" contextSuffix="" open={opener(paged.text)} />);
    await screen.findByRole("table", { name: "Atlas nodes" });
    const nodeRows = (): HTMLElement => screen.getByRole("navigation", { name: "Atlas node rows" });
    const relationRows = (): HTMLElement =>
      screen.getByRole("navigation", { name: "Atlas relation rows" });
    const control = (pager: HTMLElement, name: string): HTMLElement =>
      within(pager).getByRole("button", { name });
    expect(nodeRows().textContent).toContain("Rows 1–100 of 150");
    expect(control(nodeRows(), "Previous rows").hasAttribute("disabled")).toBe(true);
    fireEvent.click(graphButton("Node p120"));
    expect(nodeRows().textContent).toContain("Rows 101–150 of 150");
    expect(screen.getByRole("row", { selected: true }).textContent).toContain("Node p120");
    expect(control(nodeRows(), "Next rows").hasAttribute("disabled")).toBe(true);
    fireEvent.click(control(nodeRows(), "Previous rows"));
    expect(nodeRows().textContent).toContain("Rows 1–100 of 150");
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "node p" } });
    await waitFor(() => expect(nodeRows().textContent).toContain("Rows 101–150 of 150"));
    expect(relationRows().textContent).toContain("Rows 1–100 of 149");
    fireEvent.click(control(relationRows(), "Next rows"));
    expect(relationRows().textContent).toContain("Rows 101–149 of 149");
    const relations = screen.getByRole("table", { name: "Atlas relations table" });
    fireEvent.click(
      within(relations).getAllByRole("button", { name: "requires" })[0] as HTMLElement,
    );
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "node" } });
    await waitFor(() => expect(nodeRows().textContent).toContain("Rows 1–100 of 150"));
    expect(relationRows().textContent).toContain("Rows 101–149 of 149");
    expect(within(relations).getByRole("row", { selected: true }).textContent).toContain(
      "definition:fixture/p100",
    );
  }, 30_000);

  it("aggregates relations between different kinds at the kind zoom", async () => {
    const mixed = await pinOf(
      [node("a"), node("t", "test")],
      [{ ...edge("a", "a"), kind: "contains", target: "test:fixture/t" }],
    );
    render(<AtlasView instrument="architecture" contextSuffix="" open={opener(mixed.text)} />);
    await screen.findByRole("table", { name: "Atlas nodes" });
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(document.querySelector('[data-link="definition contains test"]')).toBeTruthy();
    expect(atlasNodeZoomLimit).toBeGreaterThan(2);
  });

  it("shows the refusal of an inadmissible committed snapshot and the neighbourhood controls", async () => {
    await act(async () => {
      render(
        <AtlasView
          instrument="architecture"
          contextSuffix=""
          open={() => ({
            tier: "main thread",
            result: Promise.resolve({ ok: false, reason: "broken pin" }),
            cancel: () => {},
          })}
        />,
      );
    });
    expect((await screen.findByRole("alert")).textContent).toContain("broken pin");
    cleanup();
    await openDiamond();
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "node e" } });
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."));
    fireEvent.click(graphButton("Node e"));
    fireEvent.click(screen.getByRole("button", { name: "Show its neighbourhood" }));
    fireEvent.change(screen.getByLabelText("Hops"), { target: { value: "2" } });
    await screen.findByRole("button", { name: "Leave the neighbourhood of definition:fixture/e" });
    fireEvent.click(
      screen.getByRole("button", { name: "Leave the neighbourhood of definition:fixture/e" }),
    );
    await act(async () => undefined);
    expect(screen.getByRole("list", { name: "Atlas explanation" }).textContent).toContain(
      "edge kinds all admitted kinds",
    );
  });

  it("drops a selection the answer no longer holds instead of inspecting a missing node", async () => {
    await openDiamond();
    fireEvent.click(graphButton("Node b"));
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "node e" } });
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."));
    expect(screen.getByText("Select a node or a relation to inspect it.")).toBeTruthy();
    fireEvent.click(
      within(screen.getByRole("navigation", { name: "Atlas breadcrumbs" })).getByRole("button", {
        name: "definition:fixture/b",
      }),
    );
    expect(screen.getByText("Select a node or a relation to inspect it.")).toBeTruthy();
  });

  it("drops a selected relation the answer no longer holds", async () => {
    await openDiamond();
    const relations = screen.getByRole("table", { name: "Atlas relations table" });
    fireEvent.click(
      within(relations).getAllByRole("button", { name: "requires" })[0] as HTMLElement,
    );
    expect(
      within(screen.getByRole("region", { name: "Atlas inspector" })).getByText("Relation"),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "node e" } });
    await waitFor(() =>
      expect(screen.getByText("Select a node or a relation to inspect it.")).toBeTruthy(),
    );
  });

  it("selects relations from the table and moves with keys that have a target only", async () => {
    await openDiamond();
    fireEvent.keyDown(graphButton("Node c"), { key: "ArrowDown" });
    expect(graphButton("Node a").getAttribute("aria-pressed")).toBe("true");
    fireEvent.keyDown(graphButton("Node a"), { key: "Home" });
    expect(graphButton("Node a").getAttribute("aria-pressed")).toBe("true");
    const relations = screen.getByRole("table", { name: "Atlas relations table" });
    fireEvent.click(
      within(relations).getAllByRole("button", { name: "requires" })[0] as HTMLElement,
    );
    const inspector = screen.getByRole("region", { name: "Atlas inspector" });
    expect(within(inspector).getByText("Relation")).toBeTruthy();
    expect(within(inspector).getByText("declared from fixture/edges")).toBeTruthy();
  });

  it("reports omitted relations and refuses a snapshot above the ceiling", async () => {
    const names = Array.from(
      { length: 50 },
      (_unused, index) => `m${String(index).padStart(2, "0")}`,
    );
    const dense = await pinOf(
      names.map((name) => node(name)),
      names.flatMap((source) =>
        names.filter((target) => target !== source).map((target) => edge(source, target)),
      ),
    );
    render(<AtlasView instrument="architecture" contextSuffix="" open={opener(dense.text)} />);
    await waitFor(
      () => expect(summary()).toBe("Showing 50 of 50 nodes; 450 relations are not shown."),
      {
        timeout: 30_000,
      },
    );
    cleanup();
    const oversized = await pinOf(
      Array.from({ length: 10_001 }, (_unused, index) =>
        node(`o${String(index).padStart(5, "0")}`),
      ),
      [],
    );
    render(<AtlasView instrument="architecture" contextSuffix="" open={opener(oversized.text)} />);
    expect((await screen.findByRole("alert", {}, { timeout: 30_000 })).textContent).toBe(
      "above the admitted node or edge ceiling",
    );
  }, 60_000);
});

describe("Opening another snapshot pin", () => {
  function input(): HTMLInputElement {
    return screen.getByLabelText("Open a snapshot pin") as HTMLInputElement;
  }

  it("replaces the open snapshot, keeps it on a refusal and clears the chooser", async () => {
    await openDiamond();
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(screen.getByText("Detail: kinds")).toBeTruthy();
    const single = await pinOf([node("solo")], []);
    fireEvent.change(input(), { target: { files: [pinFile(single.text, "solo.json")] } });
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."));
    expect(screen.getByText(/ · solo\.json · /)).toBeTruthy();
    expect(screen.getByText("Detail: nodes")).toBeTruthy();
    expect(input().value).toBe("");
    fireEvent.change(input(), { target: { files: [pinFile("{}", "bad.json")] } });
    expect((await screen.findByText(/bad\.json was refused/)).textContent).toBe(
      "bad.json was refused: The Atlas snapshot pin has an unknown schema. The open snapshot stays.",
    );
    expect(summary()).toBe("Showing 1 of 1 nodes.");
    fireEvent.change(input(), { target: { files: [] } });
    expect(summary()).toBe("Showing 1 of 1 nodes.");
  });

  it("lets an opened pin override the requested snapshot until the request changes", async () => {
    const view = (contextSuffix: string) => (
      <AtlasView
        instrument="architecture"
        contextSuffix={contextSuffix}
        open={opener(diamond.text)}
      />
    );
    const { rerender } = render(view(`?snapshot=${diamond.digest}`));
    await screen.findByRole("table", { name: "Atlas nodes" });
    const single = await pinOf([node("solo")], []);
    fireEvent.change(input(), { target: { files: [pinFile(single.text, "solo.json")] } });
    await waitFor(() => expect(summary()).toBe("Showing 1 of 1 nodes."));
    expect(screen.queryByRole("alert")).toBeNull();
    rerender(view(`?snapshot=${single.digest}`));
    expect(summary()).toBe("Showing 1 of 1 nodes.");
    rerender(view("?snapshot=other"));
    expect((await screen.findByRole("alert")).textContent).toContain(
      "the requested snapshot other is not the open snapshot",
    );
  });

  it("cancels a large pin's worker admission and discards an opening that a later one replaced", async () => {
    const results: ((value: AtlasOpening | null) => void)[] = [];
    let cancelled = 0;
    const openFile = (): AtlasOpeningTask => ({
      tier: "worker",
      result: new Promise((resolve) => results.push(resolve)),
      cancel: () => {
        cancelled += 1;
      },
    });
    render(
      <AtlasView
        instrument="architecture"
        contextSuffix=""
        open={opener(diamond.text)}
        openFile={openFile}
      />,
    );
    await screen.findByRole("table", { name: "Atlas nodes" });
    fireEvent.change(input(), { target: { files: [pinFile("{}", "first.json")] } });
    const progress = await screen.findByRole("status", { name: "Atlas progress" });
    expect(progress.textContent).toContain("Verifying first.json in a worker…");
    fireEvent.change(input(), { target: { files: [pinFile("{}", "second.json")] } });
    expect(cancelled).toBe(1);
    let disposed = false;
    const replaced = await pinOf([node("late")], []);
    const late = await openAtlasSession(replaced.text, "late").result;
    if (late === null || !late.ok) throw new Error("expected a session");
    await act(async () => {
      results[0]?.({ ok: true, session: { ...late.session, dispose: () => (disposed = true) } });
    });
    expect(disposed).toBe(true);
    fireEvent.click(
      within(screen.getByRole("status", { name: "Atlas progress" })).getByRole("button", {
        name: "Cancel",
      }),
    );
    expect(cancelled).toBe(2);
    await act(async () => results[1]?.(null));
    expect(screen.getByRole("alert").textContent).toBe(
      "Opening second.json was cancelled; the open snapshot stays.",
    );
    expect(summary()).toBe("Showing 5 of 5 nodes.");
  });

  it("reports a cancelled committed opening and disposes a late refused or unmounted session", async () => {
    let finish: (value: AtlasOpening | null) => void = () => {};
    const { unmount } = render(
      <AtlasView
        instrument="architecture"
        contextSuffix=""
        open={() => ({
          tier: "worker",
          result: new Promise((resolve) => {
            finish = resolve;
          }),
          cancel: () => finish(null),
        })}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Opening the committed snapshot was cancelled.",
    );
    unmount();
    let disposed = false;
    const results: ((value: AtlasOpening | null) => void)[] = [];
    const second = render(
      <AtlasView
        instrument="architecture"
        contextSuffix=""
        open={opener(diamond.text)}
        openFile={() => ({
          tier: "main thread",
          result: new Promise((resolve) => results.push(resolve)),
          cancel: () => {},
        })}
      />,
    );
    await screen.findByRole("table", { name: "Atlas nodes" });
    fireEvent.change(input(), { target: { files: [pinFile("{}", "slow.json")] } });
    expect(screen.getByRole("status", { name: "Atlas progress" }).textContent).toBe(
      "Verifying slow.json…",
    );
    fireEvent.change(input(), { target: { files: [pinFile("{}", "slower.json")] } });
    await act(async () => results[0]?.({ ok: false, reason: "superseded and refused" }));
    expect(screen.getByRole("status", { name: "Atlas progress" }).textContent).toBe(
      "Verifying slower.json…",
    );
    expect(screen.queryByRole("alert")).toBeNull();
    second.unmount();
    const session = await openAtlasSession(diamond.text, "late").result;
    if (session === null || !session.ok) throw new Error("expected a session");
    await act(async () =>
      results[1]?.({ ok: true, session: { ...session.session, dispose: () => (disposed = true) } }),
    );
    expect(disposed).toBe(true);
  });
});
