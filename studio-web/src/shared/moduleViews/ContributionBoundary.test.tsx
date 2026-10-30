// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — one failing or loading contribution leaves its slot working

import { cleanup, render, screen } from "@testing-library/react";
import { lazy } from "react";
import type { ReactElement } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MODULE_API_VERSION, createModuleRegistry } from "../contracts/module";
import type { ModuleDocument, StudioModule } from "../contracts/module";
import { ContributionBoundary } from "./ContributionBoundary";
import { ModuleDocumentViews } from "./ModuleDocumentViews";
import { ModuleExplorerView } from "./ModuleExplorerView";
import { ModuleStageEditors } from "./ModuleStageEditors";

type Services = { readonly word: string };

const declared = { verbs: [], schemas: [] } as const;
const services: Services = { word: "shell" };
const notes: ModuleDocument = { schema: "test.notes.v1", sha256: "a".repeat(64), content: "{}" };
const later: ModuleDocument = { ...notes, sha256: "b".repeat(64), content: '{"n":1}' };

function broken(): never {
  throw new Error("contribution defect");
}

/** A module whose single contribution in every slot either works or throws. */
function contributor(id: string, fails: boolean | ((document: ModuleDocument) => boolean)) {
  const failsFor = (document: ModuleDocument) =>
    typeof fails === "function" ? fails(document) : fails;
  const module: StudioModule<Services> = {
    id,
    apiVersion: MODULE_API_VERSION,
    requires: { verbs: [], schemas: [] },
    stages: [],
    editors: [
      {
        kind: "stage",
        id: `${id}-editor`,
        label: `${id} editor`,
        stage: "stage",
        render: (shell) => (fails === true ? broken() : <p>{`${id} editor for ${shell.word}`}</p>),
      },
      {
        kind: "document",
        id: `${id}-view`,
        label: `${id} view`,
        schemas: [notes.schema],
        render: (_shell, document) =>
          failsFor(document) ? broken() : <p>{`${id} view of ${document.content}`}</p>,
      },
    ],
    explorer: [
      {
        id: `${id}-tree`,
        label: `${id} tree`,
        kinds: ["note"],
        nodes: () =>
          fails === true
            ? broken()
            : [{ id: "n", kind: "note", label: `${id} note`, document: notes }],
      },
    ],
  };
  return module;
}

const registry = createModuleRegistry([contributor("sound", false), contributor("faulty", true)]);

beforeEach(() => {
  // React reports every contained render error on the console.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it("contains a failing stage editor and keeps its neighbour", () => {
  render(
    <ModuleStageEditors
      registry={registry}
      stage="stage"
      services={services}
      declared={declared}
    />,
  );
  expect(screen.getByText("sound editor for shell")).toBeTruthy();
  const failure = screen.getByRole("alert", { name: "faulty editor failed" });
  expect(failure.textContent).toContain("The faulty module could not show this part.");
  expect(failure.textContent).not.toContain("contribution defect");
});

it("contains a failing document view and retries it for another document", () => {
  const flaky = createModuleRegistry([
    contributor("sound", false),
    contributor("flaky", (document) => document.sha256 === notes.sha256),
  ]);
  const view = (document: ModuleDocument) => (
    <ModuleDocumentViews
      registry={flaky}
      slot="editors"
      document={document}
      services={services}
      declared={declared}
    />
  );
  const { rerender } = render(view(notes));
  expect(screen.getByText("sound view of {}")).toBeTruthy();
  expect(screen.getByRole("alert", { name: "flaky view failed" })).toBeTruthy();
  rerender(view(later));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByText('flaky view of {"n":1}')).toBeTruthy();
});

it("contains an explorer whose nodes cannot be read", () => {
  render(
    <ModuleExplorerView
      registry={registry}
      services={services}
      declared={declared}
      documents={[notes]}
      selected={null}
      select={() => undefined}
    />,
  );
  expect(screen.getByRole("button", { name: "sound note" })).toBeTruthy();
  expect(screen.getByRole("alert", { name: "faulty tree failed" })).toBeTruthy();
});

it("shows a loading contribution in place and then its content", async () => {
  let resolve: (value: { default: () => ReactElement }) => void = () => undefined;
  const Late = lazy(
    () =>
      new Promise<{ default: () => ReactElement }>((done) => {
        resolve = done;
      }),
  );
  render(
    <>
      <ContributionBoundary label="Late view" owner="late" render={() => <Late />} />
      <p>neighbour</p>
    </>,
  );
  expect(screen.getByRole("status").textContent).toBe("Loading Late view…");
  expect(screen.getByText("neighbour")).toBeTruthy();
  resolve({ default: () => <p>late content</p> });
  expect(await screen.findByText("late content")).toBeTruthy();
});

it("reports lazy code that fails to load as a failure of that contribution", async () => {
  const Missing = lazy(() => Promise.reject(new Error("chunk refused")));
  render(<ContributionBoundary label="Missing view" owner="missing" render={() => <Missing />} />);
  expect(await screen.findByRole("alert", { name: "Missing view failed" })).toBeTruthy();
});
