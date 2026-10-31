// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of the modules compiled into the workbench

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, expectTypeOf, it } from "vitest";
import { Workbench } from "./Workbench";
afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});
import { createModuleRegistry } from "../shared/contracts/module";
import type { StudioModule } from "../shared/contracts/module";
import { studioManifest } from "../shared/browser/data";
import { declaredBy, workbenchModules, workbenchRegistry } from "./moduleRegistry";
import type { WorkbenchServices } from "./moduleRegistry";
import type { WorkspaceViewServices } from "../shared/contracts/workbench";
import { parseWorkbenchRoute } from "./routing";
import type { WorkbenchView } from "./routing";

describe("workbenchRegistry", () => {
  it("admits every address the workbench had before it was modular, plus Compare", () => {
    expectTypeOf<WorkbenchView>().toEqualTypeOf<
      | "workspace"
      | "build"
      | "experiments"
      | "results"
      | "compare"
      | "evidence"
      | "atlas"
      | "operations"
    >();
    expect(
      [...workbenchRegistry.paths].map(([path, { view, instrument }]) => [path, view, instrument]),
    ).toEqual([
      ["/workspace", "workspace", null],
      ["/build", "build", null],
      ["/build/compile-recompute", "build", "compile-recompute"],
      ["/experiments", "experiments", null],
      ["/results", "results", null],
      ["/results/program-ad-replay", "results", "program-ad-replay"],
      ["/results/gradient-debugger", "results", "gradient-debugger"],
      ["/results/qec-resources", "results", "qec-resources"],
      ["/compare", "compare", null],
      ["/evidence", "evidence", null],
      ["/atlas", "atlas", null],
      ["/operations", "operations", null],
    ]);
  });

  it("orders the research loop after Workspace and keeps Atlas and operations as context", () => {
    expect(workbenchRegistry.stages.map(({ label, group }) => [label, group])).toEqual([
      ["Workspace", "primary"],
      ["Model", "primary"],
      ["Run", "primary"],
      ["Analyse", "primary"],
      ["Compare", "primary"],
      ["Evidence", "primary"],
      ["Atlas", "context"],
      ["Devices & Operations", "context"],
    ]);
  });

  it("renders the host overview on the Workspace stage through the real shell services", async () => {
    window.history.replaceState(null, "", "#/workspace");
    render(
      <Workbench>
        {(_context, editors) => (
          <section aria-label="Host overview">host overview{editors}</section>
        )}
      </Workbench>,
    );
    expect(screen.getByRole("region", { name: "Host overview" }).textContent).toContain(
      "host overview",
    );
    expect(await screen.findByRole("region", { name: "Capability catalogue" })).toBeTruthy();
  });

  it("requires compile independently for the Build stage and its source editor modules", () => {
    expect(workbenchRegistry.missing({ verbs: [], schemas: [] })).toEqual([
      { module: "build", kind: "verb", name: "compile" },
      { module: "programs", kind: "verb", name: "compile" },
      { module: "compiler", kind: "verb", name: "compile" },
    ]);
  });

  it("is satisfied by the committed manifest and by nothing a refused manifest declares", () => {
    const declared = declaredBy(studioManifest);
    expect(declared.verbs).toContain("compile");
    expect(declared.schemas.length).toBeGreaterThan(0);
    expect(workbenchRegistry.missing(declared)).toEqual([]);
    const refused = declaredBy({ ok: false, reason: "guard failed" });
    expect(refused).toEqual({ verbs: [], schemas: [] });
    expect(workbenchRegistry.missing(refused)).toEqual([
      { module: "build", kind: "verb", name: "compile" },
      { module: "programs", kind: "verb", name: "compile" },
      { module: "compiler", kind: "verb", name: "compile" },
    ]);
  });

  it("gains a destination from one registry entry and loses none of its own", () => {
    const fixture: StudioModule<WorkbenchServices, WorkspaceViewServices> = {
      id: "fixture",
      apiVersion: "2.0",
      requires: { verbs: ["mitigate"], schemas: [] },
      stages: [
        {
          id: "fixture",
          label: "Fixture",
          group: "context",
          instruments: ["probe"],
          render: () => "fixture",
        },
      ],
    };
    const extended = createModuleRegistry([...workbenchModules, fixture]);
    expect([...extended.paths].slice(0, workbenchRegistry.paths.size)).toEqual([
      ...workbenchRegistry.paths,
    ]);
    expect(extended.paths.get("/fixture/probe")).toEqual({ view: "fixture", instrument: "probe" });
    expect(extended.missing({ verbs: ["compile"], schemas: [] })).toEqual([
      { module: "fixture", kind: "verb", name: "mitigate" },
    ]);
    // The build itself does not contain the fixture: its address stays refused.
    expect(parseWorkbenchRoute("#/fixture")).toEqual({
      ok: false,
      reason: "Unsupported workbench path",
    });
  });
});
