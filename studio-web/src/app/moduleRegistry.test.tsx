// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of the modules compiled into the workbench

import { describe, expect, it } from "vitest";
import { createModuleRegistry } from "../shared/contracts/module";
import type { StudioModule } from "../shared/contracts/module";
import { studioManifest } from "../panel/data";
import { declaredBy, workbenchModules, workbenchRegistry } from "./moduleRegistry";
import type { WorkbenchServices } from "./moduleRegistry";
import { parseWorkbenchRoute } from "./routing";

describe("workbenchRegistry", () => {
  it("admits exactly the addresses the workbench had before it was modular", () => {
    expect(
      [...workbenchRegistry.paths].map(([path, { view, instrument }]) => [path, view, instrument]),
    ).toEqual([
      ["/workspace", "workspace", null],
      ["/build", "build", null],
      ["/build/compile-recompute", "build", "compile-recompute"],
      ["/experiments", "experiments", null],
      ["/results", "results", null],
      ["/results/program-ad-replay", "results", "program-ad-replay"],
      ["/atlas", "atlas", null],
      ["/operations", "operations", null],
    ]);
  });

  it("keeps the navigation labels, their order and their groups", () => {
    expect(workbenchRegistry.stages.map(({ label, group }) => [label, group])).toEqual([
      ["Workspace", "primary"],
      ["Build", "primary"],
      ["Experiments", "primary"],
      ["Results", "primary"],
      ["Atlas", "primary"],
      ["Devices & Operations", "context"],
    ]);
  });

  it("renders the host overview on the Workspace stage through the services alone", () => {
    const services = { overview: () => "host overview" } as unknown as WorkbenchServices;
    expect(workbenchRegistry.stage("workspace")?.render(services, null)).toBe("host overview");
  });

  it("requires the compile verb for Build and nothing else", () => {
    expect(workbenchRegistry.missing({ verbs: [], schemas: [] })).toEqual([
      { module: "build", kind: "verb", name: "compile" },
    ]);
  });

  it("is satisfied by the committed manifest and by nothing a refused manifest declares", () => {
    const declared = declaredBy(studioManifest);
    expect(declared.verbs).toContain("compile");
    expect(declared.schemas.length).toBeGreaterThan(0);
    expect(workbenchRegistry.missing(declared)).toEqual([]);
    const refused = declaredBy({ ok: false, reason: "guard failed" });
    expect(refused).toEqual({ verbs: [], schemas: [] });
    expect(workbenchRegistry.missing(refused)).toHaveLength(1);
  });

  it("gains a destination from one registry entry and loses none of its own", () => {
    const fixture: StudioModule<WorkbenchServices> = {
      id: "fixture",
      version: "1",
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
