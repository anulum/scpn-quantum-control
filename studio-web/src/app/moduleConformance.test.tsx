// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — conformance every registered workbench module must keep

import { webcrypto } from "node:crypto";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { studioManifest } from "../shared/browser/data";
import { createModuleRegistry, isCompatibleApiVersion } from "../shared/contracts/module";
import type { StudioModule } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { Workbench } from "./Workbench";
import { declaredBy, workbenchModules } from "./moduleRegistry";

type RegisteredModule = StudioModule<WorkbenchServices, WorkspaceViewServices>;

const committed = declaredBy(studioManifest);
/** The module whose stage is the shell's home: breadcrumb root and recovery target. */
const HOME_MODULE = "workspace";
const modules: readonly RegisteredModule[] = workbenchModules;
const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");

beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  // React reports a contained contribution failure on the console; a failure is asserted below.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
  window.history.replaceState(null, "", "/");
});

/** Open an address in a fresh workbench and wait for its lazy view to settle. */
async function open(
  address: string,
  registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(modules),
  declared = committed,
): Promise<HTMLElement> {
  window.history.replaceState(null, "", address);
  render(
    <Workbench registry={registry} declared={declared}>
      {(_context, editors) => editors}
    </Workbench>,
  );
  await act(() => vi.dynamicImportSettled());
  return screen.getByRole("region", { name: "Workbench view" });
}

/** Failures the shell would show for a route or a contribution. */
function failures(view: HTMLElement): string[] {
  return within(view)
    .queryAllByRole("alert")
    .map((alert) => alert.textContent ?? "")
    .filter((text) => /View unavailable|could not show this part|Route unavailable/.test(text));
}

describe.each(modules.map((module) => [module.id, module] as const))("module %s", (_id, module) => {
  it("declares a compatible contract and only valid, registered contributions", () => {
    expect(isCompatibleApiVersion(module.apiVersion)).toBe(true);
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([module]);
    for (const stage of module.stages) expect(registry.stage(stage.id)).toBe(stage);
  });

  it.each(module.stages.map((stage) => [stage.id, stage.label] as const))(
    "opens stage %s with its label in the breadcrumb and no failure",
    async (stage, label) => {
      const view = await open(`#/${stage}`);
      expect(screen.getByRole("navigation", { name: "Breadcrumbs" }).textContent).toContain(label);
      expect(failures(view)).toEqual([]);
    },
  );

  it("stays named, not crashed, when the installation declares nothing", async () => {
    const needs = module.requires.verbs.length + module.requires.schemas.length;
    for (const stage of module.stages) {
      const view = await open(`#/${stage.id}`, undefined, { verbs: [], schemas: [] });
      expect(failures(view)).toEqual([]);
      if (needs > 0)
        expect(
          within(view).getByRole("heading", { name: `${stage.label} unavailable` }),
        ).toBeTruthy();
      cleanup();
    }
  });

  it("can be removed while every other stage keeps working and its addresses close", async () => {
    const rest = modules.filter((other) => other !== module);
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(rest);
    if (module.id === HOME_MODULE) {
      // The shell's home stage is part of its contract; leaving it out is refused loudly.
      expect(() =>
        render(
          <Workbench registry={registry} declared={committed}>
            {() => null}
          </Workbench>,
        ),
      ).toThrow("Unregistered workbench stage");
      return;
    }
    for (const stage of module.stages) {
      const view = await open(`#/${stage.id}`, registry);
      expect(within(view).getByRole("heading", { name: "Route unavailable" })).toBeTruthy();
      cleanup();
    }
    const view = await open("#/workspace", registry);
    expect(failures(view)).toEqual([]);
    const rail = screen.getByRole("navigation", { name: "Workbench views" });
    for (const stage of module.stages)
      expect(within(rail).queryByRole("link", { name: stage.label })).toBeNull();
  });
});
