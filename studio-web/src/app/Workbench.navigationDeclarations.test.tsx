// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — compiled navigation declarations retain registered addresses

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { createModuleRegistry } from "../shared/contracts/module";
import type { ModuleCommand } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { Workbench } from "./Workbench";
import { workbenchModules } from "./moduleRegistry";

beforeEach(() => window.history.replaceState(null, "", "/"));
afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});

it("refuses a compiled command whose retained declaration changes to an unregistered stage after registry admission", () => {
  const command: ModuleCommand = {
    id: "source-navigation-lifetime",
    label: "Open original Atlas declaration",
    action: { kind: "navigate", stage: "atlas", instrument: null },
  };
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "navigation-lifetime",
      version: "1",
      requires: { verbs: [], schemas: [] },
      stages: [],
      commands: [command],
    },
  ]);
  render(<Workbench registry={registry}>{() => null}</Workbench>);
  const before = window.location.hash;
  Reflect.set(command.action, "stage", "unregistered-source-stage");
  fireEvent.click(screen.getByRole("button", { name: command.label }));
  expect(window.location.hash).toBe(before);
  expect(screen.queryByRole("heading", { name: "Atlas" })).toBeNull();
  expect(screen.getByRole("navigation", { name: "Workbench views" })).toBeTruthy();
});

it("refuses an application registry missing its original workspace destination instead of manufacturing a replacement", () => {
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(
    workbenchModules.filter((module) => module.id !== "workspace"),
  );
  expect(() => render(<Workbench registry={registry}>{() => null}</Workbench>)).toThrow(
    "Unregistered workbench stage",
  );
  expect(window.location.hash).toBe("");
});
