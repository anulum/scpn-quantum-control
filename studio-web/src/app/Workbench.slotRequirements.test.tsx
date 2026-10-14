// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — missing installation requirements across actual compiled shell slots

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { unownedContributionsModule } from "../../test-fixtures/unowned-contributions";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { Workbench } from "./Workbench";
import { workbenchModules } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
  window.history.replaceState(null, "", "/");
});

it("names missing verb and schema in each compiled slot without invoking its renderer or rewriting admitted source", async () => {
  const plan = await stationaryPlan();
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      ...unownedContributionsModule,
      requires: { verbs: ["not-installed-view"], schemas: ["not-installed.v1"] },
    },
  ]);
  render(
    <Workbench registry={registry}>
      {(_context, editors) => <section>Host overview{editors}</section>}
    </Workbench>,
  );
  const stage = screen.getByRole("region", { name: "Unowned stage editor unavailable" });
  expect(within(stage).getByText("not-installed-view")).toBeTruthy();
  expect(within(stage).getByText("not-installed.v1")).toBeTruthy();
  const explorer = screen.getByRole("heading", {
    name: "Unowned explorer unavailable",
  }).parentElement;
  expect(explorer?.textContent).toContain("verb not-installed-view");
  expect(explorer?.textContent).toContain("schema not-installed.v1");
  const commands = within(screen.getByRole("navigation", { name: "Workbench commands" }));
  expect(commands.getByRole("button", { name: "Open unowned probe" })).toMatchObject({
    disabled: true,
  });
  const editor = screen.getByLabelText("Workspace archive JSON");
  await waitFor(() => expect(editor).toMatchObject({ disabled: false }));
  fireEvent.change(editor, { target: { value: plan.sourceJson } });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Workspace source admission" }).textContent).toBe(
      "Original workspace sources admitted",
    ),
  );
  const settings = screen.getByRole("region", { name: "Unowned settings unavailable" });
  expect(within(settings).getByText("not-installed-view")).toBeTruthy();
  expect(within(settings).getByText("not-installed.v1")).toBeTruthy();
  const sources = within(screen.getByRole("navigation", { name: "Project explorer" }));
  fireEvent.click(sources.getByRole("button", { name: /^quantum_workspace_archive\.v1 ·/ }));
  const inspector = await screen.findByRole("region", { name: "Unowned inspector unavailable" });
  expect(within(inspector).getByText("not-installed-view")).toBeTruthy();
  expect(within(inspector).getByText("not-installed.v1")).toBeTruthy();
  expect(screen.queryByText("An unowned contribution was invoked")).toBeNull();
  expect((editor as HTMLTextAreaElement).value).toBe(plan.sourceJson);
  fireEvent.change(commands.getByLabelText("Search commands"), {
    target: { value: "no-matching-installed-command" },
  });
  expect(commands.getByText("No matching commands")).toBeTruthy();
});
