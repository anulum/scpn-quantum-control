// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — settings module removal retains the admitted workspace

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
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

it("retains the original archive and reports the missing settings view after the compiled settings module is removed", async () => {
  const plan = await stationaryPlan();
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(
    workbenchModules.filter((module) => module.id !== "settings"),
  );
  render(<Workbench registry={registry}>{() => null}</Workbench>);
  const editor = screen.getByLabelText<HTMLTextAreaElement>("Workspace archive JSON");
  await waitFor(() => expect(editor).toMatchObject({ disabled: false }));
  fireEvent.change(editor, { target: { value: plan.sourceJson } });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Workspace source admission" }).textContent).toBe(
      "Original workspace sources admitted",
    ),
  );
  expect(
    screen.getByText(
      "No settings view is registered for quantum_workspace_archive.v1. The original document is retained.",
    ),
  ).toBeTruthy();
  expect(screen.getByRole("link", { name: "Model" }).getAttribute("href")).toBe("#/build");
  expect(editor.value).toBe(plan.sourceJson);
});
