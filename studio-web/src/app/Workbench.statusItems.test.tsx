// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — title-bar status items come from available modules

import { webcrypto } from "node:crypto";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { MODULE_API_VERSION, createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { Workbench } from "./Workbench";
import { studioManifest } from "../shared/browser/data";
import { declaredBy, workbenchModules } from "./moduleRegistry";

const committedDeclarations = declaredBy(studioManifest);

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

function chips(): string[] {
  const list = screen.getByRole("list", { name: "Execution boundary" });
  return within(list)
    .getAllByRole("listitem")
    .map((item) => `${item.className}|${item.textContent}`);
}

it("shows the Build kernel item while Build is available, and the shell's own limit always", () => {
  render(<Workbench declared={committedDeclarations}>{() => null}</Workbench>);
  expect(chips()).toEqual([
    "qsp-chip|Browser kernel · WASM",
    "qsp-chip qsp-chip-boundary|Remote execution off",
  ]);
});

it("hides a status item whose module the installation cannot run", () => {
  render(
    <Workbench declared={{ verbs: [], schemas: committedDeclarations.schemas }}>
      {() => null}
    </Workbench>,
  );
  expect(chips()).toEqual(["qsp-chip qsp-chip-boundary|Remote execution off"]);
});

it("draws boundary items from modules without the active mark", () => {
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "offline-limits",
      apiVersion: MODULE_API_VERSION,
      requires: { verbs: [], schemas: [] },
      stages: [],
      statusItems: [{ id: "offline", label: "Offline profiles only", tone: "boundary" }],
    },
  ]);
  render(
    <Workbench registry={registry} declared={committedDeclarations}>
      {() => null}
    </Workbench>,
  );
  const list = screen.getByRole("list", { name: "Execution boundary" });
  const offline = within(list).getByText("Offline profiles only");
  expect(offline.className).toBe("qsp-chip qsp-chip-boundary");
  expect(offline.querySelector(".qsp-dot")).toBeNull();
  expect(within(list).getByText("Browser kernel · WASM").querySelector(".qsp-dot")).not.toBeNull();
});
