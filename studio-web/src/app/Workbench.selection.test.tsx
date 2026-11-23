// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — the workbench's shared selection

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import corpusText from "../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../browser-tests/workspaceFixture";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { admitWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { Workbench } from "./Workbench";
import { workbenchModules } from "./moduleRegistry";

const originalCrypto = globalThis.crypto;
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  Object.defineProperty(globalThis, "crypto", { value: originalCrypto, configurable: true });
  window.history.replaceState(null, "", "/");
});

async function preview(
  registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(workbenchModules),
) {
  const archive = await conformanceArchive(corpusText, false);
  const admitted = await admitWorkspaceArchive(archive.json, conformanceCodecs);
  render(
    <Workbench registry={registry} rawCodecs={conformanceCodecs}>
      {(_context, editors) => <section>Host overview{editors}</section>}
    </Workbench>,
  );
  fireEvent.click(screen.getAllByRole("link", { name: "Workspace" })[0] as HTMLElement);
  fireEvent.change(screen.getByLabelText("Workspace archive JSON"), {
    target: { value: archive.json },
  });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  const explorer = screen.getByRole("navigation", { name: "Project explorer" });
  return { admitted, explorer };
}

it("publishes an explorer source to the shared selection and links it to the Atlas", async () => {
  const { admitted, explorer } = await preview();
  const draft = requirePresent(
    (admitted.manifest.body["revision_refs"] as readonly { readonly sha256: string }[]).at(-1),
  );
  expect(screen.queryByRole("region", { name: "Selected object" })).toBeNull();
  const revisions = await within(explorer).findAllByRole(
    "button",
    { name: /^experiment_revision\.v1 ·/ },
    { timeout: 15_000 },
  );
  fireEvent.click(
    requirePresent(revisions.find((node) => node.textContent?.includes(draft.sha256.slice(0, 12)))),
  );
  const line = await screen.findByRole("region", { name: "Selected object" });
  expect(line.textContent).toContain(
    `revision:sha256:${draft.sha256} · content ${draft.sha256.slice(0, 12)} · selected in source · change 1`,
  );
  expect(
    within(line).getByRole("link", { name: "Show it in the Atlas" }).getAttribute("href"),
  ).toMatch(/^#\/atlas\/workspace/);
  const raw = await within(explorer).findAllByRole("button", { name: /^review_fixture\.v1 ·/ });
  fireEvent.click(requirePresent(raw[0]));
  expect(screen.getByRole("region", { name: "Selected object" }).textContent).toContain("change 1");
}, 30_000);

it("offers no Atlas link when no Atlas stage is registered", async () => {
  const { admitted, explorer } = await preview(
    createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(
      workbenchModules.filter((module) => module.id !== "atlas"),
    ),
  );
  const draft = requirePresent(
    (admitted.manifest.body["revision_refs"] as readonly { readonly sha256: string }[]).at(-1),
  );
  const revisions = await within(explorer).findAllByRole(
    "button",
    { name: /^experiment_revision\.v1 ·/ },
    { timeout: 15_000 },
  );
  fireEvent.click(
    requirePresent(revisions.find((node) => node.textContent?.includes(draft.sha256.slice(0, 12)))),
  );
  const line = await screen.findByRole("region", { name: "Selected object" });
  expect(within(line).queryByRole("link")).toBeNull();
}, 30_000);

it("names a selected parameter by its key alone", async () => {
  const { explorer } = await preview();
  const specs = await within(explorer).findAllByRole(
    "button",
    { name: /^parameter_spec\.v1 ·/ },
    { timeout: 15_000 },
  );
  fireEvent.click(requirePresent(specs[0]));
  const line = await screen.findByRole("region", { name: "Selected object" });
  expect(line.textContent).toMatch(
    /^parameter:workspace\.[0-9a-f-]+\/[A-Za-z0-9_.:-]+ · selected in source · change 1/,
  );
}, 30_000);
