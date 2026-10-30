// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — tests of a module the installation cannot serve

import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { Workbench } from "./Workbench";

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});

function visit(hash: string) {
  act(() => {
    window.history.replaceState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
}

it("keeps a stage visible and names what the installation does not declare", async () => {
  render(<Workbench declared={{ verbs: [], schemas: [] }}>{() => <p>host overview</p>}</Workbench>);
  expect(screen.getByText("host overview")).toBeTruthy();
  visit("#/build?project=p");
  const view = screen.getByRole("region", { name: "Workbench view" });
  expect(within(view).getByRole("heading", { name: "Model unavailable" })).toBeTruthy();
  expect(within(view).getByRole("listitem").textContent).toBe("verb compile");
  expect(within(view).getByRole("link", { name: "Open Workspace" }).getAttribute("href")).toBe(
    "#/workspace?project=p",
  );
  const navigation = screen.getByRole("navigation", { name: "Workbench views" });
  expect(within(navigation).getByRole("link", { name: "Model" }).getAttribute("aria-current")).toBe(
    "page",
  );
  // A stage of a module that requires nothing is served under the same declarations.
  visit("#/atlas");
  expect(await screen.findByRole("heading", { name: "Atlas unavailable" })).toBeTruthy();
  expect(screen.queryByText("This installation does not declare what the view needs:")).toBeNull();
});

it("serves the stage once the installation declares the verb", async () => {
  render(
    <Workbench declared={{ verbs: ["compile"], schemas: [] }}>
      {() => <p>host overview</p>}
    </Workbench>,
  );
  visit("#/build");
  expect(screen.queryByRole("heading", { name: "Model unavailable" })).toBeNull();
  expect(await screen.findByRole("heading", { name: "Compile recompute" })).toBeTruthy();
});
