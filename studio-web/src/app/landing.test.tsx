// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Workspace landing leads to work; evidence has its own destination

import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import QuantumStudioPanel from "../QuantumStudioPanel";

afterEach(() => {
  cleanup();
  window.history.replaceState(null, "", "/");
});

/**
 * Follow a workbench address the way a link activation does.
 *
 * @param href - Fragment address to open.
 */
async function open(href: string): Promise<void> {
  await act(async () => {
    window.history.pushState(null, "", href);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
}

it("test_landing_01: the former landing instruments are reachable at their existing deep links", async () => {
  window.history.replaceState(null, "", "#/workspace?project=project%20%CE%B1");
  render(<QuantumStudioPanel />);
  const work = within(screen.getByRole("navigation", { name: "Continue working" }));
  const links = work.getAllByRole("link");
  expect(links.map((link) => [link.textContent, link.getAttribute("href")])).toEqual([
    ["Build a model", "#/build?project=project+%CE%B1"],
    ["Run an experiment", "#/experiments?project=project+%CE%B1"],
    ["Compare results", "#/compare?project=project+%CE%B1"],
    ["Evidence catalogue", "#/evidence?project=project+%CE%B1"],
  ]);
  expect(screen.getByRole("region", { name: "Capability catalogue" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Inspect evidence JSON" })).toBeTruthy();
  expect(screen.queryByText("Differentiate support explorer")).toBeNull();
  expect(screen.queryByText("Baseline scorecard")).toBeNull();

  await open(links[3]?.getAttribute("href") as string);
  await screen.findByRole("heading", { name: "Evidence catalogue" }, { timeout: 10000 });
  const evidence = within(screen.getByRole("article", { name: "Committed evidence catalogue" }));
  expect(evidence.getByText("Capabilities")).toBeTruthy();
  expect(evidence.getByText("Differentiate support explorer")).toBeTruthy();
  expect(evidence.getByText("Gradient-plan explanation")).toBeTruthy();
  expect(evidence.getByText("Baseline scorecard")).toBeTruthy();
  expect(screen.queryByRole("navigation", { name: "Continue working" })).toBeNull();

  for (const address of ["/build/compile-recompute", "/results/program-ad-replay"]) {
    await open(`#${address}`);
    await waitFor(() => expect(document.getElementById(address)).not.toBeNull(), {
      timeout: 10000,
    });
    expect(
      screen
        .getByRole("region", { name: "Workbench view" })
        .contains(document.getElementById(address)),
    ).toBe(true);
  }
}, 30000);
