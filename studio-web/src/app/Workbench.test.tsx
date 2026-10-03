// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original public workbench navigation

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import QuantumStudioPanel, { QuantumStudioPanel as NamedPanel } from "../QuantumStudioPanel";

beforeEach(() => { window.history.replaceState(null, "", "/"); });
afterEach(() => { cleanup(); window.history.replaceState(null, "", "/"); });

function visit(hash: string) {
  act(() => {
    window.history.pushState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
}

it("test_workbench_navigation_01: original named/default consumer retains every original card", async () => {
  expect(NamedPanel).toBe(QuantumStudioPanel);
  render(<QuantumStudioPanel rawCodecs={new Map()} />);
  expect(screen.getByRole("navigation", { name: "Workbench views" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Local workspace" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Capability catalogue" })).toBeTruthy();
  expect(screen.getByRole("region", { name: "Inspect evidence JSON" })).toBeTruthy();
  expect(screen.getByText("Differentiate support explorer")).toBeTruthy();
  expect(screen.getByText("Baseline scorecard")).toBeTruthy();
  await waitFor(() => expect(screen.getByRole("region", { name: "Local workspace" }).textContent).toContain("IndexedDB unavailable"));
});

it("test_workbench_navigation_02: fresh deep link and history events retain exact opaque identity", async () => {
  window.history.replaceState(null, "", "#/experiments?project=project%20%CE%B1&revision=r%2F1&snapshot=s%2B2");
  render(<QuantumStudioPanel />);
  const inspector = screen.getByRole("complementary", { name: "Workbench inspector" });
  expect(within(inspector).getByText("project α", { exact: true })).toBeTruthy();
  expect(within(inspector).getByText("r/1", { exact: true })).toBeTruthy();
  expect(within(inspector).getByText("s+2", { exact: true })).toBeTruthy();
  const atlas = screen.getByRole("navigation", { name: "Workbench views" }).querySelector('a[href^="#/atlas"]');
  expect(atlas?.getAttribute("href")).toBe("#/atlas?project=project+%CE%B1&revision=r%2F1&snapshot=s%2B2");
  visit("#/atlas?project=project%20%CE%B1&revision=r%2F1&snapshot=s%2B2");
  await screen.findByRole("heading", { name: "Atlas unavailable" });
  act(() => {
    window.history.replaceState(null, "", "#/experiments?project=project%20%CE%B1&revision=r%2F1&snapshot=s%2B2");
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  await screen.findByRole("heading", { name: "Experiments unavailable" });
  expect(within(inspector).getByText("r/1", { exact: true })).toBeTruthy();
  expect(inspector.textContent).toContain("not loaded or admitted by navigation");
});

it("test_workbench_navigation_03: embedded mode exposes keyboard targets and current breadcrumbs", async () => {
  render(<QuantumStudioPanel mode="embedded" />);
  expect(screen.getByText(/Embedded workbench/)).toBeTruthy();
  visit("#/atlas");
  await screen.findByRole("heading", { name: "Atlas unavailable" });
  const content = screen.getByRole("region", { name: "Workbench view" });
  expect(document.activeElement).toBe(content);
  expect(content.tabIndex).toBe(-1);
  expect(screen.getByRole("navigation", { name: "Breadcrumbs" }).textContent).toContain("Atlas");
  expect(screen.getByRole("navigation", { name: "Workbench views" }).querySelector('[aria-current="page"]')?.textContent).toBe("Atlas");
});

it("test_workbench_navigation_04: rejected route retains the original live editor and its bytes", async () => {
  render(<QuantumStudioPanel />);
  const editor = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  fireEvent.change(editor, { target: { value: '{"unsaved":"exact α bytes"}' } });
  visit("#/build?revision=%E0%A4%A");
  expect(screen.getByRole("alert").textContent).toContain("Route unavailable");
  visit("#/workspace");
  expect(screen.getByLabelText("Workspace archive JSON")).toBe(editor);
  expect(editor.value).toBe('{"unsaved":"exact α bytes"}');
  expect(screen.queryByText("Workspace transaction committed.")).toBeNull();
});

it("test_workbench_navigation_05: feature view is absent until navigation and reuses the original instrument", async () => {
  render(<QuantumStudioPanel />);
  expect(screen.queryByRole("heading", { name: "Build" })).toBeNull();
  visit("#/build/compile-recompute?project=p&revision=r");
  await screen.findByRole("heading", { name: "Build" });
  expect(screen.getByRole("button", { name: "Recompute in browser" })).toBeTruthy();
  expect(screen.queryByRole("region", { name: "Capability catalogue" })).toBeNull();
  expect(document.getElementById("/build/compile-recompute")?.tabIndex).toBe(-1);
  visit("#/results/program-ad-replay?project=p&revision=r");
  await screen.findByRole("heading", { name: "Results" });
  expect(screen.getByRole("region", { name: "Inspect evidence JSON" })).toBeTruthy();
});

it("operator profiles are reachable from the production shell without replacing the original workspace draft", async () => {
 render(<QuantumStudioPanel />);
 const editor=screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
 fireEvent.change(editor,{target:{value:'{"unsaved":"profile-navigation α"}'}});
 const navigation=screen.getByRole("navigation",{name:"Workbench context destinations"});
 expect(within(navigation).getByRole("link",{name:"Devices & Operations"}).getAttribute("href")).toBe("#/operations");
 visit("#/operations");
 await screen.findByRole("heading",{name:"Devices & Operations"});
 expect(within(navigation).getByRole("link",{name:"Devices & Operations"}).getAttribute("aria-current")).toBe("page");
 fireEvent.click(screen.getByRole("button",{name:"Open declared profiles"}));
 await screen.findByLabelText("Backend profile");
 expect(screen.getByRole("navigation",{name:"Workbench views"}).querySelectorAll("a")).toHaveLength(5);
 visit("#/workspace");expect(screen.getByLabelText("Workspace archive JSON")).toBe(editor);
 expect(editor.value).toBe('{"unsaved":"profile-navigation α"}');
});
