// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — whole workbench accessibility regressions

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import QuantumStudioPanel from "../QuantumStudioPanel";

beforeEach(() => { window.history.replaceState(null, "", "/"); });
afterEach(() => { cleanup(); window.history.replaceState(null, "", "/"); });

it("test_workbench_accessibility_01: the public shell exposes keyboard help and a skip control without touching draft bytes", () => {
  render(<QuantumStudioPanel />);
  const editor = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  fireEvent.change(editor, { target: { value: '{"unsaved":"exact accessibility α"}' } });
  const skip = screen.getByRole("button", { name: "Skip to current view" });
  skip.focus();
  fireEvent.click(skip);
  expect(document.activeElement).toBe(screen.getByRole("region", { name: "Workbench view" }));
  expect(screen.getByRole("button", { name: "Keyboard help" })).toBeTruthy();
  expect(editor.value).toBe('{"unsaved":"exact accessibility α"}');
  expect(window.location.hash).toBe("");
});

it("test_workbench_accessibility_03: malformed navigation exposes a readable refusal and preserves the original editor", () => {
  render(<QuantumStudioPanel />);
  const editor = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  fireEvent.change(editor, { target: { value: '{"private":"synthetic local-only"}' } });
  act(() => {
    window.history.pushState(null, "", "#/build?revision=%E0%A4%A");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
  expect(screen.getByRole("alert").textContent).toContain("Route unavailable");
  expect(screen.getByRole("button", { name: "Keyboard help" })).toBeTruthy();
  expect(document.activeElement).toBe(screen.getByRole("region", { name: "Workbench view" }));
  expect(screen.queryByRole("textbox", { name: "Workspace archive JSON" })).toBeNull();
  act(() => {
    window.history.pushState(null, "", "#/workspace");
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
  expect(screen.getByLabelText("Workspace archive JSON")).toBe(editor);
  expect(editor.value).toBe('{"private":"synthetic local-only"}');
});

it("test_workbench_accessibility_01: native fragment focus loss recovers the view and preserves the next action", () => {
  const component = render(<QuantumStudioPanel />);
  const view = screen.getByRole("region", { name: "Workbench view" });
  view.blur();
  expect(document.activeElement).toBe(document.body);
  act(() => { window.dispatchEvent(new HashChangeEvent("hashchange")); });
  expect(document.activeElement).toBe(view);
  const next = screen.getByRole("button", { name: "Skip to current view" });
  next.focus();
  act(() => { window.dispatchEvent(new HashChangeEvent("hashchange")); });
  expect(document.activeElement).toBe(next);
  component.unmount();
  act(() => { window.dispatchEvent(new HashChangeEvent("hashchange")); });
  expect(document.activeElement).toBe(document.body);
});
