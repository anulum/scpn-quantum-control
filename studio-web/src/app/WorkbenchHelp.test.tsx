// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — keyboard-help event wiring; native dialog behaviour has browser ownership

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WorkbenchHelp } from "./WorkbenchHelp";

afterEach(cleanup);

function dialogEvents(dialog: HTMLDialogElement) {
  const show = vi.fn(() => { dialog.open = true; });
  const close = vi.fn(() => { dialog.open = false; fireEvent(dialog, new Event("close")); });
  Object.defineProperty(dialog, "showModal", { value: show, configurable: true });
  Object.defineProperty(dialog, "close", { value: close, configurable: true });
  return { show, close };
}

it("invokes the native modal and returns focus to the original help control on close", () => {
  render(<WorkbenchHelp routeKey="#/workspace" />);
  const dialog = screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
  const native = dialogEvents(dialog);
  const origin = screen.getByRole("button", { name: "Keyboard help" });
  origin.focus();
  fireEvent.click(origin);
  expect(native.show).toHaveBeenCalledOnce();
  expect(screen.getByRole("dialog", { name: "Keyboard help" })).toBe(dialog);
  const close = screen.getByRole("button", { name: "Close keyboard help" });
  close.focus();
  expect(fireEvent.keyDown(close, { key: "Tab" })).toBe(false);
  expect(document.activeElement).toBe(close);
  expect(fireEvent.keyDown(close, { key: "Tab", shiftKey: true })).toBe(false);
  expect(fireEvent.keyDown(close, { key: "Escape" })).toBe(true);
  fireEvent.click(close);
  expect(native.close).toHaveBeenCalledOnce();
  expect(document.activeElement).toBe(origin);
  expect(dialog.open).toBe(false);
});

it("closes help on a changed route without stealing the new route focus", () => {
  const component = render(<><WorkbenchHelp routeKey="#/workspace" /><button type="button">New route focus</button></>);
  const dialog = screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
  const native = dialogEvents(dialog);
  fireEvent.click(screen.getByRole("button", { name: "Keyboard help" }));
  const target = screen.getByRole("button", { name: "New route focus" });
  target.focus();
  component.rerender(<><WorkbenchHelp routeKey="#/results" /><button type="button">New route focus</button></>);
  expect(native.close).toHaveBeenCalledOnce();
  expect(document.activeElement).toBe(target);
});

it("supports native Escape close events and can reopen after navigation", () => {
  const component = render(<WorkbenchHelp routeKey="#/workspace" />);
  const dialog = screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
  dialogEvents(dialog);
  component.rerender(<WorkbenchHelp routeKey="#/build" />);
  const origin = screen.getByRole("button", { name: "Keyboard help" });
  fireEvent.click(origin);
  dialog.open = false;
  fireEvent(dialog, new Event("close"));
  expect(document.activeElement).toBe(origin);
  expect(screen.getByText(/local match does not certify a scientific claim/)).toBeTruthy();
});

it("a delayed native close event preserves focus on the user's next action", () => {
  render(<><WorkbenchHelp routeKey="#/workspace" /><button type="button">Next action</button></>);
  const dialog = screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
  dialogEvents(dialog);
  fireEvent.click(screen.getByRole("button", { name: "Keyboard help" }));
  dialog.open = false;
  const next = screen.getByRole("button", { name: "Next action" });
  next.focus();
  fireEvent(dialog, new Event("close"));
  expect(document.activeElement).toBe(next);
});

it("a queued old close event cannot move focus out of a reopened modal", () => {
  render(<WorkbenchHelp routeKey="#/workspace" />);
  const dialog = screen.getByRole("dialog", { hidden: true }) as HTMLDialogElement;
  dialogEvents(dialog);
  fireEvent.click(screen.getByRole("button", { name: "Keyboard help" }));
  dialog.open = false;
  fireEvent.click(screen.getByRole("button", { name: "Keyboard help" }));
  const close = screen.getByRole("button", { name: "Close keyboard help" });
  close.focus();
  fireEvent(dialog, new Event("close"));
  expect(dialog.open).toBe(true);
  expect(document.activeElement).toBe(close);
});
