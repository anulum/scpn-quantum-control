// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real React error isolation and recovery

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { RouteBoundary } from "./RouteBoundary";

function FailingView({ fail }: { fail: boolean }) {
  if (fail) throw new Error("Transport exception details must not enter the view");
  return <p>Actual recovered route content</p>;
}
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

it("contains a real rendering failure, preserves its sibling draft and recovers only on route remount", () => {
  const logged = vi.spyOn(console, "error").mockImplementation(() => undefined);
  const content = (key: string, fail: boolean) => <><textarea aria-label="Persistent editor" defaultValue="original" /><RouteBoundary key={key} workspaceHref="#/workspace?project=p&revision=r"><FailingView fail={fail} /></RouteBoundary></>;
  const { rerender } = render(content("results", false));
  const editor = screen.getByLabelText("Persistent editor") as HTMLTextAreaElement;
  fireEvent.change(editor, { target: { value: "unsaved exact bytes α" } });
  rerender(content("results", true));
  expect(screen.getByRole("alert").textContent).toContain("Your workspace and editor are retained");
  expect(screen.getByRole("link", { name: "Return to Workspace" }).getAttribute("href")).toBe("#/workspace?project=p&revision=r");
  expect(screen.queryByText("Transport exception details must not enter the view")).toBeNull();
  expect(screen.getByLabelText("Persistent editor")).toBe(editor);
  expect(editor.value).toBe("unsaved exact bytes α");
  expect(logged).toHaveBeenCalled();
  rerender(content("results", false));
  expect(screen.getByRole("alert")).toBeTruthy();
  rerender(content("workspace", false));
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.getByText("Actual recovered route content")).toBeTruthy();
  expect(editor.value).toBe("unsaved exact bytes α");
});
