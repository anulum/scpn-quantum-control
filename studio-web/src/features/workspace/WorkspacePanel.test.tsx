// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workspace UI refusal tests

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it } from "vitest";
import { WorkspacePanel } from "./WorkspacePanel";

it("shows native storage absence and keeps successful save unavailable", async () => {
  render(<WorkspacePanel />);
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("IndexedDB unavailable"));
  const save = screen.getByRole("button", { name: "Save draft and revision references" }) as HTMLButtonElement;
  expect(save.disabled).toBe(true);
  expect(screen.queryByRole("definition", { name: "Saved workspace digest" })).toBeNull();
  expect(screen.getByText(/browser cache is not server storage/)).toBeTruthy();
});

it("requires preview and shows malformed archive refusal without claiming a saved revision", async () => {
  render(<WorkspacePanel />);
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("IndexedDB unavailable"));
  fireEvent.change(screen.getByLabelText("Workspace archive JSON"), { target: { value: '{"schema":"one","schema":"two"}' } });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("duplicate"));
  expect((screen.getByRole("button", { name: "Save draft and revision references" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("button", { name: "Export preview archive" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("button", { name: "Export saved archive" }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.queryByText("Workspace transaction committed.")).toBeNull();
});

it("refuses invalid project titles rather than manufacturing an empty valid draft", async () => {
  render(<WorkspacePanel />);
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("IndexedDB unavailable"));
  fireEvent.change(screen.getByLabelText("New project title"), { target: { value: "   " } });
  fireEvent.click(screen.getByRole("button", { name: "Create empty project" }));
  await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Project title"));
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe("");
});
