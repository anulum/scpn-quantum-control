// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original archive identity projection

import { webcrypto } from "node:crypto";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import corpus from "../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive } from "../../browser-tests/workspaceFixture";
import { previewWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { writeJson } from "../shared/contracts";
import { emptyWorkbenchContext } from "./routing";
import { WorkbenchInspector } from "./WorkbenchInspector";

// Actual Node WebCrypto qualifies previews; native IndexedDB integration is exercised separately.
beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("retains requested context without inventing an admitted saved workspace", () => {
  render(<WorkbenchInspector context={{ project: "requested α", revision: "unresolved r", snapshot: "unresolved s" }} saved={null} />);
  const inspector = screen.getByRole("complementary", { name: "Workbench inspector" });
  expect(inspector.textContent).toContain("requested α");
  expect(inspector.textContent).toContain("unresolved r");
  expect(inspector.textContent).toContain("unresolved s");
  expect(inspector.textContent).toContain("No admitted saved workspace");
  expect(screen.queryByText("Admitted saved project")).toBeNull();
});

it("projects every exact admitted fixture digest while leaving requested identity separate", async () => {
  const preview = await conformanceArchive(corpus, false);
  render(<WorkbenchInspector context={{ project: "another project", revision: "unresolved revision", snapshot: null }} saved={{ preview, durability: "browser-cache-export-required" }} />);
  expect(screen.getByText(preview.projectId)).toBeTruthy();
  expect(screen.getByText(preview.workspaceHash)).toBeTruthy();
  expect(screen.getByText(preview.archiveDigest)).toBeTruthy();
  expect(screen.getByText(preview.documentHashes.join(" · "))).toBeTruthy();
  expect(screen.getByText("another project")).toBeTruthy();
  expect(screen.getByText("unresolved revision")).toBeTruthy();
  expect(screen.getByText("Not selected")).toBeTruthy();
  expect(screen.queryByText(/revision verified|snapshot loaded/i)).toBeNull();
});

it("projects an actually admitted empty archive without manufacturing revision documents", async () => {
  const preview = await previewWorkspaceArchive(writeJson({ schema: "quantum_workspace_archive.v1", manifest: {
    schema: "quantum_workspace.v1", extensions: {}, body: { project_id: "00000000-0000-4000-8000-000000000001", revision_refs: [], draft_ref: null, created_at: "2026-09-30T00:00:00Z", updated_at: "2026-09-30T00:00:00Z", artefact_refs: [] },
  }, members: [], parameter_units: {} }));
  render(<WorkbenchInspector context={emptyWorkbenchContext} saved={{ preview, durability: "browser-cache-export-required" }} />);
  expect(screen.getByText("No revision documents")).toBeTruthy();
  expect(screen.getAllByText("Not selected")).toHaveLength(3);
  expect(screen.getByText(preview.archiveDigest)).toBeTruthy();
});
