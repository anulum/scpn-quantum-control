// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — compiled explorer nodes remain bound to actual admitted sources

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { admitWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { Workbench } from "./Workbench";
import { workbenchModules } from "./moduleRegistry";

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

it("does not open duplicate, undeclared-kind or different admitted source nodes supplied by a compiled explorer", async () => {
  const plan = await stationaryPlan();
  const other = await admitWorkspaceArchive(`${plan.sourceJson}\n`, localExperimentCodecs);
  const orphan = {
    schema: other.preview.schema,
    sha256: other.preview.archiveDigest,
    content: other.preview.json,
  };
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "source-explorer-probe",
      apiVersion: "2.0",
      requires: { verbs: [], schemas: [] },
      stages: [],
      explorer: [
        {
          id: "source-explorer-probe",
          label: "Compiled source binding probe",
          kinds: ["allowed-source"],
          nodes: (services) => {
            if (services.documents.length === 0) return [];
            const document = requirePresent(
              services.documents.find((source) => source.schema === "quantum_workspace_archive.v1"),
            );
            return [
              { id: "duplicate", label: "Duplicate first", kind: "allowed-source", document },
              { id: "duplicate", label: "Duplicate second", kind: "allowed-source", document },
              { id: "wrong-kind", label: "Undeclared kind", kind: "undeclared-kind", document },
              {
                id: "other-source",
                label: "Other actual source",
                kind: "allowed-source",
                document: orphan,
              },
            ];
          },
        },
      ],
    },
  ]);
  render(<Workbench registry={registry}>{() => null}</Workbench>);
  const editor = screen.getByLabelText("Workspace archive JSON");
  await waitFor(() => expect(editor).toMatchObject({ disabled: false }));
  fireEvent.change(editor, { target: { value: plan.sourceJson } });
  fireEvent.click(screen.getByRole("button", { name: "Preview archive" }));
  await screen.findByRole("button", { name: "Duplicate first" });
  for (const label of [
    "Duplicate first",
    "Duplicate second",
    "Undeclared kind",
    "Other actual source",
  ]) {
    const node = screen.getByRole("button", { name: label });
    expect(node).toMatchObject({ disabled: true });
    fireEvent.click(node);
  }
  expect(screen.queryByRole("navigation", { name: "Document tabs" })).toBeNull();
  expect(screen.getAllByText(/Source unavailable/).length).toBeGreaterThanOrEqual(4);
  expect((editor as HTMLTextAreaElement).value).toBe(plan.sourceJson);
});
