// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workflow owner vetoes incompatible module estimates

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryWorkflowPlan } from "../../test-fixtures/stationary-workflow-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { ModuleJobEstimate } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { workbenchModules, workbenchRegistry } from "./moduleRegistry";

type WorkflowEstimate = Extract<ModuleJobEstimate, { readonly kind: "workflow-runtime" }>;
const changes: readonly [string, (estimate: WorkflowEstimate) => WorkflowEstimate][] = [
  ["archive snapshot", (estimate) => ({ ...estimate, archiveDigest: "0".repeat(64) })],
  ["selected graph", (estimate) => ({ ...estimate, workflowDigest: "0".repeat(64) })],
  ["baseline revision", (estimate) => ({ ...estimate, baselineRevision: "0".repeat(64) })],
  ["stage count", (estimate) => ({ ...estimate, stages: estimate.stages + 1 })],
  ["cell count", (estimate) => ({ ...estimate, cells: estimate.cells + 1 })],
];
const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(changes)(
  "refuses a module's changed workflow %s without invoking the original execution owner",
  async (_name, change) => {
    const plan = await stationaryWorkflowPlan();
    const original = requirePresent(
      workbenchRegistry.jobs.find((job) => job.id === "local-workflow"),
    );
    let invocations = 0;
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
      ...workbenchModules,
      {
        id: "workflow-estimate-probe",
        apiVersion: "2.0",
        requires: { verbs: [], schemas: [] },
        stages: [],
        jobs: [
          {
            ...original,
            id: "workflow-estimate-probe",
            label: "Run workflow estimate probe",
            estimate: async (services, document) => {
              const estimate = await original.estimate(services, document);
              if (estimate.kind !== "workflow-runtime")
                throw new Error("Original selected workflow required");
              return change(estimate);
            },
            run: async (services, document, signal) => {
              invocations++;
              await original.run(services, document, signal);
            },
          },
        ],
      },
    ]);
    const started = BuiltKernelWorker.started;
    render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Original workspace state" }).textContent,
      ).not.toContain("Opening browser"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
    fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Run workflow estimate probe" })).toMatchObject({
        disabled: false,
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Run workflow estimate probe" }));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("Job estimate differs from the original selected workflow"),
    );
    expect(invocations).toBe(0);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
  },
);
