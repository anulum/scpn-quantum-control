// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original resource admission vetoes incompatible compiled jobs

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { policyProbeModule } from "../../test-fixtures/policy-probe";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import type { ResourceBuffer, ResourcePlan } from "../shared/resources/admission";
import { instantiateKuramoto } from "../shared/browser/kuramoto";
import { readLocalExperiment } from "../shared/experiments/experimentArchive";
import { prepareExperimentPlan } from "../shared/experiments/experimentPlan";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { workbenchModules } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Change one original live buffer declaration while retaining all other supplied declarations. */
function bufferChange(change: (buffer: ResourceBuffer) => ResourceBuffer) {
  return (plan: ResourcePlan): ResourcePlan => ({
    ...plan,
    buffers: plan.buffers.map((buffer, index) => (index === 0 ? change(buffer) : buffer)),
  });
}

const differences: readonly [string, (plan: ResourcePlan) => ResourcePlan][] = [
  ["backend", (plan) => ({ ...plan, backend: `${plan.backend}-other` })],
  ["method", (plan) => ({ ...plan, method: `${plan.method}-other` })],
  ["concurrency", (plan) => ({ ...plan, concurrency: plan.concurrency + 1n })],
  ["work units", (plan) => ({ ...plan, workUnits: (plan.workUnits ?? 0n) + 1n })],
  ["buffer count", (plan) => ({ ...plan, buffers: plan.buffers.slice(1) })],
  ["buffer name", bufferChange((buffer) => ({ ...buffer, name: `${buffer.name}-other` }))],
  [
    "buffer owner",
    bufferChange((buffer) => ({ ...buffer, role: buffer.role === "graph" ? "transfer" : "graph" })),
  ],
  [
    "buffer precision",
    bufferChange((buffer) => ({
      ...buffer,
      dtype: buffer.dtype === "float64" ? "float32" : "float64",
    })),
  ],
  ["buffer copies", bufferChange((buffer) => ({ ...buffer, count: buffer.count + 1n }))],
  ["buffer rank", bufferChange((buffer) => ({ ...buffer, shape: [...buffer.shape, 1n] }))],
  [
    "buffer dimension",
    bufferChange((buffer) => ({
      ...buffer,
      shape: buffer.shape.map((value, index) => (index === 0 ? value + 1n : value)),
    })),
  ],
];

it.each(differences)(
  "refuses a module's changed %s despite its allowed flag, retaining the genuinely admitted source without allocating",
  async (_name, change) => {
    const plan = await stationaryPlan();
    expect(plan.admission.allowed).toBe(true);
    let invocations = 0;
    const probe = policyProbeModule(() => {
      invocations++;
    });
    const original = requirePresent(probe.jobs?.[0]);
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
      ...workbenchModules,
      {
        ...probe,
        jobs: [
          {
            ...original,
            estimate: async (services, document) => {
              const estimate = await original.estimate(services, document);
              if (estimate.kind !== "declared-resources")
                throw new Error("Original numerical declarations required");
              return { ...estimate, plan: change(estimate.plan) };
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
      expect(screen.getByRole("button", { name: "Run policy probe" })).toMatchObject({
        disabled: false,
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
        "planned",
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Run policy probe" }));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("Job resource declarations differ from the original prepared plan"),
    );
    expect(invocations).toBe(0);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
  },
);

it("rechecks the actual zero-budget policy when a caller relabels its denied receipt as allowed", async () => {
  const baseline = await stationaryPlan();
  const kernel = await instantiateKuramoto(baseline.wasmBytes);
  const denied = await prepareExperimentPlan(
    await readLocalExperiment(baseline.sourceJson, localExperimentCodecs),
    kernel,
    { memoryBudget: "0" },
  );
  expect(denied.admission.allowed).toBe(false);
  expect(denied.admission.blockers).toContain("declared_storage_exceeds_budget");
  const supplied = { ...denied, admission: { ...denied.admission, allowed: true } };
  let invocations = 0;
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    policyProbeModule(() => {
      invocations++;
    }),
  ]);
  const started = BuiltKernelWorker.started;
  render(<RegisteredExperimentRuntime plan={supplied} registry={registry} />);
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run policy probe" })).toMatchObject({
      disabled: false,
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      "planned",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run policy probe" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Job policy refused: declared_storage_exceeds_budget",
    ),
  );
  expect(invocations).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    baseline.sourceJson,
  );
});
