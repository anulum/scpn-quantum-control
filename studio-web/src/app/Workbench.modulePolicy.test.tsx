// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original numerical policy veto of a compiled module job

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { policyProbeModule } from "../../test-fixtures/policy-probe";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { instantiateKuramoto } from "../shared/browser/kuramoto";
import { createModuleRegistry } from "../shared/contracts/module";
import type { ModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import {
  createLocalExperiment,
  readLocalExperiment,
} from "../shared/experiments/experimentArchive";
import { prepareExperimentPlan } from "../shared/experiments/experimentPlan";
import type { ExperimentPlan } from "../shared/experiments/experimentPlan";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { useExperimentRun } from "../shared/experiments/useExperimentRun";
import { ModuleJobDock } from "../shared/moduleViews/ModuleJobDock";
import { useWorkflowRun } from "../shared/workflows/useWorkflowRun";
import { useWorkspace } from "../shared/workspace/useWorkspace";
import { workbenchModules } from "./moduleRegistry";

const originalCrypto = globalThis.crypto;
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true });
});
afterEach(() => {
  cleanup();
  Object.defineProperty(globalThis, "crypto", { value: originalCrypto, configurable: true });
});

function OriginalOwners({
  plan,
  registry,
}: {
  readonly plan: ExperimentPlan;
  readonly registry: ModuleRegistry<WorkbenchServices, WorkspaceViewServices>;
}) {
  const workspace = useWorkspace(localExperimentCodecs);
  const experiment = useExperimentRun(workspace.draft, true, {
    workerFactory: () => new BuiltKernelWorker(),
  });
  const workflow = useWorkflowRun(workspace.draft, true);
  const services: WorkbenchServices = {
    workspace,
    experiment,
    workflow,
    codecs: localExperimentCodecs,
    documents: [],
    renderEditors: () => null,
    overview: () => null,
    contextSuffix: "",
    workspaceHref: "#/workspace",
  };
  const preview = workspace.preview;
  return (
    <>
      <p role="status" aria-label="Original workspace state">
        {workspace.message}
      </p>
      <p role="status" aria-label="Original experiment state">
        {experiment.status}: {experiment.reason}
      </p>
      <button type="button" onClick={() => workspace.edit(plan.sourceJson)}>
        Import original source
      </button>
      <button
        type="button"
        onClick={() => {
          void workspace.inspect();
        }}
      >
        Admit original source
      </button>
      <button
        type="button"
        onClick={() => {
          void experiment.prepare(plan);
        }}
      >
        Prepare original refused plan
      </button>
      <textarea aria-label="Original archive bytes" readOnly value={workspace.draft} />
      <ModuleJobDock
        registry={registry}
        services={services}
        declared={{ verbs: ["compile"], schemas: [] }}
        activeStage="experiments"
        request={null}
        document={
          preview === null
            ? null
            : { schema: preview.schema, sha256: preview.archiveDigest, content: preview.json }
        }
      />
    </>
  );
}

it("a compiled module's allowed flag cannot override a genuinely refused original numerical plan or start a worker", async () => {
  const wasm = new Uint8Array(
    readFileSync(
      process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
        "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
    ),
  );
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 },
    "independent analytic fixture",
    localExperimentCodecs,
    "test runtime",
  );
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const plan = await prepareExperimentPlan(source, kernel, { memoryBudget: "0" });
  expect(plan.admission.allowed).toBe(false);
  expect(plan.admission.blockers).toContain("declared_storage_exceeds_budget");
  let invocations = 0;
  const probe = policyProbeModule(() => {
    invocations++;
  });
  expect(probe.jobs?.[0]).toHaveProperty("allowed", true);
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    probe,
  ]);
  const started = BuiltKernelWorker.started;
  render(<OriginalOwners plan={plan} registry={registry} />);
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(
      (screen.getByRole("button", { name: "Run policy probe" }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
  fireEvent.click(screen.getByRole("button", { name: "Prepare original refused plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toContain(
      "refused",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run policy probe" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Prepare an allowed current numerical plan",
    ),
  );
  expect(invocations).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    archive.json,
  );
});
