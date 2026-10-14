// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — documented admission API over actual mounted runtime owners

import { webcrypto } from "node:crypto";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../../test-fixtures/registered-experiment-runtime";
import {
  admitRegisteredSource,
  prepareRegisteredPlan,
} from "../../../test-fixtures/registered-job-controls";
import { stationaryPlan } from "../../../test-fixtures/stationary-experiment-plan";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { requirePresent } from "../../../test-support/requirePresent";
import { workbenchRegistry } from "../../app/moduleRegistry";
import type { ModuleDocument, ModuleJobEstimate } from "../contracts/module";
import type { WorkbenchServices } from "../contracts/workbench";
import { admitWorkspaceArchive } from "../storage/workspaceArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { admitModuleJob } from "./moduleJobPolicy";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Bind documented API inputs to the actual original controllers and admitted source. */
async function mountedPlan() {
  const plan = await stationaryPlan();
  const observed: { services: WorkbenchServices | null; document: ModuleDocument | null } = {
    services: null,
    document: null,
  };
  render(
    <RegisteredExperimentRuntime
      plan={plan}
      observeServices={(services, document) => {
        observed.services = services;
        observed.document = document;
      }}
    />,
  );
  await admitRegisteredSource();
  await prepareRegisteredPlan();
  await waitFor(() => expect(observed.services?.experiment.plan).toBe(plan));
  const job = requirePresent(
    workbenchRegistry.jobs.find((entry) => entry.id === "local-experiment"),
  );
  const document = requirePresent(observed.document);
  const estimate = await job.estimate(requirePresent(observed.services), document);
  return { plan, observed, job, document, estimate };
}

it("the documented admission API refuses an already aborted request against genuine current runtime inputs", async () => {
  const state = await mountedPlan();
  const started = BuiltKernelWorker.started;
  await expect(
    admitModuleJob(
      requirePresent(state.observed.services),
      state.job,
      state.document,
      state.estimate,
      AbortSignal.abort(),
    ),
  ).rejects.toThrow("Job cancelled before admission");
  expect(BuiltKernelWorker.started).toBe(started);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    state.plan.sourceJson,
  );
});

it.each(["different source", "mixed source identity", "edited before admission"])(
  "the documented API refuses %s without relabelling the current original archive",
  async (cause) => {
    const state = await mountedPlan();
    const other = await admitWorkspaceArchive(`${state.plan.sourceJson}\n`, localExperimentCodecs);
    let document = state.document;
    if (cause === "different source")
      document = {
        schema: other.preview.schema,
        content: other.preview.json,
        sha256: other.preview.archiveDigest,
      };
    if (cause === "mixed source identity")
      document = { ...state.document, sha256: other.preview.archiveDigest };
    if (cause === "edited before admission")
      fireEvent.change(screen.getByLabelText("Original archive bytes"), {
        target: { value: other.preview.json },
      });
    const started = BuiltKernelWorker.started;
    await expect(
      admitModuleJob(
        requirePresent(state.observed.services),
        state.job,
        document,
        state.estimate,
        new AbortController().signal,
      ),
    ).rejects.toThrow("Select the current admitted source before running a job");
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      cause === "edited before admission" ? other.preview.json : state.plan.sourceJson,
    );
  },
);

it("refuses a second admission while the actual original worker is running", async () => {
  const plan = await stationaryPlan();
  const job = requirePresent(
    workbenchRegistry.jobs.find((entry) => entry.id === "local-experiment"),
  );
  const observed: {
    services: WorkbenchServices | null;
    document: ModuleDocument | null;
    estimate: ModuleJobEstimate | null;
    refusal: Promise<void> | null;
  } = { services: null, document: null, estimate: null, refusal: null };
  const started = BuiltKernelWorker.started;
  render(
    <RegisteredExperimentRuntime
      plan={plan}
      observeServices={(services, document) => {
        observed.services = services;
        observed.document = document;
        if (
          services.experiment.status === "running" &&
          observed.estimate !== null &&
          observed.refusal === null
        ) {
          observed.refusal = expect(
            admitModuleJob(
              services,
              job,
              requirePresent(document),
              observed.estimate,
              new AbortController().signal,
            ),
          ).rejects.toThrow(
            "The original workspace or worker owner is busy or disposal is unconfirmed",
          );
        }
      }}
    />,
  );
  await admitRegisteredSource();
  await prepareRegisteredPlan();
  await waitFor(() => expect(observed.services?.experiment.plan).toBe(plan));
  observed.estimate = await job.estimate(
    requirePresent(observed.services),
    requirePresent(observed.document),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run prepared local experiment" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "settled:",
    ),
  );
  await requirePresent(observed.refusal);
  expect(BuiltKernelWorker.started).toBe(started + 1);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(screen.getByLabelText("Original final phases").textContent).toBe("0,0");
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("refuses a caller's missing cached component against genuine dense original resource declarations", async () => {
  const state = await mountedPlan();
  const components = [...state.plan.admission.estimate.components];
  Reflect.deleteProperty(components, "0");
  const supplied = {
    ...state.plan,
    admission: {
      ...state.plan.admission,
      estimate: { ...state.plan.admission.estimate, components },
    },
  };
  await act(async () => {
    await requirePresent(state.observed.services).experiment.prepare(supplied);
  });
  const started = BuiltKernelWorker.started;
  await expect(
    admitModuleJob(
      requirePresent(state.observed.services),
      state.job,
      state.document,
      state.estimate,
      new AbortController().signal,
    ),
  ).rejects.toThrow("Job resource declarations differ from the original prepared plan");
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    state.plan.sourceJson,
  );
});
