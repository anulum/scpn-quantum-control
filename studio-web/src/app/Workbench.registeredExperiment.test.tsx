// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered numerical jobs through actual source and worker owners

import { webcrypto } from "node:crypto";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import type { ModuleDocument } from "../shared/contracts/module";
import type { WorkbenchServices } from "../shared/contracts/workbench";
import { admitWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { workbenchRegistry } from "./moduleRegistry";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Admit the original source through visible controls without supplying a synthetic preview. */
async function admitSource(): Promise<void> {
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run prepared local experiment" })).toMatchObject({
      disabled: false,
    }),
  );
}

it("runs the actual registered experiment once, retains its exact source and observes native worker disposal", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  render(<RegisteredExperimentRuntime plan={plan} />);
  await admitSource();
  fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      "planned",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run prepared local experiment" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "settled",
    ),
  );
  expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
    "succeeded",
  );
  expect(screen.getByLabelText("Original final phases").textContent).toBe("0,0");
  expect(BuiltKernelWorker.started).toBe(started + 1);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
  expect(screen.getByLabelText("Registered job estimate").textContent).toContain(
    plan.admission.estimate.backend,
  );
});

it("disposes the actual original worker when its registered host unmounts during execution", async () => {
  const plan = await stationaryPlan();
  const originalSource = plan.sourceJson;
  const started = BuiltKernelWorker.started;
  let unmounted = false;
  const host = render(
    <RegisteredExperimentRuntime
      plan={plan}
      onWorkerAllocated={() => {
        expect(BuiltKernelWorker.activeCount).toBe(1);
        host.unmount();
        unmounted = true;
      }}
    />,
  );
  await admitSource();
  fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      "planned",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run prepared local experiment" }));
  await waitFor(() => expect(unmounted).toBe(true));
  await waitFor(() => expect(BuiltKernelWorker.activeCount).toBe(0));
  await act(async () => {
    await Promise.resolve();
  });
  expect(BuiltKernelWorker.started).toBe(started + 1);
  expect(host.container.childElementCount).toBe(0);
  expect(plan.sourceJson).toBe(originalSource);
});

it("refuses the registered estimate before allocation when the real original source has no prepared plan", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  render(<RegisteredExperimentRuntime plan={plan} />);
  await admitSource();
  fireEvent.click(screen.getByRole("button", { name: "Run prepared local experiment" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Prepare the current numerical plan",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("cancels the registered job after genuine thread allocation and retains source until observed disposal", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  render(
    <RegisteredExperimentRuntime
      plan={plan}
      onWorkerAllocated={() => {
        expect(BuiltKernelWorker.started).toBe(started + 1);
        expect(BuiltKernelWorker.activeCount).toBe(1);
        fireEvent.click(screen.getByRole("button", { name: "Cancel registered job" }));
      }}
    />,
  );
  await admitSource();
  fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      "planned",
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run prepared local experiment" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "cancelled",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started + 1);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
    "cancelled",
  );
  expect(screen.getByLabelText("Original final phases").textContent).toBe("No result");
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it.each(["aborted", "different admitted source"])(
  "the public registered runner refuses %s before allocating a native worker",
  async (cause) => {
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
    await admitSource();
    fireEvent.click(screen.getByRole("button", { name: "Prepare original plan" }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
        "planned",
      ),
    );
    const services = requirePresent(observed.services);
    let document = requirePresent(observed.document);
    if (cause === "different admitted source") {
      const admitted = await admitWorkspaceArchive(`${plan.sourceJson}\n`, localExperimentCodecs);
      document = {
        schema: admitted.preview.schema,
        content: admitted.preview.json,
        sha256: admitted.preview.archiveDigest,
      };
      expect(document.sha256).not.toBe(requirePresent(observed.document).sha256);
    }
    const job = requirePresent(
      workbenchRegistry.jobs.find((entry) => entry.id === "local-experiment"),
    );
    const started = BuiltKernelWorker.started;
    await expect(
      job.run(
        services,
        document,
        cause === "aborted" ? AbortSignal.abort() : new AbortController().signal,
      ),
    ).rejects.toThrow("Experiment source changed or execution cancelled");
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
  },
);
