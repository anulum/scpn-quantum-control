// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered workflow estimation and genuine source transaction refusal

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { nativeKernelHttp } from "../../test-fixtures/native-kernel-http";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { stationaryWorkflowPlan } from "../../test-fixtures/stationary-workflow-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import type { ModuleDocument } from "../shared/contracts/module";
import type { WorkbenchServices } from "../shared/contracts/workbench";
import { admitWorkspaceArchive } from "../shared/storage/workspaceArchive";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { workbenchRegistry } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

/** Release only the actual portable archive admitted through the original workspace owner. */
async function admitSource(): Promise<void> {
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run selected local workflow" })).toMatchObject({
      disabled: false,
    }),
  );
}

it("refuses a registered workflow estimate when the actual admitted archive has no saved graph", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  render(<RegisteredExperimentRuntime plan={plan} />);
  await admitSource();
  fireEvent.click(screen.getByRole("button", { name: "Run selected local workflow" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Compose and save a workflow in Experiments first",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("runs the registered workflow through native kernel HTTP and retains the original archive when real storage is unavailable", async () => {
  const plan = await stationaryWorkflowPlan();
  const transport = await nativeKernelHttp();
  const started = BuiltKernelWorker.started;
  try {
    render(<RegisteredExperimentRuntime plan={plan} />);
    await admitSource();
    fireEvent.click(screen.getByRole("button", { name: "Run selected local workflow" }));
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Original workflow state" }).textContent).toContain(
        "failed:",
      ),
    );
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("settled:"),
    );
    expect(screen.getByRole("status", { name: "Original workflow state" }).textContent).toContain(
      "prior data retained",
    );
    expect(transport.requests).toEqual(["/kernel.wasm"]);
    expect(screen.getByLabelText("Registered workflow estimate").textContent).toContain(
      "Not estimated",
    );
    expect(screen.getByLabelText("Registered workflow estimate").textContent).toContain(
      "Cells1Stages per cell1",
    );
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
  } finally {
    cleanup();
    await transport.close();
  }
});

it.each(["missing selected graph", "aborted", "different admitted source"])(
  "the public registered workflow runner refuses %s without starting its native owner",
  async (cause) => {
    const plan =
      cause === "missing selected graph" ? await stationaryPlan() : await stationaryWorkflowPlan();
    const transport = await nativeKernelHttp();
    const observed: { services: WorkbenchServices | null; document: ModuleDocument | null } = {
      services: null,
      document: null,
    };
    const started = BuiltKernelWorker.started;
    try {
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
        workbenchRegistry.jobs.find((entry) => entry.id === "local-workflow"),
      );
      await expect(
        job.run(
          services,
          document,
          cause === "aborted" ? AbortSignal.abort() : new AbortController().signal,
        ),
      ).rejects.toThrow(
        cause === "missing selected graph"
          ? "Original selected workflow unavailable"
          : "Workflow source changed or execution cancelled",
      );
      expect(transport.requests).toEqual(
        cause === "missing selected graph" ? [] : ["/kernel.wasm"],
      );
      expect(screen.getByRole("status", { name: "Original workflow state" }).textContent).toContain(
        "idle:",
      );
      expect(BuiltKernelWorker.started).toBe(started);
      expect(BuiltKernelWorker.activeCount).toBe(0);
      expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
        plan.sourceJson,
      );
    } finally {
      cleanup();
      await transport.close();
    }
  },
);

it("cancels a registered workflow during the original traversal before native allocation and retains source", async () => {
  const plan = await stationaryWorkflowPlan();
  const transport = await nativeKernelHttp();
  const started = BuiltKernelWorker.started;
  let requested = false;
  try {
    render(
      <RegisteredExperimentRuntime
        plan={plan}
        observeServices={(services) => {
          if (services.workflow.status === "running" && !requested) {
            requested = true;
            queueMicrotask(() => {
              fireEvent.click(screen.getByRole("button", { name: "Cancel registered job" }));
            });
          }
        }}
      />,
    );
    await admitSource();
    fireEvent.click(screen.getByRole("button", { name: "Run selected local workflow" }));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("cancelled:"),
    );
    expect(requested).toBe(true);
    expect(transport.requests).toEqual(["/kernel.wasm"]);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
  } finally {
    cleanup();
    await transport.close();
  }
});
