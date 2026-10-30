// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual host changes at registered admission completion

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import {
  admitRegisteredSource,
  prepareRegisteredPlan,
} from "../../test-fixtures/registered-job-controls";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { workbenchModules, workbenchRegistry } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(["edit", "stage", "unmount"])(
  "refuses a real prepared job when the host chooses %s at admission completion",
  async (change) => {
    const plan = await stationaryPlan();
    const original = requirePresent(
      workbenchRegistry.jobs.find((job) => job.id === "local-experiment"),
    );
    let changeHost: (() => void) | undefined;
    let changed = false,
      runs = 0;
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
      ...workbenchModules,
      {
        id: "completion-boundary",
        apiVersion: "2.0",
        requires: { verbs: [], schemas: [] },
        stages: [],
        jobs: [
          {
            ...original,
            id: "completion-boundary",
            label: "Run completion boundary",
            estimate: async (services, document) => {
              const estimate = await original.estimate(services, document);
              queueMicrotask(() =>
                queueMicrotask(() => {
                  changed = true;
                  requirePresent(changeHost)();
                }),
              );
              return estimate;
            },
            run: async (services, document, signal) => {
              runs++;
              await original.run(services, document, signal);
            },
          },
        ],
      },
    ]);
    const started = BuiltKernelWorker.started;
    const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
    await admitRegisteredSource();
    await prepareRegisteredPlan();
    const replacement = `${plan.sourceJson}\n`;
    changeHost = () => {
      if (change === "edit")
        fireEvent.change(screen.getByLabelText("Original archive bytes"), {
          target: { value: replacement },
        });
      else if (change === "stage")
        host.rerender(
          <RegisteredExperimentRuntime plan={plan} registry={registry} stage="workspace" />,
        );
      else host.unmount();
    };
    fireEvent.click(screen.getByRole("button", { name: "Run completion boundary" }));
    await waitFor(() => expect(changed).toBe(true));
    if (change === "unmount") expect(host.container.childElementCount).toBe(0);
    else {
      await waitFor(() =>
        expect(
          screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
        ).toContain("Job cancelled or current source/module changed during admission"),
      );
      expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
        change === "edit" ? replacement : plan.sourceJson,
      );
    }
    expect(runs).toBe(0);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
  },
);

it.each(["estimate", "run"])(
  "contains a compiled host's primitive %s refusal while retaining the actual original runtime's outcome",
  async (phase) => {
    const plan = await stationaryPlan();
    const original = requirePresent(
      workbenchRegistry.jobs.find((job) => job.id === "local-experiment"),
    );
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
      ...workbenchModules,
      {
        id: "primitive-refusal",
        apiVersion: "2.0",
        requires: { verbs: [], schemas: [] },
        stages: [],
        jobs: [
          {
            ...original,
            id: "primitive-refusal",
            label: "Run primitive refusal",
            estimate: async (services, document) => {
              const estimate = await original.estimate(services, document);
              if (phase === "estimate") throw "private compiled host cause";
              return estimate;
            },
            run: async (services, document, signal) => {
              await original.run(services, document, signal);
              throw "private compiled host cause";
            },
          },
        ],
      },
    ]);
    const started = BuiltKernelWorker.started;
    render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
    await admitRegisteredSource();
    await prepareRegisteredPlan();
    fireEvent.click(screen.getByRole("button", { name: "Run primitive refusal" }));
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("Job request refused; original source retained"),
    );
    expect(screen.queryByText("private compiled host cause")).toBeNull();
    expect(BuiltKernelWorker.started).toBe(started + (phase === "run" ? 1 : 0));
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
    expect(screen.getByRole("status", { name: "Original experiment state" }).textContent).toBe(
      phase === "run" ? "succeeded" : "planned",
    );
  },
);
