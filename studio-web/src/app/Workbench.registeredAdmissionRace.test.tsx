// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original source changes while a registered job is being admitted

import { webcrypto } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { stationaryWorkflowPlan } from "../../test-fixtures/stationary-workflow-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import type { RawCodec } from "../shared/contracts";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { localExperimentCodecs } from "../shared/experiments/kuramotoArtifacts";
import { workbenchModules, workbenchRegistry } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it("retains a real edit made during source admission after the original registered estimate completes", async () => {
  const plan = await stationaryWorkflowPlan();
  const replacement = `${plan.sourceJson}\n`;
  const original = requirePresent(
    workbenchRegistry.jobs.find((job) => job.id === "local-workflow"),
  );
  let estimated = false,
    edited = false,
    invocations = 0;
  const codecs = new Map<string, RawCodec>();
  for (const [schema, codec] of localExperimentCodecs) {
    codecs.set(schema, async (content) => {
      const result = await codec(content);
      if (estimated && !edited) {
        edited = true;
        fireEvent.change(screen.getByLabelText("Original archive bytes"), {
          target: { value: replacement },
        });
      }
      return result;
    });
  }
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "admission-source-race",
      version: "1",
      requires: { verbs: [], schemas: [] },
      stages: [],
      jobs: [
        {
          ...original,
          id: "admission-source-race",
          label: "Run admission source race",
          estimate: async (services, document) => {
            const estimate = await original.estimate(services, document);
            estimated = true;
            return estimate;
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
  render(<RegisteredExperimentRuntime plan={plan} registry={registry} rawCodecs={codecs} />);
  await waitFor(() =>
    expect(
      screen.getByRole("status", { name: "Original workspace state" }).textContent,
    ).not.toContain("Opening browser"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Import original source" }));
  fireEvent.click(screen.getByRole("button", { name: "Admit original source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Run admission source race" })).toMatchObject({
      disabled: false,
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Run admission source race" }));
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Source changed or job cancelled during admission",
    ),
  );
  expect(estimated).toBe(true);
  expect(edited).toBe(true);
  expect(invocations).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    replacement,
  );
});
