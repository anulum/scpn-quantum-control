// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — retained compiled declarations cannot acquire a different owner

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
import type { ModuleCommand, ModuleExplorer } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { Workbench } from "./Workbench";
import { workbenchModules, workbenchRegistry } from "./moduleRegistry";

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

it("renders a command and explorer as unavailable when their producer changes retained identifiers after registry admission", async () => {
  const command: ModuleCommand = {
    id: "retained-atlas-command",
    label: "Retained Atlas command",
    action: { kind: "navigate", stage: "atlas", instrument: null },
  };
  const explorer: ModuleExplorer<WorkbenchServices> = {
    id: "retained-explorer",
    label: "Retained source explorer",
    kinds: [],
    nodes: () => [],
  };
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "retained-declarations",
      apiVersion: "2.0",
      requires: { verbs: [], schemas: [] },
      stages: [],
      commands: [command],
      explorer: [explorer],
    },
  ]);
  Reflect.set(command, "id", "unregistered-retained-command");
  Reflect.set(explorer, "id", "unregistered-retained-explorer");
  render(<Workbench registry={registry}>{() => null}</Workbench>);
  await waitFor(() =>
    expect(screen.getByLabelText("Workspace archive JSON")).toMatchObject({ disabled: false }),
  );
  const control = screen.getByRole("button", { name: command.label });
  expect(control).toMatchObject({ disabled: true });
  expect(screen.getByText(/Registered command owner unavailable$/)).toBeTruthy();
  expect(screen.getByText("Registered explorer owner unavailable")).toBeTruthy();
  const before = window.location.hash;
  fireEvent.click(control);
  expect(window.location.hash).toBe(before);
});

it("refuses a retained original job declaration whose identifier no longer resolves its registered owner", async () => {
  const plan = await stationaryPlan();
  const original = requirePresent(
    workbenchRegistry.jobs.find((job) => job.id === "local-experiment"),
  );
  const job = { ...original, id: "retained-original-job", label: "Retained original experiment" };
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    {
      id: "retained-job-owner",
      apiVersion: "2.0",
      requires: { verbs: [], schemas: [] },
      stages: [],
      jobs: [job],
    },
  ]);
  const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
  await admitRegisteredSource();
  await prepareRegisteredPlan();
  const started = BuiltKernelWorker.started;
  Reflect.set(job, "id", "unregistered-retained-job");
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      registry={registry}
      request={{ sequence: 1, job: job.id }}
    />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Registered job owner unavailable",
    ),
  );
  expect(
    screen.getByText(/Retained original experiment: registered owner unavailable/),
  ).toBeTruthy();
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(screen.getByLabelText<HTMLTextAreaElement>("Original archive bytes").value).toBe(
    plan.sourceJson,
  );
});
