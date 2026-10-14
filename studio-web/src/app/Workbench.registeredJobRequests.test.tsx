// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — host job requests retain actual admission and availability boundaries

import { webcrypto } from "node:crypto";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import {
  admitRegisteredSource,
  prepareRegisteredPlan,
} from "../../test-fixtures/registered-job-controls";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { policyProbeModule } from "../../test-fixtures/policy-probe";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { requirePresent } from "../../test-support/requirePresent";
import { createModuleRegistry } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
import { workbenchModules } from "./moduleRegistry";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it("refuses an actual command before an original source has been admitted", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  render(
    <RegisteredExperimentRuntime plan={plan} request={{ sequence: 1, job: "local-experiment" }} />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Select an admitted source and registered job first",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses an unknown host job request without replacing the genuinely admitted source", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  const host = render(<RegisteredExperimentRuntime plan={plan} />);
  await admitRegisteredSource();
  host.rerender(
    <RegisteredExperimentRuntime plan={plan} request={{ sequence: 1, job: "unknown-job" }} />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Select an admitted source and registered job first",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("names missing verb and schema requirements and refuses a host request despite the disabled job control", async () => {
  const plan = await stationaryPlan();
  let invocations = 0;
  const probe = policyProbeModule(() => {
    invocations++;
  });
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    { ...probe, requires: { verbs: ["not-installed-submit"], schemas: ["not-installed.v1"] } },
  ]);
  const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
  await admitRegisteredSource();
  expect(screen.getByRole("button", { name: "Run policy probe" })).toMatchObject({
    disabled: true,
  });
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      registry={registry}
      request={{ sequence: 1, job: "policy-probe" }}
    />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Job unavailable: verb not-installed-submit, schema not-installed.v1",
    ),
  );
  expect(invocations).toBe(0);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("does not activate an absent original experiment runtime through a host request from Workspace", async () => {
  const plan = await stationaryPlan();
  const started = BuiltKernelWorker.started;
  const host = render(<RegisteredExperimentRuntime plan={plan} />);
  await admitRegisteredSource();
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      stage="workspace"
      request={{ sequence: 1, job: "local-experiment" }}
    />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Open experiments before requesting this job",
    ),
  );
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("keeps a compiled local-service job unavailable and refuses its host command before execution", async () => {
  const plan = await stationaryPlan();
  let invocations = 0;
  const probe = policyProbeModule(() => {
    invocations++;
  });
  const original = requirePresent(probe.jobs?.[0]);
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    { ...probe, jobs: [{ ...original, runner: "local-service" }] },
  ]);
  const started = BuiltKernelWorker.started;
  const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
  await admitRegisteredSource();
  await prepareRegisteredPlan();
  expect(screen.getByRole("button", { name: "Run policy probe" })).toMatchObject({
    disabled: true,
  });
  expect(screen.getByText(/Local service unavailable/)).toBeTruthy();
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      registry={registry}
      request={{ sequence: 1, job: "policy-probe" }}
    />,
  );
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "Local job service is unavailable in this build",
    ),
  );
  expect(invocations).toBe(0);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});
