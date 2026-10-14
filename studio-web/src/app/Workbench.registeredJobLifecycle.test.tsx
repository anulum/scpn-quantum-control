// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source and module ownership across pending registered jobs

import { webcrypto } from "node:crypto";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { heldWorkflowEstimate } from "../../test-fixtures/held-workflow-estimate";
import { RegisteredExperimentRuntime } from "../../test-fixtures/registered-experiment-runtime";
import { admitRegisteredSource } from "../../test-fixtures/registered-job-controls";
import { stationaryWorkflowPlan } from "../../test-fixtures/stationary-workflow-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
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

it.each(["cancel", "edit source", "change stage", "remove module"])(
  "does not execute an earlier genuine estimate after the user chooses to %s",
  async (change) => {
    const plan = await stationaryWorkflowPlan();
    const held = heldWorkflowEstimate();
    const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
      ...workbenchModules,
      held.module,
    ]);
    const started = BuiltKernelWorker.started;
    const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
    await admitRegisteredSource();
    fireEvent.click(screen.getByRole("button", { name: "Run held original workflow" }));
    await held.entered;
    const replacement = `${plan.sourceJson}\n`;
    if (change === "cancel")
      fireEvent.click(screen.getByRole("button", { name: "Cancel registered job" }));
    if (change === "edit source")
      fireEvent.change(screen.getByLabelText("Original archive bytes"), {
        target: { value: replacement },
      });
    if (change === "change stage")
      host.rerender(
        <RegisteredExperimentRuntime plan={plan} registry={registry} stage="workspace" />,
      );
    if (change === "remove module")
      host.rerender(<RegisteredExperimentRuntime plan={plan} registry={workbenchRegistry} />);
    await act(async () => {
      held.release();
    });
    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
      ).toContain("Job cancelled or source/module changed during preparation"),
    );
    expect(held.estimates).toBe(1);
    expect(held.runs).toBe(0);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
      change === "edit source" ? replacement : plan.sourceJson,
    );
    if (change === "remove module")
      expect(screen.queryByRole("button", { name: "Run held original workflow" })).toBeNull();
  },
);

it("coalesces a second host request while the first genuine estimate is owned", async () => {
  const plan = await stationaryWorkflowPlan();
  const held = heldWorkflowEstimate();
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    held.module,
  ]);
  const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
  await admitRegisteredSource();
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      registry={registry}
      request={{ sequence: 1, job: "held-workflow" }}
    />,
  );
  await held.entered;
  host.rerender(
    <RegisteredExperimentRuntime
      plan={plan}
      registry={registry}
      request={{ sequence: 2, job: "held-workflow" }}
    />,
  );
  expect(held.estimates).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "Cancel registered job" }));
  await act(async () => {
    held.release();
  });
  await waitFor(() =>
    expect(screen.getByRole("status", { name: "Registered job lifecycle" }).textContent).toContain(
      "cancelled:",
    ),
  );
  expect(held.estimates).toBe(1);
  expect(held.runs).toBe(0);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect((screen.getByLabelText("Original archive bytes") as HTMLTextAreaElement).value).toBe(
    plan.sourceJson,
  );
});

it("unmounts during a real estimate without a late execution or view update", async () => {
  const plan = await stationaryWorkflowPlan();
  const held = heldWorkflowEstimate();
  const registry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>([
    ...workbenchModules,
    held.module,
  ]);
  const host = render(<RegisteredExperimentRuntime plan={plan} registry={registry} />);
  await admitRegisteredSource();
  fireEvent.click(screen.getByRole("button", { name: "Run held original workflow" }));
  await held.entered;
  host.unmount();
  await act(async () => {
    held.release();
  });
  expect(host.container.childElementCount).toBe(0);
  expect(held.estimates).toBe(1);
  expect(held.runs).toBe(0);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
