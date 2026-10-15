// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real workbench command, source inspector and numerical job wiring

import { webcrypto } from "node:crypto";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { nativeKernelHttp } from "../../test-fixtures/native-kernel-http";
import { stationaryPlan } from "../../test-fixtures/stationary-experiment-plan";
import { BuiltKernelWorker } from "../../test-support/kernelWorker";
import { Workbench } from "./Workbench";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
const originalWorker = Object.getOwnPropertyDescriptor(globalThis, "Worker");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
  // Run the actual built browser entry in a genuinely disposable native thread.
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: BuiltKernelWorker });
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
  if (originalWorker) Object.defineProperty(globalThis, "Worker", originalWorker);
  else Reflect.deleteProperty(globalThis, "Worker");
  window.history.replaceState(null, "", "/");
});

/** Activate only a production control that has completed its original admission work. */
async function clickReady(name: string): Promise<void> {
  const control = screen.getByRole("button", { name });
  await waitFor(() => expect(control).toMatchObject({ disabled: false }));
  fireEvent.click(control);
}

it("dispatches two real jobs through command search and opens the registered settings inspector without changing source", async () => {
  const nativeDisposals: Promise<void>[] = [];
  class ObservedKernelWorker extends BuiltKernelWorker {
    private acknowledgeDisposal: (() => void) | undefined;

    constructor(...parameters: ConstructorParameters<typeof BuiltKernelWorker>) {
      super(...parameters);
      nativeDisposals.push(
        new Promise<void>((resolve) => {
          this.acknowledgeDisposal = resolve;
        }),
      );
    }

    override async terminate(): Promise<void> {
      await super.terminate();
      this.acknowledgeDisposal?.();
    }
  }
  Object.defineProperty(globalThis, "Worker", { configurable: true, value: ObservedKernelWorker });
  const plan = await stationaryPlan();
  const transport = await nativeKernelHttp();
  const started = BuiltKernelWorker.started;
  try {
    render(<Workbench>{() => null}</Workbench>);
    const editor = screen.getByLabelText("Workspace archive JSON");
    await waitFor(() => expect(editor).toMatchObject({ disabled: false }));
    fireEvent.change(editor, { target: { value: plan.sourceJson } });
    await clickReady("Preview archive");
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Workspace source admission" }).textContent).toBe(
        "Original workspace sources admitted",
      ),
    );
    const commands = within(screen.getByRole("navigation", { name: "Workbench commands" }));
    fireEvent.change(commands.getByLabelText("Search commands"), {
      target: { value: "Open Experiments" },
    });
    fireEvent.click(commands.getByRole("button", { name: "Open Experiments" }));
    await screen.findByRole("heading", { name: "Local experiment" });
    await clickReady("Validate experiment archive");
    await clickReady("Prepare numerical plan");
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain(
        "planned",
      ),
    );
    fireEvent.change(commands.getByLabelText("Search commands"), {
      target: { value: "Run prepared local experiment" },
    });
    for (let attempt = 1; attempt <= 2; attempt++) {
      fireEvent.click(commands.getByRole("button", { name: "Run prepared local experiment" }));
      await waitFor(() => expect(BuiltKernelWorker.started).toBe(started + attempt));
      expect(nativeDisposals[attempt - 1]).toBeDefined();
      await act(async () => {
        await nativeDisposals[attempt - 1];
      });
      await waitFor(() =>
        expect(
          screen.getByRole("status", { name: "Registered job lifecycle" }).textContent,
        ).toContain("settled:"),
      );
      expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain(
        "succeeded",
      );
      expect(BuiltKernelWorker.activeCount).toBe(0);
      expect(
        within(screen.getByLabelText("Original final phases"))
          .getAllByRole("definition")
          .map((value) => value.textContent),
      ).toEqual(["0", "0"]);
    }
    const explorer = within(screen.getByRole("navigation", { name: "Project explorer" }));
    fireEvent.click(explorer.getByRole("button", { name: /^resolved_settings\.v1 ·/ }));
    await screen.findByRole("region", { name: "Settings document inspector" });
    expect(screen.getByRole("table", { name: "Requested and effective settings" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Return to Experiments" }));
    await screen.findByRole("heading", { name: "Local experiment" });
    expect(screen.queryByRole("region", { name: "Settings document inspector" })).toBeNull();
    expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(
      plan.sourceJson,
    );
    expect(BuiltKernelWorker.started).toBe(started + 2);
    expect(BuiltKernelWorker.activeCount).toBe(0);
    expect(transport.requests).toEqual(["/kernel.wasm"]);
  } finally {
    cleanup();
    await transport.close();
  }
});
