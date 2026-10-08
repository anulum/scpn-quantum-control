// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native workflow view refusal and lifetime

import { flushSync } from "react-dom";
import { createRoot } from "react-dom/client";
import { fetchKuramoto } from "../src/panel/kuramoto";
import { localExperimentCodecs } from "../src/features/experiments/kuramotoArtifacts";
import { useWorkspace } from "../src/features/workspace/useWorkspace";
import type { WorkspaceController } from "../src/features/workspace/useWorkspace";
import { WorkflowRunner } from "../src/features/workflows/WorkflowRunner";
import { useWorkflowRun } from "../src/features/workflows/useWorkflowRun";

async function tick(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

/** Exercise the real view, native store, kernel loader and controller without allocating after refusal or unmount. */
export async function runNativeWorkflowViewRefusals(): Promise<Record<string, unknown>> {
  const receipts: Record<string, unknown> = {};
  const observed = (
    window as unknown as {
      __ownedKernel: { started: number; active: number };
    }
  ).__ownedKernel;
  if (observed === undefined) throw new Error("Actual native worker observation is required");
  for (const phase of ["inactive", "disposed"] as const) {
    const container = document.createElement("main");
    document.body.append(container);
    const root = createRoot(container);
    const state: { workspace: WorkspaceController | null } = { workspace: null };
    let release!: () => void;
    const loading = new Promise<void>((resolve) => {
      release = resolve;
    });
    let settle!: () => void;
    const settled = new Promise<void>((resolve) => {
      settle = resolve;
    });
    let loadStarted = false;
    let unmounted = false;
    function Host() {
      const workspace = useWorkspace(localExperimentCodecs);
      const original = useWorkflowRun(workspace.draft, phase !== "inactive");
      state.workspace = workspace;
      return (
        <WorkflowRunner
          workspace={workspace}
          rawCodecs={localExperimentCodecs}
          experimentBlocked={false}
          loadKernel={async () => {
            loadStarted = true;
            await loading;
            return fetchKuramoto();
          }}
          run={{
            ...original,
            run: async (request) => {
              try {
                await original.run(request);
              } finally {
                settle();
              }
            },
          }}
        />
      );
    }
    const started = observed.started;
    try {
      flushSync(() => root.render(<Host />));
      const deadline = performance.now() + 10_000;
      let run: HTMLButtonElement | null = null;
      while (run === null || run.disabled) {
        if (performance.now() > deadline)
          throw new Error("Original saved workflow view did not become available");
        await tick();
        run =
          [...container.querySelectorAll("button")].find(
            (button) => button.textContent === "Run or resume workflow",
          ) ?? null;
      }
      const prior = state.workspace?.draft;
      if (prior === undefined || prior.length === 0 || !state.workspace?.storageAvailable)
        throw new Error("Actual native saved workspace is required");
      run.click();
      if (!loadStarted) throw new Error("Original kernel loading did not start through the view");
      if (phase === "disposed") {
        flushSync(() => root.unmount());
        unmounted = true;
      }
      release();
      await settled;
      await tick();
      if (observed.started !== started || observed.active !== 0)
        throw new Error(
          `Refused ${phase} view changed native workers: started ${started}->${observed.started}, active ${observed.active}`,
        );
      if (state.workspace?.draft !== prior)
        throw new Error("Refused view replaced its original saved source");
      while (
        phase === "inactive" &&
        !container.textContent?.includes("original workflow view is inactive")
      ) {
        if (performance.now() > deadline)
          throw new Error(
            `Actual inactive controller refusal was not visible: ${container.textContent}`,
          );
        await tick();
      }
      if (phase === "disposed" && container.childElementCount !== 0)
        throw new Error("Disposed view rendered a late completion");
      receipts[phase] = {
        loadStarted,
        unmounted,
        sourceRetained: true,
        workersAllocated: observed.started - started,
      };
    } finally {
      release();
      if (!unmounted) flushSync(() => root.unmount());
      container.remove();
    }
  }
  return receipts;
}
