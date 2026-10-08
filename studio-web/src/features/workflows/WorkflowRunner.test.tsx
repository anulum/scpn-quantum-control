// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workspace workflow view tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { useRef, useState } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { instantiateKuramoto } from "../../panel/kuramoto";
import { readJson, writeJson } from "../../shared/contracts";
import { previewWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { useWorkspace } from "../workspace/useWorkspace";
import type { WorkflowExecutionResult } from "./workflowExecution";
import { runLocalWorkflow } from "./workflowExecution";
import { parseWorkflow, workflowDocument } from "./workflowModel";
import { readWorkflowArchive } from "./workflowArchive";
import type { WorkflowRunRequest } from "./useWorkflowRun";
import { WorkflowRunner } from "./WorkflowRunner";
import { useWorkflowRun } from "./useWorkflowRun";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);

function OriginalWorkspace({
  source = "",
  nativeRequest,
  onRevisionAttempt,
}: {
  readonly source?: string;
  readonly nativeRequest?: Omit<WorkflowRunRequest, "save">;
  readonly onRevisionAttempt?: (prior: string, signal: AbortSignal) => void;
}) {
  const workspace = useWorkspace(localExperimentCodecs);
  const run = useWorkflowRun(workspace.draft, true);
  const current = useRef(workspace);
  current.current = workspace;
  const [hostRefusal, setHostRefusal] = useState("");
  return (
    <>
      <button type="button" onClick={() => workspace.edit(source)}>
        Load original workflow source
      </button>
      <button
        type="button"
        onClick={() => {
          void workspace.create("Original empty workflow project");
        }}
      >
        Create original empty project
      </button>
      <button type="button" onClick={() => workspace.edit("")}>
        Clear current workflow source
      </button>
      {nativeRequest && (
        <button
          type="button"
          onClick={() => {
            void run
              .run({
                ...nativeRequest,
                save: async (archive, prior, signal) => {
                  if (signal.aborted || current.current.draft !== prior)
                    throw new Error("Original in-memory source adoption refused");
                  const admitted = await previewWorkspaceArchive(
                    archive.json,
                    localExperimentCodecs,
                  );
                  if (signal.aborted || current.current.draft !== prior)
                    throw new Error("Original in-memory source changed during admission");
                  current.current.edit(admitted.json);
                },
              })
              .catch((cause: unknown) => {
                setHostRefusal(cause instanceof Error ? cause.message : "Original host refused");
              });
          }}
        >
          Run through original native controller
        </button>
      )}
      <p>{hostRefusal}</p>
      <p>{workspace.message}</p>
      <output aria-label="Exact original workspace text">{workspace.draft}</output>
      <WorkflowRunner
        workspace={{
          ...workspace,
          saveRevision: async (archive, prior, signal) => {
            onRevisionAttempt?.(prior, signal);
            await workspace.saveRevision(archive, prior, signal);
          },
        }}
        run={run}
        rawCodecs={localExperimentCodecs}
        experimentBlocked={false}
        loadKernel={() => instantiateKuramoto(wasm)}
      />
    </>
  );
}
it("empty original workspace leaves graph composition and execution explicitly unavailable", async () => {
  render(<OriginalWorkspace />);
  expect(screen.getByRole("region", { name: "Reproducible workflows" })).toBeTruthy();
  await waitFor(() => expect(screen.getByText(/unavailable/i, { selector: "p" })).toBeTruthy());
  expect(screen.getByRole("button", { name: "Run or resume workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
    "disabled",
    true,
  );
});
it("composes a graph from actual WASM source and refuses storage without changing the original draft", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const original = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "original workflow view fixture",
    localExperimentCodecs,
    "native view test",
  );
  render(<OriginalWorkspace source={original.json} />);
  fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
      "disabled",
      false,
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Compose local workflow" }));
  await waitFor(() =>
    expect(screen.getByRole("list", { name: "Original workflow dependency graph" })).toBeTruthy(),
  );
  expect(screen.getByRole("button", { name: "Run or resume workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
      "disabled",
      false,
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
  await waitFor(() =>
    expect(screen.getByText(/Original workflow transaction refused/)).toBeTruthy(),
  );
  expect(screen.getByLabelText("Exact original workspace text").textContent).toBe(original.json);
});

let restored: WorkflowExecutionResult;
beforeAll(async () => {
  vi.stubGlobal("crypto", webcrypto);
  const kernel = await instantiateKuramoto(wasm);
  const original = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "restored original workflow history",
    localExperimentCodecs,
    "native restore test",
  );
  const baseline = await readLocalExperiment(original.json, localExperimentCodecs);
  const definition = parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "restored-native-history",
      stages: [
        {
          id: "simulate",
          adapter: "local-kuramoto",
          verb: "simulate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [],
          depends_on: [],
        },
      ],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
    },
    extensions: {},
  });
  let current = original;
  const result = await runLocalWorkflow({
    sourceJson: original.json,
    baseRevision: baseline.revisionHash,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    workerFactory: () => new BuiltKernelWorker(),
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  restored = result;
  vi.unstubAllGlobals();
}, 30000);
it("restores a genuinely completed original journal before another run without allocating a worker", async () => {
  const result = restored;
  expect(result.journal.state).toBe("complete");
  expect(BuiltKernelWorker.activeCount).toBe(0);
  const started = BuiltKernelWorker.started;
  render(<OriginalWorkspace source={result.archive.json} />);
  fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
  await waitFor(() =>
    expect(screen.getByRole("region", { name: "Original workflow attempt history" })).toBeTruthy(),
  );
  const history = screen.getByRole("region", { name: "Original workflow attempt history" });
  expect(history.textContent).toContain("Journal: complete");
  expect(history.textContent).toContain("recorded source");
  expect(history.textContent).toContain(result.journal.workflow_digest);
  expect(BuiltKernelWorker.started).toBe(started);
});

it.each(["invalid-json", "future-workflow"] as const)(
  "refuses %s source through the actual workspace view without allocating a worker",
  async (fault) => {
    const kernel = await instantiateKuramoto(wasm);
    const original = await createLocalExperiment(
      kernel,
      {
        mode: "mean-field",
        omega: [0.2, 0.2],
        theta0: [0, 0.8],
        coupling: 1.4,
        dt: 0.01,
        steps: 4,
      },
      "original refused workflow view",
      localExperimentCodecs,
      "native view test",
    );
    const wire = readJson(original.json) as { manifest: { extensions: Record<string, unknown> } };
    wire.manifest.extensions["experiment_workflows"] = { version: 2n, selected: null, items: [] };
    const source = fault === "invalid-json" ? "{" : writeJson(wire);
    const started = BuiltKernelWorker.started;
    render(<OriginalWorkspace source={source} />);
    fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
    await waitFor(() =>
      expect(
        screen.getByText(
          fault === "invalid-json"
            ? "Original workflow source refused; prior saved data retained"
            : "unsupported workflow archive version or history bound",
        ),
      ).toBeTruthy(),
    );
    expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
      "disabled",
      true,
    );
    expect(screen.getByLabelText("Exact original workspace text").textContent).toBe(source);
    expect(BuiltKernelWorker.started).toBe(started);
  },
);

it("shows a refused disposal receipt from a real native thread and blocks the next allocation", async () => {
  class RefusedReceiptWorker extends BuiltKernelWorker {
    override async terminate(): Promise<void> {
      await super.terminate();
      throw new Error("Native disposal receipt refused after actual thread exit");
    }
  }
  const original = await readWorkflowArchive(restored.archive.json, localExperimentCodecs);
  const selected = original.workflows.find((item) => item.hash === original.selected);
  if (selected === undefined) throw new Error("Genuine saved workflow missing");
  const document = workflowDocument(selected.definition);
  const body = document["body"];
  if (body === null || typeof body !== "object" || Array.isArray(body))
    throw new Error("Genuine saved workflow body missing");
  const definition = parseWorkflow({
    ...document,
    body: { ...body, workflow_id: "native-view-refused-receipt" },
  });
  render(
    <OriginalWorkspace
      source={restored.archive.json}
      nativeRequest={{
        definition,
        baseRevision: selected.base_revision_hash,
        kernel: await instantiateKuramoto(wasm),
        rawCodecs: localExperimentCodecs,
        workerFactory: () => new RefusedReceiptWorker(),
      }}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
  await waitFor(() =>
    expect(screen.getByText("Original saved graph and history restored")).toBeTruthy(),
  );
  const started = BuiltKernelWorker.started;
  fireEvent.click(screen.getByRole("button", { name: "Run through original native controller" }));
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain("worker disposal is unconfirmed"),
  );
  const history = screen.getByRole("region", { name: "Original workflow attempt history" });
  await waitFor(() => expect(history.textContent).toContain("Journal: partial"));
  expect(history.textContent).toContain("original worker disposal unconfirmed");
  expect(history.textContent).toContain("failed");
  expect(screen.getByRole("button", { name: "Run or resume workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(BuiltKernelWorker.started - started).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "Run through original native controller" }));
  await waitFor(() =>
    expect(
      screen.getByText("original worker is active or its disposal is unconfirmed"),
    ).toBeTruthy(),
  );
  expect(BuiltKernelWorker.started - started).toBe(1);
});

it.each(["admitted", "malformed"] as const)(
  "keeps an empty current view when a superseded %s source finishes admission",
  async (kind) => {
    const started = BuiltKernelWorker.started;
    render(<OriginalWorkspace source={kind === "admitted" ? restored.archive.json : "{"} />);
    fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear current workflow source" }));
    await act(async () => {});
    expect(screen.getByLabelText("Exact original workspace text").textContent).toBe("");
    expect(screen.getByText("Baseline revision: unavailable")).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Original workflow attempt history" })).toBeNull();
    expect(
      screen.queryByText("Original workflow source refused; prior saved data retained"),
    ).toBeNull();
    expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
      "disabled",
      true,
    );
    expect(BuiltKernelWorker.started).toBe(started);
  },
);

it("retains an original empty project without inventing an experiment baseline", async () => {
  const started = BuiltKernelWorker.started;
  render(<OriginalWorkspace />);
  fireEvent.click(screen.getByRole("button", { name: "Create original empty project" }));
  await waitFor(() =>
    expect(
      screen.getByText("Original workflow source refused; prior saved data retained"),
    ).toBeTruthy(),
  );
  expect(screen.getByText("Baseline revision: unavailable")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  const source = screen.getByLabelText("Exact original workspace text").textContent;
  if (source === null) throw new Error("Original empty project source missing");
  const admitted = await previewWorkspaceArchive(source, localExperimentCodecs);
  expect(admitted.documentHashes).toEqual([]);
  expect(admitted.rawHashes).toEqual([]);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("forwards an aborted view lifetime when a genuine graph save finishes after unmount", async () => {
  const attempts: { prior: string; signal: AbortSignal }[] = [];
  const started = BuiltKernelWorker.started;
  const view = render(
    <OriginalWorkspace
      source={restored.archive.json}
      onRevisionAttempt={(prior, signal) => {
        attempts.push({ prior, signal });
      }}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
  await waitFor(() =>
    expect(screen.getByText("Original saved graph and history restored")).toBeTruthy(),
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
      "disabled",
      false,
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
  view.unmount();
  await waitFor(() => expect(attempts).toHaveLength(1));
  expect(attempts[0]?.prior).toBe(restored.archive.json);
  expect(attempts[0]?.signal.aborted).toBe(true);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("passes the exact prior source to the original transaction when the editor changes during a graph save", async () => {
  const attempts: { prior: string; signal: AbortSignal }[] = [];
  render(
    <OriginalWorkspace
      source={restored.archive.json}
      onRevisionAttempt={(prior, signal) => {
        attempts.push({ prior, signal });
      }}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Load original workflow source" }));
  await waitFor(() =>
    expect(screen.getByText("Original saved graph and history restored")).toBeTruthy(),
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
      "disabled",
      false,
    ),
  );
  fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
  fireEvent.click(screen.getByRole("button", { name: "Clear current workflow source" }));
  await waitFor(() => expect(attempts).toHaveLength(1));
  expect(attempts[0]?.prior).toBe(restored.archive.json);
  expect(screen.getByLabelText("Exact original workspace text").textContent).toBe("");
  expect(screen.getByRole("button", { name: "Run or resume workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
