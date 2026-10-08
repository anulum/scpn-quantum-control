// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workspace reproducible workflow journey

import { useEffect, useRef, useState } from "react";
import { fetchKuramoto } from "../../panel/kuramoto";
import type { KuramotoKernel } from "../../panel/kuramoto";
import { canonicalDigest, writeJson } from "../../shared/contracts";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceController } from "../workspace/useWorkspace";
import { readLocalExperiment } from "../experiments/experimentArchive";
import { archiveWorkflow, readWorkflowArchive } from "./workflowArchive";
import type { ArchivedWorkflow } from "./workflowArchive";
import { WorkflowEditor } from "./WorkflowEditor";
import { parseWorkflow, WorkflowRefusal, workflowDocument } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";
import type { WorkflowRunController } from "./useWorkflowRun";

/** Original continuously mounted workspace and actual worker ownership controls. */
export interface WorkflowRunnerProps {
  /** The original shared workspace store and exact source text. */ readonly workspace: WorkspaceController;
  /** Trusted original product/host source verifiers. */ readonly rawCodecs: ReadonlyMap<
    string,
    RawCodec
  >;
  /** The same mounted graph owner through source and route changes. */ readonly run: WorkflowRunController;
  /** Original standalone experiment owns an active or unconfirmed worker. */ readonly experimentBlocked: boolean;
  /** Original kernel loader seam; production fetches only its shipped asset. */ readonly loadKernel?: () => Promise<KuramotoKernel>;
}

function localGraph(): WorkflowDefinition {
  const type = { schema: "studio.workflow-run-reference.v1", dtype: "json", shape: [], unit: "1" };
  return parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "local-classical-workflow",
      stages: [
        {
          id: "validate",
          adapter: "local-kuramoto",
          verb: "validate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [],
          depends_on: [],
        },
        {
          id: "simulate",
          adapter: "local-kuramoto",
          verb: "simulate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [{ name: "run", path: ["run_ref"], type }],
          depends_on: ["validate"],
        },
        {
          id: "analyse",
          adapter: "local-kuramoto",
          verb: "analyse",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [{ parameter: "run_ref", source_stage: "simulate", source_port: "run", type }],
          outputs: [],
          depends_on: [],
        },
      ],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 3n },
    },
    extensions: {
      model: "original classical Kuramoto",
      seed_boundary: "Cell identity only; this deterministic source does not consume random seeds",
    },
  });
}
function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

/** Compose, save and resume original graphs while retaining prior complete and partial workspace evidence. */
export function WorkflowRunner({
  workspace,
  rawCodecs,
  run,
  experimentBlocked,
  loadKernel = fetchKuramoto,
}: WorkflowRunnerProps) {
  const [selection, setSelection] = useState<ArchivedWorkflow | null>(null),
    [history, setHistory] = useState<readonly ArchivedWorkflow[]>([]);
  const [baseline, setBaseline] = useState<string | null>(null),
    [message, setMessage] = useState("Import or open an original experiment to compose a workflow");
  const [pending, setPending] = useState(false);
  const saveOwnership = useRef(new AbortController());
  const live = useRef(true),
    generation = useRef(0),
    currentWorkspace = useRef(workspace);
  currentWorkspace.current = workspace;
  const running = run.status === "running" || run.status === "cancelling";
  const activeResult =
    selection !== null && run.result?.journal.workflow_digest === selection.hash
      ? run.result
      : null;
  const journal = activeResult?.journal ?? selection?.journal ?? null;
  const occurrences = new Map<string, number>();
  const attempts = (journal?.entries ?? []).map((entry) => {
    const identity = `${entry.cell_id}:${entry.stage_id}:${entry.fingerprint}`;
    const ordinal = occurrences.get(identity) ?? 0;
    occurrences.set(identity, ordinal + 1);
    return { entry, key: `${identity}:${ordinal}` };
  });
  useEffect(() => {
    const epoch = ++generation.current;
    if (running) return;
    const open = async () => {
      if (workspace.draft.length === 0) {
        setSelection(null);
        setHistory([]);
        setBaseline(null);
        return;
      }
      try {
        const original = await readWorkflowArchive(workspace.draft, rawCodecs);
        const selected = original.workflows.find((item) => item.hash === original.selected) ?? null;
        const sourceBaseline =
          selected === null
            ? (await readLocalExperiment(workspace.draft, rawCodecs)).revisionHash
            : selected.base_revision_hash;
        if (!live.current || generation.current !== epoch) return;
        setSelection(selected);
        setHistory(original.workflows);
        setBaseline(sourceBaseline);
        setMessage(
          selected === null
            ? "Original baseline admitted; compose a graph before running"
            : "Original saved graph and history restored",
        );
      } catch (cause: unknown) {
        if (!live.current || generation.current !== epoch) return;
        setSelection(null);
        setHistory([]);
        setBaseline(null);
        setMessage(
          cause instanceof WorkflowRefusal
            ? cause.message
            : "Original workflow source refused; prior saved data retained",
        );
      }
    };
    void open();
  }, [workspace.draft, rawCodecs, running]);
  useEffect(() => {
    saveOwnership.current = new AbortController();
    live.current = true;
    return () => {
      live.current = false;
      saveOwnership.current.abort();
      ++generation.current;
    };
  }, []);
  const disabled = pending || running || workspace.busy || experimentBlocked || run.blocked;
  const transact = async (operation: () => Promise<void>) => {
    setPending(true);
    try {
      await operation();
    } catch (cause: unknown) {
      if (live.current)
        setMessage(
          cause instanceof WorkflowRefusal
            ? cause.message
            : "Original workflow operation refused; prior evidence retained",
        );
    } finally {
      if (live.current) setPending(false);
    }
  };
  const saveDefinition =
    baseline === null
      ? undefined
      : async (definition: WorkflowDefinition) => {
          const captured = currentWorkspace.current,
            prior = captured.draft,
            signal = saveOwnership.current.signal;
          const original = await readWorkflowArchive(prior, rawCodecs);
          const hash = await canonicalDigest(
            "studio.workflow-definition.v1",
            workflowDocument(definition),
          );
          const previous = original.workflows.find((item) => item.hash === hash);
          const saved = await archiveWorkflow(
            prior,
            previous?.base_revision_hash ?? baseline,
            definition,
            previous?.journal ?? null,
            rawCodecs,
          );
          await currentWorkspace.current.saveRevision(saved, prior, signal);
        };
  return (
    <section
      aria-label="Reproducible workflows"
      data-workflow-traversal={run.traversal}
      data-workflow-settled={
        run.progress?.workflowDigest === selection?.hash ? (run.progress?.settledStages ?? 0) : 0
      }
    >
      <h3>Reproducible workflows</h3>
      <p>
        Compose typed stages, bounded parameter grids and explicit cell seeds. The local browser
        runs the original classical Kuramoto WASM. Quantum executive graphs export to the local
        workflow CLI.
      </p>
      <p role="status">{message}</p>
      <p>Baseline revision: {baseline ?? "unavailable"}</p>
      <WorkflowEditor
        definition={selection?.definition ?? null}
        disabled={disabled || baseline === null}
        onSave={saveDefinition}
        onCompose={
          baseline === null
            ? undefined
            : async () => {
                await readLocalExperiment(currentWorkspace.current.draft, rawCodecs, baseline);
                return localGraph();
              }
        }
      />
      <button
        type="button"
        disabled={
          disabled || selection === null || baseline === null || !workspace.storageAvailable
        }
        onClick={
          selection === null || baseline === null
            ? undefined
            : () => {
                void transact(async () => {
                  const kernel = await loadKernel();
                  await run.run({
                    definition: selection.definition,
                    baseRevision: baseline,
                    kernel,
                    rawCodecs,
                    save: async (archive, priorJson, signal) =>
                      currentWorkspace.current.saveRevision(archive, priorJson, signal),
                  });
                });
              }
        }
      >
        Run or resume workflow
      </button>
      <button
        type="button"
        disabled={!running}
        onClick={() => {
          void run.cancel();
        }}
      >
        Cancel workflow
      </button>
      <button
        type="button"
        disabled={selection === null}
        onClick={
          selection === null
            ? undefined
            : () => {
                download(
                  "experiment-workflow.json",
                  writeJson(workflowDocument(selection.definition)),
                );
              }
        }
      >
        Export workflow JSON
      </button>
      <p role="status">{run.reason}</p>
      {run.progress?.workflowDigest === selection?.hash && run.progress !== null && (
        <p role="status" aria-label="Workflow stage progress">
          {run.progress.settledStages}/{run.progress.totalStages} original stages settled ·{" "}
          {run.progress.stageId} ·{" "}
          {run.progress.reused
            ? "original source and plan revalidated"
            : "original attempt retained"}
        </p>
      )}
      {journal !== null && (
        <section aria-label="Original workflow attempt history">
          <p>
            Journal: {journal.state} · reserved evaluations {journal.evaluations.toString()} ·
            original worker disposal{" "}
            {activeResult === null
              ? "recorded source"
              : activeResult.disposalConfirmed
                ? "confirmed"
                : "unconfirmed"}
          </p>
          <p>Original graph: {journal.workflow_digest}</p>
          <ol>
            {attempts.map(({ entry, key }) => (
              <li key={key}>
                Cell {entry.cell_id} · {entry.stage_id} · {entry.status}
                {entry.reason === null ? "" : ` · ${entry.reason}`}
              </li>
            ))}
          </ol>
        </section>
      )}
      <details>
        <summary>Retained graphs and partial journals</summary>
        <ul>
          {history.map((item) => (
            <li key={item.hash}>
              {item.definition.workflow_id} · {item.hash} · {item.journal?.state ?? "not run"} ·{" "}
              {item.journal?.entries.length ?? 0} original attempts
            </li>
          ))}
        </ul>
      </details>
      {run.blocked && (
        <p role="alert">Original worker disposal is unconfirmed. Another allocation is blocked.</p>
      )}
    </section>
  );
}

export default WorkflowRunner;
