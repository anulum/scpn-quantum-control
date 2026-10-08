// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — continuously mounted original workflow ownership

import { useCallback, useEffect, useRef, useState } from "react";
import type { KuramotoKernel, OwnedKuramotoHandle } from "../../panel/kuramoto";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import type { OwnedKernelOutcome } from "../../workers/kernelProtocol";
import { runLocalWorkflow } from "./workflowExecution";
import type { WorkflowExecutionResult, WorkflowStageProgress } from "./workflowExecution";
import { WorkflowRefusal } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";

/** Observable ownership state; diagnostics always retain their original source. */
export type WorkflowRunStatus =
  | "idle"
  | "running"
  | "cancelling"
  | "complete"
  | "partial"
  | "cancelled"
  | "failed"
  | "stale";
/** Original source and transaction owner captured for one full graph run. */
export interface WorkflowRunRequest {
  /** Original immutable baseline selected by the user. */ readonly baseRevision: string;
  /** Complete original admitted graph. */ readonly definition: WorkflowDefinition;
  /** Actual original loaded WASM and native bounds. */ readonly kernel: KuramotoKernel;
  /** Trusted original source validators. */ readonly rawCodecs: ReadonlyMap<string, RawCodec>;
  /** Original conditional workspace transaction and source-ownership signal. */ readonly save: (
    archive: WorkspaceArchivePreview,
    priorJson: string,
    signal: AbortSignal,
  ) => Promise<void>;
  /** Original real transport seam for native qualification. */ readonly workerFactory?: () => KernelWorkerPort;
}
/** Single owner of the active graph, actual worker disposal and original progress. */
export interface WorkflowRunController {
  /** Current source-aware state. */ readonly status: WorkflowRunStatus;
  /** Last admitted saved archive/journal, retaining prior source diagnostics. */ readonly result: WorkflowExecutionResult | null;
  /** Original terminal outcome retained even when the source transaction fails. */ readonly terminal: OwnedKernelOutcome | null;
  /** Local traversal identity; it is not a backend execution or producer identity. */ readonly traversal: number;
  /** Actual original stage settlement, including source-checked reuse. */ readonly progress: WorkflowStageProgress | null;
  /** Authored current state/refusal text. */ readonly reason: string;
  /** An unconfirmed native disposal prevents every later allocation. */ readonly blocked: boolean;
  /** Run or resume the complete original graph through the actual native adapter. */ run(
    request: WorkflowRunRequest,
  ): Promise<void>;
  /** Request cancellation and await original worker and graph settlement. */ cancel(): Promise<void>;
}
interface State {
  readonly traversal: number;
  readonly progress: WorkflowStageProgress | null;
  readonly status: WorkflowRunStatus;
  readonly result: WorkflowExecutionResult | null;
  readonly terminal: OwnedKernelOutcome | null;
  readonly reason: string;
}

/** Keep graph ownership mounted across original view changes and cancel stale source completions. */
export function useWorkflowRun(sourceJson: string, active: boolean): WorkflowRunController {
  const [state, setState] = useState<State>({
    traversal: 0,
    progress: null,
    status: "idle",
    result: null,
    terminal: null,
    reason: "No original workflow has run",
  });
  const live = useRef(true),
    source = useRef(sourceJson),
    selected = useRef(active);
  source.current = sourceJson;
  selected.current = active;
  const generation = useRef(0),
    blocked = useRef(false);
  const acceptedSource = useRef<string | null>(null),
    pendingSource = useRef<string | null>(null);
  const observedSource = useRef(sourceJson);
  const sourceAcknowledgment = useRef<{
    readonly json: string;
    readonly resolve: () => void;
    readonly reject: (reason: WorkflowRefusal) => void;
  } | null>(null);
  const nativeCancellation = useRef<AbortController | null>(null),
    sourceOwnership = useRef<AbortController | null>(null);
  const worker = useRef<OwnedKuramotoHandle | null>(null),
    task = useRef<Promise<void> | null>(null);
  const sameSource = useCallback(
    () => source.current === acceptedSource.current || source.current === pendingSource.current,
    [],
  );
  const invalidate = useCallback(() => {
    ++generation.current;
    sourceOwnership.current?.abort();
    nativeCancellation.current?.abort();
    sourceAcknowledgment.current?.reject(
      new WorkflowRefusal("original workflow source acknowledgment cancelled"),
    );
    sourceAcknowledgment.current = null;
  }, []);
  useEffect(() => {
    observedSource.current = sourceJson;
    if (sourceAcknowledgment.current?.json === sourceJson) {
      sourceAcknowledgment.current.resolve();
      sourceAcknowledgment.current = null;
    }
    const matches = sourceJson === acceptedSource.current || sourceJson === pendingSource.current;
    if (task.current !== null && (!active || !matches)) {
      invalidate();
      setState((previous) => ({
        ...previous,
        status: "stale",
        reason: "Original source or view changed; prior diagnostics retain their original revision",
      }));
    }
  }, [sourceJson, active, invalidate]);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      invalidate();
    };
  }, [invalidate]);
  const isCurrent = (epoch: number) =>
    live.current && generation.current === epoch && selected.current && sameSource();
  return {
    ...state,
    blocked: blocked.current,
    async run(request): Promise<void> {
      if (!live.current || !selected.current)
        throw new WorkflowRefusal("original workflow view is inactive");
      if (task.current !== null || worker.current !== null || blocked.current)
        throw new WorkflowRefusal("original worker is active or its disposal is unconfirmed");
      const epoch = ++generation.current;
      acceptedSource.current = source.current;
      pendingSource.current = null;
      const cancellation = new AbortController(),
        ownership = new AbortController();
      nativeCancellation.current = cancellation;
      sourceOwnership.current = ownership;
      setState((previous) => ({
        ...previous,
        status: "running",
        traversal: epoch,
        progress: null,
        terminal: null,
        reason: "Running the original source-bound workflow",
      }));
      const operation = async () => {
        try {
          const result = await runLocalWorkflow({
            sourceJson: acceptedSource.current as string,
            baseRevision: request.baseRevision,
            definition: request.definition,
            kernel: request.kernel,
            rawCodecs: request.rawCodecs,
            signal: cancellation.signal,
            ...(request.workerFactory === undefined
              ? {}
              : { workerFactory: request.workerFactory }),
            save: async (candidate, priorJson) => {
              if (!isCurrent(epoch) || ownership.signal.aborted)
                throw new WorkflowRefusal(
                  "original workflow source changed before its transaction",
                );
              pendingSource.current = candidate.json;
              await request.save(candidate, priorJson, ownership.signal);
              if (!isCurrent(epoch) || ownership.signal.aborted)
                throw new WorkflowRefusal("original workflow source changed before acknowledgment");
              if (observedSource.current !== candidate.json)
                await new Promise<void>((resolve, reject) => {
                  sourceAcknowledgment.current = { json: candidate.json, resolve, reject };
                });
              if (!isCurrent(epoch) || ownership.signal.aborted)
                throw new WorkflowRefusal(
                  "original workflow source changed during its transaction",
                );
              acceptedSource.current = candidate.json;
              pendingSource.current = null;
            },
            onWorker: (handle) => {
              worker.current = handle;
              if (handle !== null)
                void handle.result.then((outcome) => {
                  if (!outcome.disposed) blocked.current = true;
                  if (live.current) setState((previous) => ({ ...previous, terminal: outcome }));
                });
            },
            onProgress: (progress) => {
              if (isCurrent(epoch)) setState((previous) => ({ ...previous, progress }));
            },
            onCheckpoint: (result) => {
              if (!result.disposalConfirmed) blocked.current = true;
              if (isCurrent(epoch)) setState((previous) => ({ ...previous, result }));
            },
          });
          if (!result.disposalConfirmed) blocked.current = true;
          if (isCurrent(epoch))
            setState((previous) => ({
              ...previous,
              result,
              status: result.journal.state,
              reason:
                result.journal.state === "complete"
                  ? "Original graph complete; recorded source results remain distinct from fresh executions"
                  : result.journal.state === "cancelled"
                    ? "Original cancellation checkpoint retained after observed disposal"
                    : "Original partial results retained; inspect failed and blocked stages",
            }));
        } catch (cause: unknown) {
          if (isCurrent(epoch))
            setState((previous) => ({
              ...previous,
              status: "failed",
              reason:
                cause instanceof WorkflowRefusal
                  ? cause.message
                  : "Original workflow or source transaction refused; prior data retained",
            }));
        } finally {
          task.current = null;
          nativeCancellation.current = null;
          sourceOwnership.current = null;
        }
      };
      const running = operation();
      task.current = running;
      await running;
    },
    async cancel(): Promise<void> {
      if (task.current === null) return;
      if (live.current)
        setState((previous) => ({
          ...previous,
          status: "cancelling",
          reason: "Observing original worker disposal and saving its partial journal",
        }));
      nativeCancellation.current?.abort();
      await task.current;
    },
  };
}
