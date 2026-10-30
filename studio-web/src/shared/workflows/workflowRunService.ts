// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workflow run service contract

import type { KuramotoKernel } from "../browser/kuramoto";
import type { RawCodec } from "../contracts";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import type { OwnedKernelOutcome } from "../workers/kernelProtocol";
import type { WorkflowExecutionResult, WorkflowStageProgress } from "./workflowExecution";
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
  /** Whether exact source text is the original accepted or pending own-save snapshot. */
  ownsSource(json: string): boolean;
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
