// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — experiment run service contract

import type { RawCodec } from "../contracts";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";
import type { KernelWorkerEvent, OwnedKernelOutcome } from "../workers/kernelProtocol";
import type { ExperimentPlan, ReplayExpectation } from "./experimentPlan";

/** Public lifecycle statuses; stale diagnostics never authorize a new revision's result. */
export type ExperimentRunStatus =
  | "idle"
  | "planned"
  | "running"
  | "cancelling"
  | "succeeded"
  | "refused"
  | "cancelled"
  | "failed"
  | "stale";
/** Observable state of one experiment run. */
export interface ExperimentRunState {
  /** Prepared immutable admission, or null before explicit preparation. */ readonly plan: ExperimentPlan | null;
  /** Saved exact output used only for an explicitly prepared replay. */ readonly expected: ReplayExpectation | null;
  /** Original native run UUID, empty before allocation. */ readonly runId: string;
  /** Independent immutable archive attempt UUID, empty before allocation. */ readonly attemptId: string;
  /** Actual ordered worker and original client outcome diagnostics. */ readonly events: readonly KernelWorkerEvent[];
  /** Actual disposed outcome for the selected unchanged source, or null. */ readonly outcome: OwnedKernelOutcome | null;
  /** Current original plan, execution and disposal phase. */ readonly status: ExperimentRunStatus;
  /** Authored lifecycle status or preserved original worker diagnostic. */ readonly reason: string;
  /** Exact saved output comparison passed after genuine worker disposal. */ readonly replayVerified: boolean;
}

/** Original worker execution controls, owned by the continuously mounted workbench. */
export interface ExperimentRunController extends Omit<ExperimentRunState, "expected"> {
  /** Whether a current original source plan can execute with one owned worker. */ readonly canRun: boolean;
  /** Whether a disposed attempt can be committed to that unchanged source. */ readonly canSave: boolean;
  /** Disposal failed; starting another run remains forbidden in this mount. */ readonly disposalBlocked: boolean;
  /** Admit a prepared exact source/plan binding without executing it. */ prepare(
    plan: ExperimentPlan,
    expected?: ReplayExpectation,
  ): Promise<void>;
  /** Start the original source-owned disposable worker after final admission. */ run(): Promise<void>;
  /** Observe cancellation; source draft and already captured diagnostic events remain. */ cancel(): Promise<void>;
  /** Retain the same immutable result after an admitted archive-only save, never after a revision edit. */ retainSavedAttempt(
    archive: WorkspaceArchivePreview,
    priorJson: string,
    rawCodecs: ReadonlyMap<string, RawCodec>,
  ): Promise<void>;
}
