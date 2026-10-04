// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original source-bound experiment worker lifecycle

import { useEffect, useRef, useState } from "react";
import { createOwnedKuramotoRun } from "../../panel/kuramoto";
import type { OwnedKuramotoHandle } from "../../panel/kuramoto";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import type { KernelWorkerEvent, OwnedKernelOutcome } from "../../workers/kernelProtocol";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { readLocalExperiment } from "./experimentArchive";
import { verifyReplayOutcome } from "./experimentPlan";
import type { ExperimentPlan, ReplayExpectation } from "./experimentPlan";
import { ExperimentRefusal } from "./kuramotoArtifacts";

/** Public lifecycle statuses; stale diagnostics never authorize a new revision's result. */
export type ExperimentRunStatus = "idle" | "planned" | "running" | "cancelling" | "succeeded" | "refused" | "cancelled" | "failed" | "stale";
interface RunState {
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
export interface ExperimentRunController extends Omit<RunState, "expected"> {
  /** Whether a current original source plan can execute with one owned worker. */ readonly canRun: boolean;
  /** Whether a disposed attempt can be committed to that unchanged source. */ readonly canSave: boolean;
  /** Disposal failed; starting another run remains forbidden in this mount. */ readonly disposalBlocked: boolean;
  /** Admit a prepared exact source/plan binding without executing it. */ prepare(plan: ExperimentPlan, expected?: ReplayExpectation): Promise<void>;
  /** Start the original source-owned disposable worker after final admission. */ run(): Promise<void>;
  /** Observe cancellation; source draft and already captured diagnostic events remain. */ cancel(): Promise<void>;
  /** Retain the same immutable result after an admitted archive-only save, never after a revision edit. */ retainSavedAttempt(archive: WorkspaceArchivePreview, priorJson: string, rawCodecs: ReadonlyMap<string, RawCodec>): Promise<void>;
}

/** Optional original transport seam; production uses the real browser Worker implementation. */
export interface ExperimentWorkerOptions {
  /** Adapter must own a genuinely disposable worker; it cannot supply synthetic success. */ readonly workerFactory?: () => KernelWorkerPort;
}

/** Preserve one actual worker, its immutable source identity and diagnostics across view navigation. */
export function useExperimentRun(sourceJson: string, active: boolean, options: ExperimentWorkerOptions = {}): ExperimentRunController {
  const [state, setState] = useState<RunState>({ plan: null, expected: null, runId: "", attemptId: "", events: [], outcome: null, status: "idle", reason: "No local plan prepared", replayVerified: false });
  const snapshot = useRef(state); snapshot.current = state;
  const currentSource = useRef(sourceJson); currentSource.current = sourceJson;
  const currentActive = useRef(active); currentActive.current = active;
  const live = useRef(true), generation = useRef(0), starting = useRef(false), blocked = useRef(false);
  const handle = useRef<OwnedKuramotoHandle | null>(null);
  const disposal = useRef<Promise<void>>(Promise.resolve());

  const dispose = async (): Promise<void> => {
    const owned = handle.current;
    if (owned === null) return;
    // The active run owns terminal state and handle release; native disposal awaits that result.
    disposal.current = owned.dispose();
    await disposal.current;
  };
  useEffect(() => {
    live.current = true;
    return () => { live.current = false; ++generation.current; void dispose(); };
  }, []);
  useEffect(() => {
    if (!active || snapshot.current.plan?.sourceJson !== sourceJson) void dispose();
  }, [sourceJson, active]);

  const matches = state.plan !== null && state.plan.sourceJson === sourceJson;
  const status = !matches && state.plan !== null ? "stale" : state.status;
  const outcome = matches ? state.outcome : null;
  const occupied = starting.current || handle.current !== null;
  return {
    plan: state.plan, runId: state.runId, attemptId: state.attemptId, events: state.events, outcome, status,
    reason: matches || state.plan === null ? state.reason : "Source revision changed; prior attempt diagnostics retain their original revision",
    replayVerified: matches && state.replayVerified, disposalBlocked: blocked.current,
    canRun: active && matches && !occupied && !blocked.current && state.plan!.admission.allowed,
    canSave: active && matches && !occupied && !blocked.current && outcome !== null && outcome.disposed,
    async prepare(plan, expected): Promise<void> {
      if (!currentActive.current || currentSource.current !== plan.sourceJson) throw new ExperimentRefusal("plan belongs to an inactive or changed source archive");
      await dispose(); await disposal.current;
      if (blocked.current) throw new ExperimentRefusal("previous worker disposal unconfirmed; another run is forbidden");
      if (!live.current || !currentActive.current || currentSource.current !== plan.sourceJson) throw new ExperimentRefusal("source changed while observing the prior worker");
      ++generation.current;
      setState({ plan, expected: expected ?? null, runId: "", attemptId: "", events: [], outcome: null,
        status: plan.admission.allowed ? "planned" : "refused", reason: plan.admission.allowed ? "Original source and complete numerical plan admitted; execution requires Run experiment" : plan.admission.blockers.join(", "), replayVerified: false });
    },
    async run(): Promise<void> {
      const captured = snapshot.current;
      const plan = captured.plan;
      if (!plan || !currentActive.current || currentSource.current !== plan.sourceJson) throw new ExperimentRefusal("prepare a current source-bound local plan before running");
      if (!plan.admission.allowed) { setState(previous => ({ ...previous, status: "refused", reason: plan.admission.blockers.join(", ") })); return; }
      if (starting.current || handle.current !== null || blocked.current) throw new ExperimentRefusal("prior original worker is active or its disposal is unconfirmed");
      starting.current = true;
      try {
        if (!live.current) throw new ExperimentRefusal("source changed before worker allocation");
        const epoch = ++generation.current, runId = crypto.randomUUID(), attemptId = crypto.randomUUID();
        const events: KernelWorkerEvent[] = [];
        setState({ ...captured, runId, attemptId, events: [], outcome: null, status: "running", reason: "Starting the original source-bound worker", replayVerified: false });
        const owned = createOwnedKuramotoRun({ runId, revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint,
          request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs,
          ...(options.workerFactory === undefined ? {} : { workerFactory: options.workerFactory }),
          onEvent: event => {
            if (!live.current || generation.current !== epoch) return;
            events.push(event);
            setState(previous => ({ ...previous, events: Object.freeze([...events]), reason: `Original worker: ${event.kind}, sequence ${event.sequence}` }));
          } });
        handle.current = owned; starting.current = false;
        let terminal = await owned.result;
        // This run is the sole handle owner; preparation awaits its terminal result.
        handle.current = null;
        if (!terminal.disposed) blocked.current = true;
        let replayVerified = false;
        if (terminal.ok && captured.expected !== null) {
          try { verifyReplayOutcome(terminal, captured.expected); replayVerified = true; }
          catch (error: unknown) { terminal = { ok: false, code: "failed", reason: error instanceof ExperimentRefusal ? error.message : "original replay comparison failed", disposed: true }; }
        }
        if (!terminal.ok && (terminal.code !== "cancelled" || events.at(-1)?.kind !== "cancelled")) {
          events.push({ version: 1, run_id: runId, sequence: (events.at(-1)?.sequence ?? 0) + 1, kind: terminal.code === "cancelled" ? "cancelled" : "failed",
            payload: { revision_hash: plan.revisionHash, plan_hash: plan.planHash, build_fingerprint: plan.buildFingerprint, disposed: terminal.disposed,
              reason: terminal.reason, code: terminal.code, origin: "original client outcome" } });
        }
        if (!live.current || generation.current !== epoch) return;
        setState(previous => ({ ...previous, events: Object.freeze([...events]), outcome: terminal,
          status: terminal.ok ? "succeeded" : terminal.code === "cancelled" ? "cancelled" : "failed",
          reason: terminal.ok ? replayVerified ? "Original saved float64 output replayed exactly after observed worker disposal" : "Original worker result received after observed disposal" : terminal.reason, replayVerified }));
      } finally { starting.current = false; }
    },
    async cancel(): Promise<void> {
      if (handle.current === null && !starting.current) return;
      setState(previous => ({ ...previous, status: "cancelling", reason: "Observing original worker disposal; no successful result is claimed" }));
      await dispose();
    },
    async retainSavedAttempt(archive, priorJson, rawCodecs): Promise<void> {
      const captured = snapshot.current;
      if (captured.plan === null || captured.plan.sourceJson !== priorJson || captured.outcome?.disposed !== true || handle.current !== null || blocked.current) throw new ExperimentRefusal("a disposed unchanged-source attempt is required to retain its archived result");
      const selected = await readLocalExperiment(archive.json, rawCodecs);
      if (!live.current || snapshot.current !== captured || (currentSource.current !== priorJson && currentSource.current !== archive.json)
        || selected.revisionHash !== captured.plan.revisionHash || selected.kernelHash !== captured.plan.buildFingerprint) throw new ExperimentRefusal("saved archive changed the immutable source; prior result cannot bind to it");
      setState({ ...captured, plan: Object.freeze({ ...captured.plan, sourceJson: archive.json }) });
    },
  };
}
