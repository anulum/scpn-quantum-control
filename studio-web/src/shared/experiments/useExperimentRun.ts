// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original source-bound experiment worker lifecycle

import { useCallback, useEffect, useRef, useState } from "react";
import { createOwnedKuramotoRun } from "../browser/kuramoto";
import type { OwnedKuramotoHandle } from "../browser/kuramoto";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import type { KernelWorkerEvent } from "../workers/kernelProtocol";
import { readLocalExperiment } from "./experimentArchive";
import { verifyReplayOutcome } from "./experimentPlan";
import { ExperimentRefusal } from "./kuramotoArtifacts";
import type { ExperimentRunController, ExperimentRunState } from "./experimentRunService";

/** Optional original transport seam; production uses the real browser Worker implementation. */
export interface ExperimentWorkerOptions {
  /** Adapter must own a genuinely disposable worker; it cannot supply synthetic success. */ readonly workerFactory?: () => KernelWorkerPort;
}

/** Preserve one actual worker, its immutable source identity and diagnostics across view navigation. */
export function useExperimentRun(
  sourceJson: string,
  active: boolean,
  options: ExperimentWorkerOptions = {},
): ExperimentRunController {
  const [state, setState] = useState<ExperimentRunState>({
    plan: null,
    expected: null,
    runId: "",
    attemptId: "",
    events: [],
    outcome: null,
    status: "idle",
    reason: "No local plan prepared",
    replayVerified: false,
  });
  const snapshot = useRef(state);
  snapshot.current = state;
  const currentSource = useRef(sourceJson);
  currentSource.current = sourceJson;
  const currentActive = useRef(active);
  currentActive.current = active;
  const live = useRef(true),
    generation = useRef(0),
    starting = useRef(false),
    blocked = useRef(false);
  const handle = useRef<OwnedKuramotoHandle | null>(null);
  const disposal = useRef<Promise<void>>(Promise.resolve());

  const dispose = useCallback(async (): Promise<void> => {
    const owned = handle.current;
    if (owned === null) return;
    // The active run owns terminal state and handle release; native disposal awaits that result.
    disposal.current = owned.dispose();
    await disposal.current;
  }, []);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      ++generation.current;
      void dispose();
    };
  }, [dispose]);
  useEffect(() => {
    if (!active || snapshot.current.plan?.sourceJson !== sourceJson) void dispose();
  }, [sourceJson, active, dispose]);

  const matches = state.plan !== null && state.plan.sourceJson === sourceJson;
  const status = !matches && state.plan !== null ? "stale" : state.status;
  const outcome = matches ? state.outcome : null;
  const occupied = starting.current || handle.current !== null;
  return {
    plan: state.plan,
    runId: state.runId,
    attemptId: state.attemptId,
    events: state.events,
    outcome,
    status,
    reason:
      matches || state.plan === null
        ? state.reason
        : "Source revision changed; prior attempt diagnostics retain their original revision",
    replayVerified: matches && state.replayVerified,
    disposalBlocked: blocked.current,
    canRun:
      active && matches && !occupied && !blocked.current && state.plan?.admission.allowed === true,
    canSave: active && matches && !occupied && !blocked.current && outcome?.disposed === true,
    async prepare(plan, expected): Promise<void> {
      if (!currentActive.current || currentSource.current !== plan.sourceJson)
        throw new ExperimentRefusal("plan belongs to an inactive or changed source archive");
      await dispose();
      await disposal.current;
      if (blocked.current)
        throw new ExperimentRefusal(
          "previous worker disposal unconfirmed; another run is forbidden",
        );
      if (!live.current || !currentActive.current || currentSource.current !== plan.sourceJson)
        throw new ExperimentRefusal("source changed while observing the prior worker");
      ++generation.current;
      setState({
        plan,
        expected: expected ?? null,
        runId: "",
        attemptId: "",
        events: [],
        outcome: null,
        status: plan.admission.allowed ? "planned" : "refused",
        reason: plan.admission.allowed
          ? "Original source and complete numerical plan admitted; execution requires Run experiment"
          : plan.admission.blockers.join(", "),
        replayVerified: false,
      });
    },
    async run(): Promise<void> {
      const captured = snapshot.current;
      const plan = captured.plan;
      if (!plan || !currentActive.current || currentSource.current !== plan.sourceJson)
        throw new ExperimentRefusal("prepare a current source-bound local plan before running");
      if (!plan.admission.allowed) {
        setState((previous) => ({
          ...previous,
          status: "refused",
          reason: plan.admission.blockers.join(", "),
        }));
        return;
      }
      if (starting.current || handle.current !== null || blocked.current)
        throw new ExperimentRefusal(
          "prior original worker is active or its disposal is unconfirmed",
        );
      starting.current = true;
      try {
        if (!live.current) throw new ExperimentRefusal("source changed before worker allocation");
        const epoch = ++generation.current,
          runId = crypto.randomUUID(),
          attemptId = crypto.randomUUID();
        const events: KernelWorkerEvent[] = [];
        setState({
          ...captured,
          runId,
          attemptId,
          events: [],
          outcome: null,
          status: "running",
          reason: "Starting the original source-bound worker",
          replayVerified: false,
        });
        const owned = createOwnedKuramotoRun({
          runId,
          revisionHash: plan.revisionHash,
          planHash: plan.planHash,
          buildFingerprint: plan.buildFingerprint,
          request: plan.request,
          wasmBytes: plan.wasmBytes,
          bounds: plan.bounds,
          resourcePolicy: plan.policy,
          deadlineMs: plan.deadlineMs,
          ...(options.workerFactory === undefined ? {} : { workerFactory: options.workerFactory }),
          onEvent: (event) => {
            if (!live.current || generation.current !== epoch) return;
            events.push(event);
            setState((previous) => ({
              ...previous,
              events: Object.freeze([...events]),
              reason: `Original worker: ${event.kind}, sequence ${event.sequence}`,
            }));
          },
        });
        handle.current = owned;
        starting.current = false;
        let terminal = await owned.result;
        // This run is the sole handle owner; preparation awaits its terminal result.
        handle.current = null;
        if (!terminal.disposed) blocked.current = true;
        let replayVerified = false;
        if (terminal.ok && captured.expected !== null) {
          try {
            verifyReplayOutcome(terminal, captured.expected);
            replayVerified = true;
          } catch (error: unknown) {
            terminal = {
              ok: false,
              code: "failed",
              reason:
                error instanceof ExperimentRefusal
                  ? error.message
                  : "original replay comparison failed",
              disposed: true,
            };
          }
        }
        if (
          !terminal.ok &&
          (terminal.code !== "cancelled" || events.at(-1)?.kind !== "cancelled")
        ) {
          events.push({
            version: 1,
            run_id: runId,
            sequence: (events.at(-1)?.sequence ?? 0) + 1,
            kind: terminal.code === "cancelled" ? "cancelled" : "failed",
            payload: {
              revision_hash: plan.revisionHash,
              plan_hash: plan.planHash,
              build_fingerprint: plan.buildFingerprint,
              disposed: terminal.disposed,
              reason: terminal.reason,
              code: terminal.code,
              origin: "original client outcome",
            },
          });
        }
        if (!live.current || generation.current !== epoch) return;
        setState((previous) => ({
          ...previous,
          events: Object.freeze([...events]),
          outcome: terminal,
          status: terminal.ok
            ? "succeeded"
            : terminal.code === "cancelled"
              ? "cancelled"
              : "failed",
          reason: terminal.ok
            ? replayVerified
              ? "Original saved float64 output replayed exactly after observed worker disposal"
              : "Original worker result received after observed disposal"
            : terminal.reason,
          replayVerified,
        }));
      } finally {
        starting.current = false;
      }
    },
    async cancel(): Promise<void> {
      if (handle.current === null && !starting.current) return;
      setState((previous) => ({
        ...previous,
        status: "cancelling",
        reason: "Observing original worker disposal; no successful result is claimed",
      }));
      await dispose();
    },
    async retainSavedAttempt(archive, priorJson, rawCodecs): Promise<void> {
      const captured = snapshot.current;
      if (
        captured.plan === null ||
        captured.plan.sourceJson !== priorJson ||
        captured.outcome?.disposed !== true ||
        handle.current !== null ||
        blocked.current
      )
        throw new ExperimentRefusal(
          "a disposed unchanged-source attempt is required to retain its archived result",
        );
      const selected = await readLocalExperiment(archive.json, rawCodecs);
      if (
        !live.current ||
        snapshot.current !== captured ||
        (currentSource.current !== priorJson && currentSource.current !== archive.json) ||
        selected.revisionHash !== captured.plan.revisionHash ||
        selected.kernelHash !== captured.plan.buildFingerprint
      )
        throw new ExperimentRefusal(
          "saved archive changed the immutable source; prior result cannot bind to it",
        );
      setState({
        ...captured,
        plan: Object.freeze({ ...captured.plan, sourceJson: archive.json }),
      });
    },
  };
}
