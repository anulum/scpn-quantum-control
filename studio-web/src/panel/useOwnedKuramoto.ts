// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — one owned kernel run at a time per mounted panel

import { useEffect, useRef, useState } from "react";
import { canonicalDigest } from "../shared/contracts/canonical";
import type { ResourcePolicy } from "../shared/resources/admission";
import { kernelBinaryFingerprint, kernelBinarySize, ownedWorkerBinary } from "../workers/kernelProtocol";
import { admitOwnedKuramotoResources } from "../shared/resources/kuramotoResources";
import { createOwnedKuramotoRun, encodeKuramotoInput } from "./kuramoto";
import type { KuramotoKernel, KuramotoRequest, OwnedKernelOutcome, OwnedKuramotoHandle } from "./kuramoto";

/** Immutable admitted input owned by the current panel revision. */
export interface OwnedKuramotoJob {
  /** Original loaded source; its binary is checked inside every worker. */
  readonly kernel: KuramotoKernel;
  /** Deterministic source values, independent of UI display sampling. */
  readonly request: KuramotoRequest;
  /** Source-declared resource ceilings checked before transfer allocation. */
  readonly resourcePolicy: ResourcePolicy;
  /** Optional original committed reference, run sequentially after playback. */
  readonly reference: KuramotoRequest | null;
  /** Operational disposal deadline for each worker, not a hard wall guarantee. */
  readonly deadlineMs: number;
}

/** Results are visible only for the currently admitted input revision. */
export interface OwnedKuramotoState {
  /** State of the current owned operation. */
  readonly phase: "idle" | "running" | "finished";
  /** Playback outcome; no trajectory survives cancellation or a changed job. */
  readonly result: OwnedKernelOutcome | null;
  /** Separate original reference outcome, never substituted for playback. */
  readonly reference: OwnedKernelOutcome | null;
  /** Cancel and observe disposal of the current worker. */
  readonly cancel: () => Promise<void>;
}

interface CapturedState {
  readonly job: OwnedKuramotoJob | null;
  readonly phase: OwnedKuramotoState["phase"];
  readonly result: OwnedKernelOutcome | null;
  readonly reference: OwnedKernelOutcome | null;
}

/** Dispose the prior revision before starting another real worker; ignore all stale completion. */
export function useOwnedKuramoto(job: OwnedKuramotoJob | null): OwnedKuramotoState {
  const [state, setState] = useState<CapturedState>({ job: null, phase: "idle", result: null, reference: null });
  const tail = useRef<Promise<void>>(Promise.resolve());
  const undisposed = useRef<Extract<OwnedKernelOutcome, { ok: false }> | null>(null);
  const cancelCurrent = useRef<() => Promise<void>>(async () => {});

  useEffect(() => {
    let live = true;
    let cancelled = false;
    let handle: OwnedKuramotoHandle | null = null;
    const previous = tail.current;
    let finish!: () => void;
    tail.current = new Promise<void>(resolve => { finish = resolve; });
    const publish = (phase: CapturedState["phase"], result: OwnedKernelOutcome | null, reference: OwnedKernelOutcome | null) => {
      if (live) setState({ job, phase, result, reference });
    };
    const cancelledOutcome = (): OwnedKernelOutcome => ({ ok: false, code: "cancelled", reason: "owned run cancelled after worker disposal", disposed: true });
    const remember = (outcome: OwnedKernelOutcome): OwnedKernelOutcome => {
      if (!outcome.disposed) undisposed.current = outcome;
      return outcome;
    };
    cancelCurrent.current = async () => {
      cancelled = true;
      const owned = handle;
      if (owned !== null) {
        await owned.cancel();
        publish("finished", remember(await owned.result), null);
      } else publish("finished", undisposed.current ?? cancelledOutcome(), null);
    };

    const execute = async () => {
      await previous;
      if (!live || cancelled) return;
      if (job === null) { publish("idle", null, null); return; }
      if (undisposed.current !== null) { publish("finished", undisposed.current, null); return; }
      publish("running", null, null);
      try {
        const bytes = job.kernel.sourceBytes;
        if (!ownedWorkerBinary(bytes)) throw new Error("original WASM binary unavailable for owned execution");
        const admission = admitOwnedKuramotoResources({ n: job.request.omega.length, steps: job.request.steps, mode: job.request.mode }, job.kernel.bounds, kernelBinarySize(bytes), job.resourcePolicy);
        if (!admission.allowed) throw new Error(`worker resource policy refused: ${admission.blockers.join(", ")}`);
        if (encodeKuramotoInput(job.request) === null || (job.request.mode === "mean-field" && (job.request.kNm?.length ?? 0) !== 0)) throw new Error("original source request refused before identity allocation");
        const fingerprint = await kernelBinaryFingerprint(bytes);
        const run = async (request: KuramotoRequest, policy: ResourcePolicy): Promise<OwnedKernelOutcome> => {
          const revisionHash = await canonicalDigest("scpn.studio.kernel-input.v1", request);
          const planHash = await canonicalDigest("scpn.studio.kernel-plan.v1", { method: request.mode, n: request.omega.length, steps: request.steps, dt: request.dt, buildFingerprint: fingerprint, resourcePolicy: policy });
          if (!live || cancelled) return cancelledOutcome();
          handle = createOwnedKuramotoRun({ runId: crypto.randomUUID(), revisionHash, planHash, buildFingerprint: fingerprint, request, wasmBytes: bytes, bounds: job.kernel.bounds, resourcePolicy: policy, deadlineMs: job.deadlineMs });
          const owned = handle;
          const outcome = remember(await owned.result);
          handle = null;
          return outcome;
        };
        const policy = admission.policy;
        const result = await run(job.request, policy);
        if (!live || cancelled) return;
        let reference: OwnedKernelOutcome | null = null;
        if (result.ok && job.reference !== null) {
          const retainedBytes = BigInt(result.run.orderParameter.byteLength + result.run.thetaFinal.byteLength);
          // Successful admission requires a known overhead and freezes its policy.
          const referencePolicy = { ...policy, overheadBytes: policy.overheadBytes! + retainedBytes };
          reference = await run(job.reference, referencePolicy);
        }
        if (!cancelled) publish("finished", result, reference);
      } catch (error: unknown) {
        if (!cancelled) publish("finished", { ok: false, code: "refused", reason: error instanceof Error ? error.message : "owned kernel preparation failed", disposed: true }, null);
      }
    };
    void execute().finally(finish);
    return () => {
      live = false;
      cancelled = true;
      if (handle !== null) void handle.dispose();
    };
  }, [job]);

  const current = state.job === job ? state : { phase: job === null ? "idle" as const : "running" as const, result: null, reference: null };
  return { phase: current.phase, result: current.result, reference: current.reference, cancel: () => cancelCurrent.current() };
}
