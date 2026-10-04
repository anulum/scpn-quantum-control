// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — complete local experiment journey

import { useEffect, useRef, useState } from "react";
import { fetchKuramoto } from "../../panel/kuramoto";
import type { KuramotoKernel } from "../../panel/kuramoto";
import { SimulationDataTable } from "../../panel/SimulationDataTable";
import type { RawCodec } from "../../shared/contracts";
import { downloadWorkspaceArchive } from "../workspace/WorkspacePanel";
import type { WorkspaceController } from "../workspace/useWorkspace";
import { appendExperimentAttempt, createLocalSample, readLocalExperiment } from "./experimentArchive";
import { prepareExperimentPlan, prepareSavedReplay } from "./experimentPlan";
import type { ExperimentRunController } from "./useExperimentRun";
import { ExperimentRefusal } from "./kuramotoArtifacts";

/** Original workspace and worker owners supplied by the continuously mounted workbench. */
export interface ExperimentRunnerProps {
  /** The same original archive/controller as the Workspace editor. */ readonly workspace: WorkspaceController;
  /** Trusted source verifiers from the product and host, never imported plugins. */ readonly rawCodecs: ReadonlyMap<string, RawCodec>;
  /** Source-bound original worker state and observed disposal controls. */ readonly run: ExperimentRunController;
  /** Original context-preserving navigation to the single linked parameter editor. */ readonly workspaceHref?: string;
  /** Original loader seam; production fetches only its shipped module-relative WASM. */ readonly loadKernel?: () => Promise<KuramotoKernel>;
}

/** Present sample, original edit/admission, real execution, immutable save and independent replay. */
export function ExperimentRunner({ workspace, rawCodecs, run, workspaceHref = "#/workspace", loadKernel = fetchKuramoto }: ExperimentRunnerProps) {
  const [pending, setPending] = useState(false), [message, setMessage] = useState("");
  const [memoryBudget, setMemoryBudget] = useState("");
  const live = useRef(true), generation = useRef(0);
  const source = useRef(workspace.draft); source.current = workspace.draft;
  const kernel = useRef<KuramotoKernel | null>(null);
  const cancellation = useRef<AbortController | null>(null);
  useEffect(() => {
    live.current = true;
    return () => { live.current = false; ++generation.current; cancellation.current?.abort(); };
  }, []);
  const current = (epoch: number, prior: string) => {
    if (!live.current || generation.current !== epoch || source.current !== prior) throw new ExperimentRefusal("source changed while preparing the operation; current draft and saved archive retained");
  };
  const originalKernel = async () => { kernel.current ??= await loadKernel(); return kernel.current; };
  const operation = async (action: (epoch: number, prior: string) => Promise<string>) => {
    const epoch = ++generation.current, prior = workspace.draft;
    cancellation.current?.abort(); cancellation.current = new AbortController();
    setPending(true); setMessage("");
    try {
      const result = await action(epoch, prior);
      if (live.current && generation.current === epoch) setMessage(result);
    } catch (cause: unknown) {
      if (live.current && generation.current === epoch) setMessage(cause instanceof ExperimentRefusal ? cause.message : "Local experiment operation refused; current draft and saved archive retained");
    } finally { if (live.current && generation.current === epoch) setPending(false); }
  };
  const preview = workspace.preview?.json === workspace.draft ? workspace.preview : workspace.saved?.preview.json === workspace.draft ? workspace.saved.preview : null;
  const plan = run.plan;
  const planControlsCurrent = plan !== null && memoryBudget === (plan.requestedMemoryBytes === null ? "" : String(plan.requestedMemoryBytes));
  const running = run.status === "running" || run.status === "cancelling";
  const disabled = pending || workspace.busy;
  const attemptArchive = async (epoch: number, prior: string) => {
    // Both controls require canSave synchronously; the original archive owner revalidates lifecycle identity.
    const archive = await appendExperimentAttempt(plan!, run.runId, run.attemptId, run.events, run.outcome!, rawCodecs);
    current(epoch, prior);
    return archive;
  };
  return <section aria-label="Local experiment" className="qsp-workspace">
    <h3>Local experiment</h3>
    <p>Open the original classical Kuramoto sample, edit its typed parameters in Workspace, validate the complete archive, inspect a numerical plan and explicitly run the shipped Rust/WASM worker.</p>
    <p>Phases use radians and time is an unscaled model-time unit. This classical model and its numerical result provide no quantum-spin, physical-device or provider execution claim.</p>
    <div className="qsp-workspace-actions">
      <button type="button" disabled={disabled || running} onClick={() => { void operation(async (epoch, prior) => {
        const archive = await createLocalSample(await originalKernel(), rawCodecs, navigator.userAgent);
        current(epoch, prior); workspace.edit(archive.json);
        return "Original sample opened as an unsaved project draft. Validate before editing or planning.";
      }); }}>Open Kuramoto sample</button>
      <button type="button" disabled={disabled || workspace.draft === ""} onClick={() => { void workspace.inspect(); }}>Validate experiment archive</button>
      <a href={workspaceHref}>Edit source parameters in Workspace</a>
      <button type="button" disabled={disabled || running || preview === null} onClick={() => { void operation(async (epoch, prior) => {
        const captured = await readLocalExperiment(prior, rawCodecs);
        const prepared = await prepareExperimentPlan(captured, await originalKernel(), memoryBudget === "" ? {} : { memoryBudget });
        current(epoch, prior); await run.prepare(prepared);
        return prepared.admission.allowed ? "Numerical plan ready. Inspect its source, environment and complete budget before Run experiment." : `Numerical plan refused: ${prepared.admission.blockers.join(", ")}. No worker started.`;
      }); }}>Prepare numerical plan</button>
      <button type="button" disabled={disabled || running || preview === null} onClick={() => { void operation(async (epoch, prior) => {
        const replay = await prepareSavedReplay(prior, await originalKernel(), rawCodecs);
        current(epoch, prior); await run.prepare(replay.plan, replay.expected);
        setMemoryBudget(replay.plan.requestedMemoryBytes === null ? "" : String(replay.plan.requestedMemoryBytes));
        return "Original saved numerical plan and output admitted. Run experiment explicitly to compare a genuine new result.";
      }); }}>Prepare saved replay</button>
    </div>
    <label>Run numeric byte budget <input inputMode="numeric" value={memoryBudget} disabled={disabled || running} placeholder="Use recorded source policy" onChange={event => setMemoryBudget(event.target.value)} /></label>
    <p>Supply a canonical nonnegative byte count below the original policy ceiling. Leaving it empty retains the source policy. No host capacity or elapsed-time guarantee is inferred.</p>
    <p role="status" aria-label="Experiment operation">{message || workspace.message}</p>
    <p role="status" aria-label="Experiment lifecycle">{run.status}: {run.reason}</p>
    {plan !== null && !planControlsCurrent && <p role="status">Run budget changed. Prepare a new numerical plan before execution.</p>}
    {run.disposalBlocked && <p role="alert">Worker disposal is unconfirmed. Further execution and attempt saving remain unavailable.</p>}
    {plan && <dl aria-label="Numerical plan">
      <dt>Revision</dt><dd>{plan.revisionHash}</dd>
      <dt>Plan fingerprint</dt><dd>{plan.planHash}</dd>
      <dt>Kernel build</dt><dd>{plan.buildFingerprint}</dd>
      <dt>Backend</dt><dd>{plan.admission.estimate.backend}</dd>
      <dt>Method</dt><dd>{plan.request.mode} · Rust fixed-step RK4</dd>
      <dt>Precision</dt><dd>float64</dd>
      <dt>Source shape</dt><dd>{plan.request.omega.length} oscillators · {plan.request.steps} steps · dt {plan.request.dt} model-time</dd>
      <dt>Actual native bounds</dt><dd>{plan.bounds.maxOscillators} oscillators · {plan.bounds.maxSteps} steps</dd>
      <dt>Requested run byte ceiling</dt><dd>{plan.requestedMemoryBytes === null ? "Recorded source policy" : String(plan.requestedMemoryBytes)}</dd>
      <dt>Effective declared byte ceiling</dt><dd>{plan.policy.memoryBytes === null ? "Unknown" : String(plan.policy.memoryBytes)}</dd>
      <dt>Declared numeric and worker bytes</dt><dd>{plan.admission.bytesRequired === null ? "Unknown" : String(plan.admission.bytesRequired)}</dd>
      <dt>Declared work units</dt><dd>{String(plan.admission.estimate.workUnits)}</dd>
      <dt>Policy origin</dt><dd>{plan.policy.source}</dd>
      <dt>Operational disposal timeout</dt><dd>{plan.deadlineMs} ms; no compute-duration guarantee</dd>
      <dt>Numerical seed</dt><dd>Not applicable to the deterministic original request</dd>
      <dt>Admission</dt><dd>{plan.admission.allowed ? "Allowed by declared policy; native worker admission still required" : plan.admission.blockers.join(", ")}</dd>
      <dt>Qualification limit</dt><dd>{plan.admission.claimBoundary}</dd>
    </dl>}
    <div className="qsp-workspace-actions">
      <button type="button" disabled={disabled || !run.canRun || !planControlsCurrent} onClick={() => { void operation(async () => { await run.run(); return "Original attempt finished; inspect its lifecycle and result before saving."; }); }}>Run experiment</button>
      <button type="button" disabled={!running} onClick={() => { void run.cancel(); }}>Cancel experiment</button>
      <button type="button" disabled={disabled || !run.canSave || !workspace.storageAvailable} onClick={() => { void operation(async (epoch, prior) => {
        const archive = await attemptArchive(epoch, prior);
        await workspace.saveRevision(archive, prior, cancellation.current!.signal);
        await run.retainSavedAttempt(archive, prior, rawCodecs);
        return "Exact experiment attempt committed to the original browser archive. Export a portable backup.";
      }); }}>Save experiment attempt</button>
      <button type="button" disabled={disabled || !run.canSave} onClick={() => { void operation(async (epoch, prior) => {
        downloadWorkspaceArchive(await attemptArchive(epoch, prior)); return "Exact current-source attempt exported as a portable archive; browser saved state unchanged.";
      }); }}>Export experiment attempt</button>
      <button type="button" disabled={workspace.saved === null} onClick={() => downloadWorkspaceArchive(workspace.saved!.preview)}>Export saved experiment</button>
    </div>
    {run.events.length > 0 && <ol aria-label="Original attempt diagnostics">{run.events.map(event => <li key={event.sequence}>{event.sequence} · {event.kind} · revision {String(event.payload["revision_hash"])} · plan {String(event.payload["plan_hash"])}{typeof event.payload["reason"] === "string" ? ` · ${event.payload["reason"]}` : ""}</li>)}</ol>}
    {run.outcome?.ok && <div aria-label="Current experiment result">
      <p role="status">Experiment succeeded{run.replayVerified ? " · Original float64 replay verified" : ""}</p>
      <p>Run {run.runId} · revision {run.outcome.revisionHash} · plan {run.outcome.planHash} · worker disposed</p>
      <SimulationDataTable label="Experiment order parameter" orderParameter={run.outcome.run.orderParameter} />
      <dl aria-label="Original final phases">{Array.from(run.outcome.run.thetaFinal, (value, index) => <div key={index}><dt>θ{index} (rad)</dt><dd>{Object.is(value, -0) ? "-0" : String(value)}</dd></div>)}</dl>
    </div>}
    {!workspace.storageAvailable && <p>Browser persistence is unavailable. A disposed attempt can still be exported as an independent portable archive; no browser-save success is claimed.</p>}
  </section>;
}

export default ExperimentRunner;
