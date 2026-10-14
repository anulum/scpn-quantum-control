// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound registered job lifecycle

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ModuleDocument,
  ModuleJobEstimate,
  ModuleRegistry,
  ModuleRequirements,
} from "../contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../contracts/workbench";
import { admitModuleJob } from "./moduleJobPolicy";

/** Compiled jobs executed through the existing source, policy and disposal owners. */
export interface ModuleJobDockProps {
  /** Actual registered modules. */ readonly registry: ModuleRegistry<
    WorkbenchServices,
    WorkspaceViewServices
  >;
  /** Existing runtime services. */ readonly services: WorkbenchServices;
  /** Guarded installation declarations. */ readonly declared: ModuleRequirements;
  /** Exact selected admitted source, or no source. */ readonly document: ModuleDocument | null;
  /** Actual selected stage; job commands cannot activate an absent runtime implicitly. */ readonly activeStage:
    | string
    | null;
  /** Explicit command-search request; the sequence is local UI state, not a run identity. */ readonly request: {
    /** Monotonic local command request sequence. */
    readonly sequence: number;
    /** Actual registered job identity. */
    readonly job: string;
  } | null;
}

interface JobState {
  readonly job: string | null;
  readonly status: "idle" | "estimating" | "running" | "cancelled" | "settled" | "refused";
  readonly estimate: ModuleJobEstimate | null;
  readonly reason: string;
}

/** Estimate, admit and execute a registered job without accepting a module-supplied permission flag. */
export function ModuleJobDock(props: ModuleJobDockProps) {
  const current = useRef(props);
  current.current = props;
  const live = useRef(true);
  const owned = useRef<{
    readonly cancellation: AbortController;
    readonly registry: ModuleJobDockProps["registry"];
    readonly stage: string | null;
    readonly source: string | null;
  } | null>(null);
  const [state, setState] = useState<JobState>({
    job: null,
    status: "idle",
    estimate: null,
    reason: "No job requested",
  });
  const execute = useCallback(async (id: string) => {
    if (owned.current !== null) return;
    const captured = current.current;
    const job = captured.registry.jobs.find((entry) => entry.id === id);
    const source = captured.document;
    const cancellation = new AbortController();
    owned.current = {
      cancellation,
      registry: captured.registry,
      stage: captured.activeStage,
      source: source?.content ?? null,
    };
    try {
      setState({
        job: id,
        status: "estimating",
        estimate: null,
        reason: "Reading original job declarations",
      });
      if (job === undefined || source === null)
        throw new Error("Select an admitted source and registered job first");
      const owner = captured.registry.contributionOwner("jobs", id);
      if (owner === undefined) throw new Error("Registered job owner unavailable");
      const missing = captured.registry
        .missing(captured.declared)
        .filter((requirement) => requirement.module === owner.id);
      if (missing.length > 0)
        throw new Error(
          `Job unavailable: ${missing.map(({ kind, name }) => `${kind} ${name}`).join(", ")}`,
        );
      if (captured.activeStage !== job.stage)
        throw new Error(`Open ${job.stage} before requesting this job`);
      const estimate = await job.estimate(captured.services, source);
      const latest = current.current;
      if (
        cancellation.signal.aborted ||
        latest.document?.content !== source.content ||
        latest.activeStage !== job.stage ||
        latest.registry.jobs.find((entry) => entry.id === id) !== job
      )
        throw new Error("Job cancelled or source/module changed during preparation");
      await admitModuleJob(latest.services, job, source, estimate, cancellation.signal);
      const admitted = current.current;
      if (
        !live.current ||
        cancellation.signal.aborted ||
        admitted.document?.content !== source.content ||
        admitted.services.workspace.draft !== source.content ||
        admitted.activeStage !== job.stage ||
        admitted.registry.jobs.find((entry) => entry.id === id) !== job
      )
        throw new Error("Job cancelled or current source/module changed during admission");
      setState({
        job: id,
        status: "running",
        estimate,
        reason: "Original runtime owns execution and disposal",
      });
      await job.run(current.current.services, source, cancellation.signal);
      if (live.current)
        setState({
          job: id,
          status: cancellation.signal.aborted ? "cancelled" : "settled",
          estimate,
          reason: "Request settled; inspect the original runtime's outcome and diagnostics",
        });
    } catch (cause: unknown) {
      if (live.current)
        setState((previous) => ({
          ...previous,
          status: cancellation.signal.aborted ? "cancelled" : "refused",
          reason:
            cause instanceof Error
              ? cause.message
              : "Job request refused; original source retained",
        }));
    } finally {
      owned.current = null;
    }
  }, []);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      owned.current?.cancellation.abort();
    };
  }, []);
  // Reconcile the owned request with the latest host state after every committed render.
  useEffect(() => {
    const latest = current.current;
    const captured = owned.current;
    if (captured === null) return;
    // An admission effect may flush after a request has captured the same source.
    const currentDraft = latest.services.workspace.draft;
    const currentSource = latest.document?.content ?? null;
    const unchanged = currentDraft === captured.source && currentSource === captured.source;
    if (
      (latest.document !== null && latest.document.schema !== "quantum_workspace_archive.v1") ||
      (!unchanged && !latest.services.workflow.ownsSource(currentDraft))
    )
      captured.cancellation.abort();
  });
  useEffect(() => {
    const captured = owned.current;
    if (
      captured !== null &&
      (captured.stage !== props.activeStage || captured.registry !== props.registry)
    )
      captured.cancellation.abort();
  }, [props.activeStage, props.registry]);
  useEffect(() => {
    if (props.request !== null) void execute(props.request.job);
  }, [props.request, execute]);
  const busy = state.status === "estimating" || state.status === "running";
  const missing = props.registry.missing(props.declared);
  return (
    <section aria-label="Registered job dock">
      <h3>Local jobs</h3>
      <ul>
        {props.registry.jobs.map((job) => {
          const owner = props.registry.contributionOwner("jobs", job.id);
          if (owner === undefined)
            return <li key={job.id}>{job.label}: registered owner unavailable</li>;
          const lacking = missing.filter((requirement) => requirement.module === owner.id);
          return (
            <li key={job.id}>
              <button
                type="button"
                disabled={
                  busy ||
                  props.document === null ||
                  props.activeStage !== job.stage ||
                  lacking.length > 0 ||
                  job.runner !== "worker"
                }
                onClick={() => {
                  void execute(job.id);
                }}
              >
                {job.label}
              </button>
              {job.runner !== "worker" && <span> · Local service unavailable</span>}
              {lacking.length > 0 && (
                <span> · {lacking.map(({ kind, name }) => `${kind} ${name}`).join(", ")}</span>
              )}
            </li>
          );
        })}
      </ul>
      <p role="status" aria-label="Registered job lifecycle">
        {state.status}: {state.reason}
      </p>
      {state.estimate?.kind === "declared-resources" && (
        <dl aria-label="Registered job estimate">
          <dt>Backend</dt>
          <dd>{state.estimate.plan.backend}</dd>
          <dt>Method</dt>
          <dd>{state.estimate.plan.method}</dd>
          <dt>Declared work units</dt>
          <dd>{String(state.estimate.plan.workUnits)}</dd>
          <dt>Buffers</dt>
          <dd>{state.estimate.plan.buffers.length}</dd>
        </dl>
      )}
      {state.estimate?.kind === "workflow-runtime" && (
        <dl aria-label="Registered workflow estimate">
          <dt>Graph</dt>
          <dd>{state.estimate.workflowDigest}</dd>
          <dt>Cells</dt>
          <dd>{state.estimate.cells}</dd>
          <dt>Stages per cell</dt>
          <dd>{state.estimate.stages}</dd>
          <dt>Whole-graph memory</dt>
          <dd>Not estimated; the original runtime admits each numerical stage before allocation</dd>
        </dl>
      )}
      <button type="button" disabled={!busy} onClick={() => owned.current?.cancellation.abort()}>
        Cancel registered job
      </button>
    </section>
  );
}
