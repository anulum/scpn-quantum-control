// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original classical WASM workflow execution

import { createOwnedKuramotoRun } from "../../panel/kuramoto";
import type { KuramotoKernel, OwnedKuramotoHandle } from "../../panel/kuramoto";
import { canonicalDigest, documentDigest } from "../../shared/contracts";
import type { RawCodec } from "../../shared/contracts";
import type { RawIdentity } from "../../shared/contracts/graph";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { browserResourcePolicy } from "../../shared/resources/kuramotoResources";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import type { KernelWorkerEvent, OwnedKernelOutcome } from "../../workers/kernelProtocol";
import { appendExperimentAttempt, readLocalExperiment } from "../experiments/experimentArchive";
import { prepareExperimentPlan, prepareSavedReplay } from "../experiments/experimentPlan";
import type { ExperimentPlan, ExperimentPlanOptions } from "../experiments/experimentPlan";
import {
  artifactContent,
  artifactBytesDigest,
  decodeFloat64,
  encodeFloat64,
  ExperimentRefusal,
  readExperimentArtifact,
  localExperimentCodecs,
} from "../experiments/kuramotoArtifacts";
import {
  createParameterDraft,
  parameterDraftReducer,
  validateParameterSnapshot,
} from "../parameters/parameterDraft";
import type { ParameterDraftSource, ParameterValue } from "../parameters/parameterDraft";
import {
  appendParameterRevision,
  createParameterRevision,
  parameterSourceFromArchive,
} from "../parameters/parameterRevision";
import { archiveWorkflow, readWorkflowArchive, selectWorkflowRevision } from "./workflowArchive";
import {
  createWorkflowJournal,
  parseWorkflowJournal,
  workflowJournalDocument,
} from "./workflowJournal";
import type { WorkflowAttempt, WorkflowJournal } from "./workflowJournal";
import {
  parseWorkflow,
  topologicalOrder,
  validatePortValue,
  WorkflowRefusal,
  workflowDocument,
} from "./workflowModel";
import type { WorkflowDefinition, WorkflowStage } from "./workflowModel";
import { buildWorkflowCells } from "./workflowSweep";
import type { WorkflowCell } from "./workflowSweep";

/** Actual original archive and checkpoint after the last observed save. */
export interface WorkflowExecutionResult {
  /** Last complete original workspace archive preview. */ readonly archive: WorkspaceArchivePreview;
  /** Explicit original complete/partial/cancelled attempt history. */ readonly journal: WorkflowJournal;
  /** Whether every allocated original worker confirmed disposal. */ readonly disposalConfirmed: boolean;
}
/** Actual settled stage traversal for one original run or source-checked restart. */
export interface WorkflowStageProgress {
  /** Exact original graph identity. */ readonly workflowDigest: string;
  /** Original coordinate identity, without a new execution claim. */ readonly cellId: string;
  /** Original stage just settled or revalidated. */ readonly stageId: string;
  /** Stages settled in this traversal, including failed and blocked stages. */ readonly settledStages: number;
  /** All original cells times their full stage count. */ readonly totalStages: number;
  /** True only after the original cached output and current plan match. */ readonly reused: boolean;
}
/** Trusted runtime and original workspace transaction owner; imported graphs never supply callbacks. */
export interface WorkflowExecutionOptions {
  /** Exact original admitted workspace text at activation. */ readonly sourceJson: string;
  /** Explicit immutable original baseline; own subsequent saves do not rebind it. */ readonly baseRevision: string;
  /** Admitted local/classical graph. */ readonly definition: WorkflowDefinition;
  /** Actual original shipped WASM loader result. */ readonly kernel: KuramotoKernel;
  /** Original trusted source-verifier registry. */ readonly rawCodecs: ReadonlyMap<
    string,
    RawCodec
  >;
  /** Original exact-prior-text workspace transaction; throw on stale or failed save. */ readonly save: (
    archive: WorkspaceArchivePreview,
    priorJson: string,
  ) => Promise<void>;
  /** Cooperative cancellation; source-ownership cancellation belongs to the save callback. */ readonly signal: AbortSignal;
  /** Existing tighter native numeric ceiling and bounded operational deadline. */ readonly planOptions?: ExperimentPlanOptions;
  /** Observe admitted persisted progress, never a manufactured execution result. */ readonly onCheckpoint?: (
    result: WorkflowExecutionResult,
  ) => void;
  /** Observe actual traversal without manufacturing a checkpoint or charging an evaluation. */ readonly onProgress?: (
    progress: WorkflowStageProgress,
  ) => void;
  /** Trusted actual worker ownership observer for source/route/unmount cancellation. */ readonly onWorker?: (
    handle: OwnedKuramotoHandle | null,
  ) => void;
  /** Original real transport seam for native-thread qualification. */ readonly workerFactory?: () => KernelWorkerPort;
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new WorkflowRefusal("original workflow object required");
  return value as Record<string, unknown>;
}
function select(value: unknown, path: readonly string[]): unknown {
  for (const key of path) {
    const current = object(value);
    if (!Object.hasOwn(current, key))
      throw new WorkflowRefusal("original workflow output port is absent");
    value = current[key];
  }
  return value;
}
function scalarValues(value: unknown, shape: readonly bigint[], dtype: string): readonly string[] {
  if (shape.length > 0) {
    if (!Array.isArray(value) || BigInt(value.length) !== shape[0])
      throw new WorkflowRefusal("workflow override differs from original parameter shape");
    return value.flatMap((item) => scalarValues(item, shape.slice(1), dtype));
  }
  if (dtype === "float64" && typeof value === "number") return [encodeFloat64(value)];
  if ((dtype === "int64" || dtype === "uint64") && typeof value === "bigint")
    return [value.toString()];
  throw new WorkflowRefusal("workflow override differs from original parameter dtype");
}
function parameterCandidate(
  source: ParameterDraftSource,
  values: Readonly<Record<string, unknown>>,
) {
  let draft = createParameterDraft(source);
  const parameters: Readonly<Record<string, unknown>> =
    values["parameters"] === undefined ? draft.snapshot.parameters : object(values["parameters"]);
  const updated = { ...parameters };
  for (const [key, value] of Object.entries(values)) {
    if (key === "parameters") continue;
    const original = draft.snapshot.parameters[key];
    if (original === undefined)
      throw new WorkflowRefusal("workflow parameter is absent from the original specification");
    const typed: ParameterValue = {
      dtype: original.dtype,
      shape: original.shape,
      values: scalarValues(value, original.shape, original.dtype),
    };
    updated[key] = typed;
  }
  draft = parameterDraftReducer(draft, {
    type: "replace",
    parameters: updated,
    units: source.units,
  });
  if (draft.refusal !== null)
    throw new WorkflowRefusal("original parameter specification refused the workflow coordinate");
  return validateParameterSnapshot(source, draft.snapshot);
}
function parameters(
  stage: WorkflowStage,
  cell: WorkflowCell,
  completed: ReadonlyMap<string, WorkflowAttempt>,
  stages: ReadonlyMap<string, WorkflowStage>,
): Record<string, unknown> {
  const result = { ...stage.parameters, ...cell.overrides[stage.id] };
  for (const input of stage.inputs) {
    const parent = completed.get(input.source_stage) as WorkflowAttempt;
    const source = stages.get(input.source_stage) as WorkflowStage;
    const port = source.outputs.find(
      (output) => output.name === input.source_port,
    ) as WorkflowStage["outputs"][number];
    result[input.parameter] = validatePortValue(port.type, select(parent.output, port.path));
  }
  return result;
}
async function previewOutput(
  plan: ExperimentPlan,
  source: ParameterDraftSource,
): Promise<Record<string, unknown>> {
  return {
    producer_identity:
      "studio-web/src/features/experiments/experimentPlan.ts#prepareExperimentPlan",
    preview_only: true,
    revision_hash: plan.revisionHash,
    plan_hash: plan.planHash,
    build_fingerprint: plan.buildFingerprint,
    parameters: source.revision.body["parameters"],
    original_plan: await readExperimentArtifact("plan", artifactContent(plan.planMember)),
  };
}
async function savedRun(json: string, reference: unknown, options: WorkflowExecutionOptions) {
  const ref = object(reference);
  if (
    Object.keys(ref).length !== 4 ||
    ["record_ref", "revision_hash", "plan_hash", "build_fingerprint"].some(
      (key) => typeof ref[key] !== "string",
    )
  )
    throw new WorkflowRefusal("complete original workflow run reference required");
  const archive = await selectWorkflowRevision(
    json,
    ref["revision_hash"] as string,
    options.rawCodecs,
    ref["record_ref"] as string,
  );
  const admitted = await readWorkflowArchive(archive.json, options.rawCodecs);
  const record = admitted.source.documents[ref["record_ref"] as string];
  if (
    record === undefined ||
    record.body["revision_hash"] !== ref["revision_hash"] ||
    record.body["plan_hash"] !== ref["plan_hash"]
  )
    throw new WorkflowRefusal("original indexed run belongs to another workflow source");
  const replay = await prepareSavedReplay(archive.json, options.kernel, options.rawCodecs);
  if (
    replay.plan.planHash !== ref["plan_hash"] ||
    replay.plan.buildFingerprint !== ref["build_fingerprint"]
  )
    throw new WorkflowRefusal("original saved run fingerprint differs from current WASM and plan");
  const outcome: OwnedKernelOutcome = {
    ok: true,
    disposed: true,
    runId: record.body["run_id"] as string,
    revisionHash: replay.plan.revisionHash,
    planHash: replay.plan.planHash,
    buildFingerprint: replay.plan.buildFingerprint,
    run: {
      orderParameter: Float64Array.from(replay.expected.orderParameter, decodeFloat64),
      thetaFinal: Float64Array.from(replay.expected.thetaFinal, decodeFloat64),
    },
  };
  return { plan: replay.plan, outcome };
}
function archivedEvents(events: readonly KernelWorkerEvent[]): readonly unknown[] {
  return events.map((event) => ({
    ...event,
    version: 1n,
    sequence: BigInt(event.sequence),
    payload: Object.fromEntries(
      Object.entries(event.payload).map(([key, value]) => [
        key,
        ArrayBuffer.isView(value) &&
        Object.prototype.toString.call(value) === "[object Float64Array]"
          ? {
              dtype: "float64",
              shape: [BigInt((value as Float64Array).length)],
              values: Array.from(value as Float64Array, encodeFloat64),
            }
          : value,
      ]),
    ),
  }));
}

function originalCodecs(
  codecs: ReadonlyMap<string, RawCodec>,
  capacity: { limit: number },
): ReadonlyMap<string, RawCodec> {
  const scoped = new Map<string, RawCodec>();
  for (const [schema, verifier] of codecs) {
    if (localExperimentCodecs.get(schema) !== verifier) {
      scoped.set(schema, verifier);
      continue;
    }
    const verified = new Map<
      string,
      { readonly bytes: Uint8Array; readonly identity: RawIdentity }
    >();
    scoped.set(schema, async (content) => {
      const hash = await artifactBytesDigest(content);
      const known = verified.get(hash);
      if (
        known !== undefined &&
        known.bytes.length === content.length &&
        known.bytes.every((byte, index) => byte === content[index])
      )
        return known.identity;
      const identity = await verifier(content);
      if (verified.size < capacity.limit) verified.set(hash, { bytes: content.slice(), identity });
      return identity;
    });
  }
  return scoped;
}

/** Run original local validate/simulate/analyse stages, checkpointing the actual original producer history. */
export async function runLocalWorkflow(
  options: WorkflowExecutionOptions,
): Promise<WorkflowExecutionResult> {
  const definition = parseWorkflow(workflowDocument(options.definition));
  const capacity = { limit: Math.min(1000, Number(definition.sweep.evaluation_budget)) };
  options = {
    ...options,
    rawCodecs: originalCodecs(options.rawCodecs, capacity),
  };
  if (
    definition.stages.some(
      (stage) =>
        stage.adapter !== "local-kuramoto" || stage.backend !== "shipped-kuramoto-wasm-float64",
    )
  )
    throw new WorkflowRefusal(
      "browser execution requires the original classical Kuramoto WASM adapter; executive graphs run through the local CLI",
    );
  const current = await readWorkflowArchive(options.sourceJson, options.rawCodecs);
  capacity.limit = Math.min(1000, capacity.limit + current.source.preview.rawHashes.length);
  const baseline = await readLocalExperiment(
    options.sourceJson,
    options.rawCodecs,
    options.baseRevision,
  );
  const baseParameters = await parameterSourceFromArchive(
    options.sourceJson,
    options.rawCodecs,
    options.baseRevision,
  );
  await prepareExperimentPlan(baseline, options.kernel, options.planOptions);
  const sourceFingerprint = await canonicalDigest("studio.workflow-local-source.v1", {
    revision_hash: baseline.revisionHash,
    units: baseline.archive.parameterUnits,
    specs: baseline.archive.parameterSpecs,
    environment: baseline.environmentMember.sha256,
    policy: baseline.policy,
  });
  const runtimeFingerprint = await canonicalDigest("studio.workflow-local-runtime.v1", {
    kernel_sha256: baseline.kernelHash,
    bounds: options.kernel.bounds,
    product_policy: browserResourcePolicy(options.kernel.bounds),
    adapter: "original classical Kuramoto WASM",
  });
  const digest = await canonicalDigest(
    "studio.workflow-definition.v1",
    workflowDocument(definition),
  );
  const stored = current.workflows.find((item) => item.hash === digest);
  if (stored !== undefined && stored.base_revision_hash !== options.baseRevision)
    throw new WorkflowRefusal("original workflow baseline changed");
  let journal =
    stored?.journal ??
    (await createWorkflowJournal(definition, sourceFingerprint, runtimeFingerprint));
  if (
    journal.source_fingerprint !== sourceFingerprint ||
    journal.runtime_fingerprint !== runtimeFingerprint
  )
    throw new WorkflowRefusal(
      "original workflow source or runtime changed; prior journal remains retained",
    );
  let archive = current.source.preview;
  let persistedJson = options.sourceJson;
  let disposalConfirmed = true;
  let entries = [...journal.entries];
  const checkpoint = async (state: WorkflowJournal["state"]): Promise<void> => {
    const document = workflowJournalDocument(journal);
    const body = object(document["body"]);
    body["entries"] = entries;
    body["state"] = state;
    body["evaluations"] = BigInt(entries.filter((entry) => entry.evaluated).length);
    const next = await parseWorkflowJournal(document, definition);
    const candidate = await archiveWorkflow(
      archive.json,
      options.baseRevision,
      definition,
      next,
      options.rawCodecs,
    );
    await options.save(candidate, persistedJson);
    archive = candidate;
    persistedJson = candidate.json;
    journal = next;
    entries = [...next.entries];
    options.onCheckpoint?.({ archive, journal, disposalConfirmed });
  };
  let recovered = false;
  entries = entries.map((entry) => {
    if (entry.status !== "running") return entry;
    recovered = true;
    return {
      ...entry,
      status: "interrupted",
      reason: "Original worker attempt has no terminal checkpoint",
    };
  });
  if (recovered) await checkpoint("partial");
  const stages = new Map(definition.stages.map((stage) => [stage.id, stage]));
  const latest = new Map(entries.map((entry) => [`${entry.cell_id}:${entry.stage_id}`, entry]));
  let finished = true;
  const cells = await buildWorkflowCells(definition);
  let settledStages = 0;
  const progress = (cellId: string, stageId: string, reused: boolean): void => {
    ++settledStages;
    options.onProgress?.({
      workflowDigest: digest,
      cellId,
      stageId,
      settledStages,
      totalStages: cells.length * definition.stages.length,
      reused,
    });
  };
  for (const cell of cells) {
    const completed = new Map<string, WorkflowAttempt>();
    for (const sid of topologicalOrder(definition)) {
      // Saved-history validation needs an event-loop turn for rendering and user cancellation.
      if (latest.has(`${cell.id}:${sid}`))
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      if (options.signal.aborted) {
        await checkpoint("cancelled");
        return { archive, journal, disposalConfirmed };
      }
      const stage = stages.get(sid) as WorkflowStage;
      const parents = [
        ...new Set([...stage.depends_on, ...stage.inputs.map((input) => input.source_stage)]),
      ].sort();
      const dependencies = Object.fromEntries(
        parents.map((parent) => [parent, completed.get(parent)?.output_digest ?? null]),
      );
      const base = {
        cell_id: cell.id,
        stage_id: sid,
        source: sourceFingerprint,
        runtime: runtimeFingerprint,
        dependencies,
      };
      if (parents.some((parent) => !completed.has(parent))) {
        const fingerprint = await canonicalDigest("studio.workflow-stage.v1", base);
        const prior = latest.get(`${cell.id}:${sid}`);
        if (prior?.status !== "blocked" || prior.fingerprint !== fingerprint) {
          entries.push({
            cell_id: cell.id,
            stage_id: sid,
            fingerprint,
            dependencies,
            status: "blocked",
            output: null,
            output_digest: null,
            reason: "Original parent did not complete",
            evaluated: false,
          });
          await checkpoint("partial");
        }
        finished = false;
        progress(cell.id, sid, false);
        continue;
      }
      const values = parameters(stage, cell, completed, stages);
      let plan: ExperimentPlan | null = null,
        selectedParameters: ParameterDraftSource | null = null;
      let refused = false;
      try {
        if (stage.verb !== "analyse") {
          const candidate = parameterCandidate(baseParameters, values);
          let revisionHash = options.baseRevision;
          if (
            (await canonicalDigest("studio.workflow-parameters.v1", candidate.parameters)) !==
            (await canonicalDigest(
              "studio.workflow-parameters.v1",
              baseParameters.revision.body["parameters"],
            ))
          ) {
            const revision = await createParameterRevision(baseParameters, candidate);
            revisionHash = await documentDigest(revision);
            if (
              (await readWorkflowArchive(archive.json, options.rawCodecs)).source.revisions[
                revisionHash
              ] === undefined
            ) {
              const child = await appendParameterRevision(
                archive.json,
                baseParameters,
                candidate,
                options.rawCodecs,
                new Date().toISOString(),
              );
              archive = child.archive;
            }
          }
          const selected = await selectWorkflowRevision(
            archive.json,
            revisionHash,
            options.rawCodecs,
          );
          archive = selected;
          selectedParameters = await parameterSourceFromArchive(
            archive.json,
            options.rawCodecs,
            revisionHash,
          );
          const source = await readLocalExperiment(archive.json, options.rawCodecs, revisionHash);
          plan = await prepareExperimentPlan(source, options.kernel, options.planOptions);
        } else {
          if (Object.keys(values).length !== 1 || values["run_ref"] === undefined)
            throw new WorkflowRefusal("local analyse requires one original run_ref input");
          plan = (await savedRun(archive.json, values["run_ref"], options)).plan;
        }
      } catch (cause: unknown) {
        if (
          !(
            cause instanceof WorkflowRefusal ||
            cause instanceof ExperimentRefusal ||
            cause instanceof Error
          )
        )
          throw cause;
        refused = true;
      }
      const fingerprint = await canonicalDigest("studio.workflow-stage.v1", {
        ...base,
        parameters: values,
        plan_hash: plan?.planHash ?? null,
      });
      const prior = latest.get(`${cell.id}:${sid}`);
      if (prior?.status === "complete" && prior.fingerprint === fingerprint) {
        if (refused || plan === null)
          throw new WorkflowRefusal("original completed stage no longer has a current plan");
        if (stage.verb === "validate") {
          const expected = await previewOutput(plan, selectedParameters as ParameterDraftSource);
          if (
            (await canonicalDigest("studio.workflow-output.v1", expected)) !== prior.output_digest
          )
            throw new WorkflowRefusal(
              "original cached preview differs from current source and plan",
            );
        } else {
          const ref =
            stage.verb === "simulate" ? object(prior.output)["run_ref"] : values["run_ref"];
          const original = await savedRun(archive.json, ref, options);
          if (original.plan.planHash !== plan.planHash)
            throw new WorkflowRefusal("original cached run plan differs");
          if (stage.verb === "analyse") {
            const { inspectKuramotoResult } = await import("../results/resultSources");
            const expected = {
              producer_identity:
                "studio-web/src/features/results/resultSources.ts#inspectKuramotoResult",
              run_ref: ref,
              projection: await inspectKuramotoResult(original.plan, original.outcome),
              recorded_source: true,
            };
            if (
              (await canonicalDigest("studio.workflow-output.v1", expected)) !== prior.output_digest
            )
              throw new WorkflowRefusal("original cached projection differs");
          }
        }
        completed.set(sid, prior);
        progress(cell.id, sid, true);
        continue;
      }
      if (journal.evaluations >= definition.sweep.evaluation_budget) {
        await checkpoint("partial");
        return { archive, journal, disposalConfirmed };
      }
      const row: WorkflowAttempt = {
        cell_id: cell.id,
        stage_id: sid,
        fingerprint,
        dependencies,
        status: "running",
        output: null,
        output_digest: null,
        reason: "Original stage reserved; terminal checkpoint pending",
        evaluated: true,
      };
      entries.push(row);
      await checkpoint("partial");
      let output: unknown = null,
        status: WorkflowAttempt["status"] = "failed";
      let reason: string | null = "Original workflow input or plan refused";
      if (options.signal.aborted) {
        status = "cancelled";
        reason = "Cancelled before original stage execution";
      } else if (!refused && plan !== null) {
        try {
          if (!plan.admission.allowed) {
            output = {
              preview_only: true,
              original_plan: await readExperimentArtifact("plan", artifactContent(plan.planMember)),
            };
            throw new WorkflowRefusal("original resource plan refused");
          }
          if (stage.verb === "validate")
            output = await previewOutput(plan, selectedParameters as ParameterDraftSource);
          else if (stage.verb === "analyse") {
            const { inspectKuramotoResult } = await import("../results/resultSources");
            const original = await savedRun(archive.json, values["run_ref"], options);
            output = {
              producer_identity:
                "studio-web/src/features/results/resultSources.ts#inspectKuramotoResult",
              run_ref: values["run_ref"],
              projection: await inspectKuramotoResult(original.plan, original.outcome),
              recorded_source: true,
            };
          } else {
            const runId = crypto.randomUUID(),
              attemptId = crypto.randomUUID(),
              events: KernelWorkerEvent[] = [];
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
              onEvent: (event) => events.push(event),
              ...(options.workerFactory === undefined
                ? {}
                : { workerFactory: options.workerFactory }),
            });
            const cancel = () => {
              void owned.cancel();
            };
            options.signal.addEventListener("abort", cancel, { once: true });
            let outcome: OwnedKernelOutcome;
            let observerFailed = false;
            try {
              try {
                options.onWorker?.(owned);
              } catch {
                observerFailed = true;
                await owned.cancel();
              }
              if (options.signal.aborted) cancel();
              outcome = await owned.result;
            } finally {
              options.signal.removeEventListener("abort", cancel);
              await owned.dispose();
              disposalConfirmed = (await owned.result).disposed;
              try {
                options.onWorker?.(null);
              } catch {
                observerFailed = true;
              }
            }
            const terminalKind = outcome.ok
              ? "result"
              : outcome.code === "cancelled"
                ? "cancelled"
                : "failed";
            if (!outcome.ok && events.at(-1)?.kind !== terminalKind) {
              events.push({
                version: 1,
                run_id: runId,
                sequence: (events.at(-1)?.sequence ?? 0) + 1,
                kind: terminalKind,
                payload: {
                  revision_hash: plan.revisionHash,
                  plan_hash: plan.planHash,
                  build_fingerprint: plan.buildFingerprint,
                  disposed: outcome.disposed,
                  reason: outcome.reason,
                  code: outcome.code,
                  origin: "original client outcome",
                },
              });
            }
            if (outcome.disposed) {
              output = {
                producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
                disposed: true,
                archive_admitted: false,
                events: archivedEvents(events),
              };
              const saved = await appendExperimentAttempt(
                { ...plan, sourceJson: archive.json },
                runId,
                attemptId,
                events,
                outcome,
                options.rawCodecs,
              );
              archive = saved;
              output = {
                producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
                run_ref: {
                  record_ref: saved.runRecordHash,
                  revision_hash: plan.revisionHash,
                  plan_hash: plan.planHash,
                  build_fingerprint: plan.buildFingerprint,
                },
                disposed: true,
                outcome: outcome.ok ? "result" : outcome.code,
              };
            } else
              output = {
                producer_identity: "studio-web/src/workers/kernelClient.ts#createOwnedKuramotoRun",
                disposed: false,
                outcome: outcome.code,
                events: archivedEvents(events),
              };
            if (observerFailed) {
              status = "failed";
              reason = "Original worker ownership observer refused; native evidence retained";
            } else if (!outcome.ok) {
              status = outcome.code === "cancelled" ? "cancelled" : "failed";
              reason = outcome.reason;
            }
          }
          if (reason === "Original workflow input or plan refused") {
            status = "complete";
            reason = null;
          }
          if (status === "complete")
            for (const port of stage.outputs)
              validatePortValue(port.type, select(output, port.path));
        } catch (cause: unknown) {
          status = "failed";
          reason =
            cause instanceof WorkflowRefusal || cause instanceof ExperimentRefusal
              ? cause.message
              : "Original stage output or archive refused; partial evidence retained";
        }
      }
      const completedRow: WorkflowAttempt = {
        ...row,
        status,
        reason,
        output,
        output_digest:
          output === null ? null : await canonicalDigest("studio.workflow-output.v1", output),
      };
      entries[entries.length - 1] = completedRow;
      await checkpoint(options.signal.aborted ? "cancelled" : "partial");
      latest.set(`${cell.id}:${sid}`, completedRow);
      if (status === "complete") completed.set(sid, completedRow);
      else finished = false;
      progress(cell.id, sid, false);
      if (!disposalConfirmed || options.signal.aborted)
        return { archive, journal, disposalConfirmed };
    }
  }
  await checkpoint(finished ? "complete" : "partial");
  return { archive, journal, disposalConfirmed };
}
