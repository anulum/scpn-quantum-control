// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable local experiment archive binding

import { committedScenario, encodeKuramotoInput } from "../../panel/kuramoto";
import type { KuramotoKernel, KuramotoRequest } from "../../panel/kuramoto";
import { canonicalDigest, documentDigest, parseExperimentRevision, parseLocalRunRecord, parseParameterSpec, parseResolvedSettings, parseWorkspaceManifest, readJson, writeJson } from "../../shared/contracts";
import type { ExperimentRevision, ParseResult, RawCodec, WorkspaceDocument } from "../../shared/contracts";
import { admitWorkspaceArchive, previewWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import type { AdmittedWorkspaceArchive, WorkspaceArchiveMember, WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";
import { admitOwnedKuramotoResources } from "../../shared/resources/kuramotoResources";
import type { ResourcePolicy } from "../../shared/resources/admission";
import type { KernelWorkerEvent, OwnedKernelOutcome } from "../../workers/kernelProtocol";
import type { ExperimentPlan } from "./experimentPlan";
import { artifactContent, decodeFloat64, decodeKernelInput, encodeFloat64, ExperimentRefusal, experimentSchemas, makeArtifact, readExperimentArtifact } from "./kuramotoArtifacts";

/** Original complete graph and exact source values admitted for one immutable revision. */
export interface LocalExperimentSource {
  /** Fully admitted original portable archive. */ readonly archive: AdmittedWorkspaceArchive;
  /** Selected immutable v1 experiment. */ readonly revision: ExperimentRevision;
  /** Original document digest, not the hash of a playback request. */ readonly revisionHash: string;
  /** Effective original source request, with validated typed parameter overrides. */ readonly request: KuramotoRequest;
  /** Original archive kernel bytes; never automatically executed. */ readonly kernelBytes: Uint8Array<ArrayBuffer>;
  /** Source program identity to compare with the currently shipped kernel. */ readonly kernelHash: string;
  /** Recorded environment; imported observations are not new measurements. */ readonly environment: Readonly<Record<string, unknown>>;
  /** Original environment artifact identity. */ readonly environmentMember: WorkspaceArchiveMember;
  /** Source-owned declared numeric byte/work policy. */ readonly policy: ResourcePolicy;
}

/** The same worker event after original lossless archive transport converts integers to bigint. */
export type ArchivedExperimentEvent = Omit<KernelWorkerEvent, "version" | "sequence"> & {
  /** Original event protocol version after lossless JSON integer transport. */ readonly version: number | bigint;
  /** Original strictly increasing event index after lossless JSON integer transport. */ readonly sequence: number | bigint;
};

/** Verify source-bound event ordering; a recorded terminal event is never independent numerical evidence. */
export function validateExperimentLifecycle(events: readonly ArchivedExperimentEvent[], runId: string, revisionHash: string, planHash: string, kernelHash: string, terminal: KernelWorkerEvent["kind"]): void {
  let previous = 0n, accepted = false;
  if (events.length === 0 || events.at(-1)!.kind !== terminal || events.at(-1)!.payload["disposed"] !== true) throw new ExperimentRefusal("disposed original terminal lifecycle required");
  for (let index = 0; index < events.length; index++) {
    const event = events[index]!;
    if (BigInt(event.version) !== 1n || BigInt(event.sequence) <= previous || event.run_id !== runId || event.payload["revision_hash"] !== revisionHash || event.payload["plan_hash"] !== planHash || event.payload["build_fingerprint"] !== kernelHash) throw new ExperimentRefusal("original attempt lifecycle identity or sequence differs");
    previous = BigInt(event.sequence);
    if (event.kind === "accepted") {
      if (accepted || index !== 0) throw new ExperimentRefusal("original worker acceptance must occur once before progress");
      accepted = true;
    } else if (event.kind === "progress") {
      if (!accepted) throw new ExperimentRefusal("original worker progress precedes admission");
    } else if (index !== events.length - 1) {
      const hostComparisonFailure = event.kind === "result" && index === events.length - 2 && terminal === "failed" && events.at(-1)!.payload["origin"] === "original client outcome";
      if (!hostComparisonFailure) throw new ExperimentRefusal("original terminal event precedes another event");
    }
  }
  if (terminal === "result" && !accepted) throw new ExperimentRefusal("successful original attempt requires worker acceptance");
}

function take<T>(parsed: ParseResult<T>): T { if (!parsed.ok) throw new ExperimentRefusal("original workspace document refused"); return parsed.value; }
function ref(member: { readonly schema: string; readonly sha256: string }): Readonly<Record<string, string>> {
  return { schema: member.schema, sha256: member.sha256, media_type: member.schema === experimentSchemas.kernel || member.schema === experimentSchemas.input ? "application/octet-stream" : "application/json" };
}
async function documentMember(document: WorkspaceDocument): Promise<WorkspaceArchiveMember> {
  const sha256 = await documentDigest(document);
  return { name: `documents/${sha256}.json`, kind: "document", schema: document.schema, sha256, content: writeJson(document) };
}
function referenced(archive: AdmittedWorkspaceArchive, reference: unknown, expected: string): WorkspaceArchiveMember {
  const value = reference as { readonly schema: string; readonly sha256: string };
  // Complete original graph admission already requires every indexed reference to exist.
  const member = archive.members.find(item => item.sha256 === value.sha256 && item.schema === value.schema)!;
  if (member.schema !== expected) throw new ExperimentRefusal("local experiment source reference is unsupported");
  return member;
}

/** Build a complete v1 archive from a declared request and an actual original kernel, without running. */
export async function createLocalExperiment(kernel: KuramotoKernel, request: KuramotoRequest, sourceArtifactId: string,
  rawCodecs: ReadonlyMap<string, RawCodec>, browserUserAgent: string): Promise<WorkspaceArchivePreview> {
  const binary = kernel.sourceBytes;
  if (!binary || binary.length === 0 || sourceArtifactId.length < 1 || sourceArtifactId.length > 4096) throw new ExperimentRefusal("bounded original kernel bytes and source identity required");
  const input = encodeKuramotoInput(request);
  if (input === null) throw new ExperimentRefusal("original source request refused");
  const admission = admitOwnedKuramotoResources({ n: request.omega.length, steps: request.steps, mode: request.mode }, kernel.bounds, binary.length);
  if (!admission.allowed) throw new ExperimentRefusal(`sample budget refused: ${admission.blockers.join(", ")}`);
  const inputMember = await makeArtifact("input", input), kernelMember = await makeArtifact("kernel", binary);
  const policyMember = await makeArtifact("policy", { ...admission.policy });
  const environmentMember = await makeArtifact("environment", { backend: admission.estimate.backend, integrator: "Rust fixed-step RK4", dtype: "float64",
    phase_unit: "rad", frequency_unit: "rad/model-time", time_unit: "model-time", seed: null,
    kernel_sha256: kernelMember.sha256, bounds: kernel.bounds, browser_user_agent: browserUserAgent });
  const effective = { backend: admission.estimate.backend, method: request.mode, precision: "float64", seed: null, time_unit: "model-time" };
  const settings = take(parseResolvedSettings({ schema: "resolved_settings.v1", body: { requested: effective, effective,
    origins: Object.fromEntries(Object.keys(effective).map(key => [key, "original shipped kernel and explicit unscaled model-time declaration"])),
    policy_ref: ref(policyMember), environment_ref: ref(environmentMember), rejected_fields: [] }, extensions: {} }));
  const settingsMember = await documentMember(settings);
  const values: Record<string, { dtype: string; shape: readonly bigint[]; values: readonly string[] }> = {
    omega: { dtype: "float64", shape: [BigInt(request.omega.length)], values: request.omega.map(encodeFloat64) },
    theta0: { dtype: "float64", shape: [BigInt(request.theta0.length)], values: request.theta0.map(encodeFloat64) },
    coupling: { dtype: "float64", shape: [], values: [encodeFloat64(request.coupling)] },
    dt: { dtype: "float64", shape: [], values: [encodeFloat64(request.dt)] },
    steps: { dtype: "uint64", shape: [], values: [String(request.steps)] },
  };
  const units: Record<string, string> = { omega: "rad/model-time", theta0: "rad", coupling: "rad/model-time", dt: "model-time", steps: "1" };
  if (request.mode === "networked") {
    values["k_nm"] = { dtype: "float64", shape: [BigInt(request.omega.length), BigInt(request.omega.length)], values: request.kNm!.map(encodeFloat64) };
    units["k_nm"] = "rad/model-time";
  }
  const specs = await Promise.all(Object.entries(values).map(async ([key, value]) => documentMember(take(parseParameterSpec({ schema: "parameter_spec.v1", body: {
    key, dtype: value.dtype, shape: value.shape, unit: units[key],
    domain: key === "steps" ? { kind: "closed_interval", lower: "1", upper: String(Math.min(kernel.bounds.maxSteps, 4096)) }
      : key === "dt" ? { kind: "closed_interval", lower: encodeFloat64(Number.MIN_VALUE), upper: encodeFloat64(Number.MAX_VALUE) } : { kind: "finite" },
    default_source: sourceArtifactId, trainable: false, dependency_keys: [],
  }, extensions: {} })))));
  const projectId = crypto.randomUUID(), now = new Date().toISOString();
  const revision = take(parseExperimentRevision({ schema: "experiment_revision.v1", body: { project_id: projectId, parent_revision_hashes: [],
    problem_ref: ref(inputMember), program_ref: ref(kernelMember), parameters: values, semantic_settings_ref: ref(settingsMember), input_refs: specs.map(ref) },
    extensions: { local_experiment: { version: 1n, model: "classical Kuramoto", source_artifact_id: sourceArtifactId } } }));
  const revisionMember = await documentMember(revision);
  const manifest = take(parseWorkspaceManifest({ schema: "quantum_workspace.v1", body: { project_id: projectId, revision_refs: [ref(revisionMember)], draft_ref: ref(revisionMember),
    created_at: now, updated_at: now, artefact_refs: [] }, extensions: { title: "Local Kuramoto experiment" } }));
  return previewWorkspaceArchive(writeJson({ schema: "quantum_workspace_archive.v1", manifest,
    members: [inputMember, kernelMember, policyMember, environmentMember, settingsMember, ...specs, revisionMember], parameter_units: units }), rawCodecs);
}

/** Open the committed original mean-field sample without changing or running its source. */
export async function createLocalSample(kernel: KuramotoKernel, rawCodecs: ReadonlyMap<string, RawCodec>, browserUserAgent: string): Promise<WorkspaceArchivePreview> {
  if (!committedScenario.ok) throw new ExperimentRefusal("original committed sample is unavailable");
  const sample = committedScenario.value;
  return createLocalExperiment(kernel, { mode: sample.mode, omega: sample.omega, theta0: sample.theta0, coupling: sample.coupling, dt: sample.dt, steps: sample.steps }, sample.artifactId, rawCodecs, browserUserAgent);
}

/** Admit the whole original graph and derive its exact selected source request, without executing. */
export async function readLocalExperiment(json: string, rawCodecs: ReadonlyMap<string, RawCodec>): Promise<LocalExperimentSource> {
  const archive = await admitWorkspaceArchive(json, rawCodecs);
  const selection = archive.manifest.body["draft_ref"] as { readonly sha256: string } | null;
  if (selection === null) throw new ExperimentRefusal("no local experiment revision selected");
  const revision = archive.revisions[selection.sha256]!;
  const extension = revision.extensions["local_experiment"] as Readonly<Record<string, unknown>> | undefined;
  if (extension?.["version"] !== 1n || extension["model"] !== "classical Kuramoto") throw new ExperimentRefusal("unsupported local experiment source version or model");
  const input = referenced(archive, revision.body["problem_ref"], experimentSchemas.input);
  const base = decodeKernelInput(artifactContent(input));
  const kernel = referenced(archive, revision.body["program_ref"], experimentSchemas.kernel);
  const settingsMember = referenced(archive, revision.body["semantic_settings_ref"], "resolved_settings.v1");
  const settings = take(parseResolvedSettings(readJson(settingsMember.content)));
  const environmentMember = referenced(archive, settings.body["environment_ref"], experimentSchemas.environment);
  const environment = await readExperimentArtifact("environment", artifactContent(environmentMember));
  if (environment["kernel_sha256"] !== kernel.sha256) throw new ExperimentRefusal("environment and original source program differ");
  const policyMember = referenced(archive, settings.body["policy_ref"], experimentSchemas.policy);
  const sourcePolicy = await readExperimentArtifact("policy", artifactContent(policyMember));
  const parameters = revision.body["parameters"] as Readonly<Record<string, { readonly dtype: string; readonly shape: readonly bigint[]; readonly values: readonly string[] }>>;
  const keys = ["omega", "theta0", "coupling", "dt", "steps", ...(base.mode === "networked" ? ["k_nm"] : [])];
  if (Object.keys(parameters).length !== keys.length || keys.some(key => parameters[key] === undefined)) throw new ExperimentRefusal("local source parameter fields differ from the original method");
  const numeric = (key: string, shape: readonly bigint[], unit: string): readonly number[] => {
    const value = parameters[key]!;
    if (value.dtype !== "float64" || value.shape.length !== shape.length || value.shape.some((dimension, index) => dimension !== shape[index]) || archive.parameterUnits[key] !== unit) throw new ExperimentRefusal("original source dtype, shape or unit differs");
    return Object.freeze(value.values.map(decodeFloat64));
  };
  const n = BigInt(base.omega.length), stepsValue = parameters["steps"]!;
  if (stepsValue.dtype !== "uint64" || stepsValue.shape.length !== 0 || archive.parameterUnits["steps"] !== "1") throw new ExperimentRefusal("original step count dtype, shape or unit differs");
  const steps = BigInt(stepsValue.values[0]!);
  if (steps < 1n || steps > 4096n) throw new ExperimentRefusal("bounded original step count required");
  const request: KuramotoRequest = Object.freeze({ mode: base.mode, omega: numeric("omega", [n], "rad/model-time"), theta0: numeric("theta0", [n], "rad"),
    coupling: numeric("coupling", [], "rad/model-time")[0]!, dt: numeric("dt", [], "model-time")[0]!, steps: Number(steps),
    ...(base.mode === "networked" ? { kNm: numeric("k_nm", [n, n], "rad/model-time") } : {}) });
  if (encodeKuramotoInput(request) === null) throw new ExperimentRefusal("effective original source request refused");
  const effective = settings.body["effective"] as Readonly<Record<string, unknown>>;
  const expected = { backend: environment["backend"], method: request.mode, precision: "float64", seed: null, time_unit: "model-time" };
  const semanticHash = await canonicalDigest("studio.kuramoto-settings.v1", expected);
  if (await canonicalDigest("studio.kuramoto-settings.v1", effective) !== semanticHash || await canonicalDigest("studio.kuramoto-settings.v1", settings.body["requested"]) !== semanticHash
    || (settings.body["rejected_fields"] as readonly unknown[]).length !== 0) throw new ExperimentRefusal("recorded requested/effective semantic settings and original source differ");
  return Object.freeze({ archive, revision, revisionHash: selection.sha256, request, kernelBytes: artifactContent(kernel), kernelHash: kernel.sha256,
    environment, environmentMember, policy: sourcePolicy as unknown as ResourcePolicy });
}

/** Append one genuinely disposed attempt to its unchanged selected revision through original admission. */
export async function appendExperimentAttempt(plan: ExperimentPlan, runId: string, attemptId: string, events: readonly KernelWorkerEvent[],
  outcome: OwnedKernelOutcome, rawCodecs: ReadonlyMap<string, RawCodec>): Promise<WorkspaceArchivePreview> {
  if (!outcome.disposed || events.length === 0) throw new ExperimentRefusal("disposed original attempt diagnostics required");
  const source = await readLocalExperiment(plan.sourceJson, rawCodecs);
  if (source.revisionHash !== plan.revisionHash) throw new ExperimentRefusal("attempt revision differs from the selected source");
  const terminal = outcome.ok ? "result" : outcome.code === "cancelled" ? "cancelled" : "failed";
  validateExperimentLifecycle(events, runId, plan.revisionHash, plan.planHash, plan.buildFingerprint, terminal);
  const additional: WorkspaceArchiveMember[] = [plan.inputMember, plan.policyMember, plan.planMember];
  const outputRefs: Readonly<Record<string, string>>[] = [];
  if (outcome.ok) {
    if (outcome.runId !== runId || outcome.revisionHash !== plan.revisionHash || outcome.planHash !== plan.planHash || outcome.buildFingerprint !== plan.buildFingerprint || outcome.run.orderParameter.length !== plan.request.steps + 1 || outcome.run.thetaFinal.length !== plan.request.omega.length || !events.some(event => event.kind === "accepted")) throw new ExperimentRefusal("successful result does not belong to the original admitted plan");
    const output = await makeArtifact("output", { revision_hash: plan.revisionHash, plan_hash: plan.planHash, kernel_sha256: plan.buildFingerprint,
      order_parameter: Array.from(outcome.run.orderParameter, encodeFloat64), theta_final: Array.from(outcome.run.thetaFinal, encodeFloat64) });
    additional.push(output); outputRefs.push(ref(output));
  }
  const wireEvents = events.map(event => {
    const payload = { ...event.payload };
    for (const key of ["orderParameter", "thetaFinal"]) {
      const value = payload[key];
      if (ArrayBuffer.isView(value) && Object.prototype.toString.call(value) === "[object Float64Array]") {
        const vector = value as Float64Array;
        payload[key] = { dtype: "float64", shape: [BigInt(vector.length)], values: Array.from(vector, encodeFloat64) };
      }
    }
    return { ...event, version: 1n, sequence: BigInt(event.sequence), payload };
  });
  const record = take(parseLocalRunRecord({ schema: "local_run_record.v1", body: { run_id: runId, attempt_id: attemptId,
    revision_hash: plan.revisionHash, plan_hash: plan.planHash, mode: "local", events: wireEvents, output_refs: outputRefs }, extensions: {} }));
  const recordMember = await documentMember(record); additional.push(recordMember);
  const original = source.archive;
  const existing = new Set(original.members.map(member => member.sha256));
  const appended = additional.filter(member => { if (existing.has(member.sha256)) return false; existing.add(member.sha256); return true; });
  const manifest = take(parseWorkspaceManifest({ ...original.manifest, body: { ...original.manifest.body,
    artefact_refs: [...original.manifest.body["artefact_refs"] as readonly unknown[], ref(recordMember)], updated_at: new Date().toISOString() } }));
  return previewWorkspaceArchive(writeJson({ schema: original.preview.schema, manifest, members: [...original.members, ...appended], parameter_units: original.parameterUnits }), rawCodecs);
}
