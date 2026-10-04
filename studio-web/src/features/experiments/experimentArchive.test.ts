// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable experiment archive graph

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createOwnedKuramotoRun, instantiateKuramoto } from "../../panel/kuramoto";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { encodeFloat64, experimentSchemas, localExperimentCodecs, makeArtifact } from "./kuramotoArtifacts";
import { appendExperimentAttempt, createLocalExperiment, createLocalSample, readLocalExperiment } from "./experimentArchive";
import { admitWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import { parameterSourceFromArchive, appendParameterRevision } from "../parameters/parameterRevision";
import { createParameterDraft, parameterDraftReducer } from "../parameters/parameterDraft";
import { prepareExperimentPlan } from "./experimentPlan";
import { documentDigest, readJson, writeJson } from "../../shared/contracts";
import type { WorkspaceDocument } from "../../shared/contracts";
import { validateExperimentLifecycle } from "./experimentArchive";
import type { ArchivedExperimentEvent } from "./experimentArchive";
import type { KernelWorkerEvent } from "../../workers/kernelProtocol";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });
const wasm = new Uint8Array(readFileSync(process.env["STUDIO_EXPERIMENT_WASM_PATH"] ?? "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));

it("refuses another trusted program namespace even when the original verifier confirms the actual native bytes", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const original = await createLocalSample(kernel, localExperimentCodecs, "observed browser");
  const source = await readLocalExperiment(original.json, localExperimentCodecs);
  const program = source.archive.members.find(member => member.schema === experimentSchemas.kernel)!;
  const alternateSchema = "external.kuramoto-kernel.v1";
  const originalVerifier = localExperimentCodecs.get(program.schema)!;
  // Only the negative caller-supplied producer namespace differs; its byte verifier remains real.
  const codecs = new Map(localExperimentCodecs);
  codecs.set(alternateSchema, async bytes => ({ ...await originalVerifier(bytes), schema: alternateSchema }));
  const revision = { ...source.revision, body: { ...source.revision.body, program_ref: { schema: alternateSchema, sha256: program.sha256, media_type: "application/octet-stream" } } };
  const digest = await documentDigest(revision);
  const reference = { schema: revision.schema, sha256: digest, media_type: "application/json" };
  const json = writeJson({ schema: original.schema, manifest: { ...source.archive.manifest, body: { ...source.archive.manifest.body, revision_refs: [reference], draft_ref: reference } },
    members: [...source.archive.members.filter(member => member.sha256 !== source.revisionHash && member.sha256 !== program.sha256),
      { ...program, schema: alternateSchema }, { name: `documents/${digest}.json`, kind: "document", schema: revision.schema, sha256: digest, content: writeJson(revision) }], parameter_units: source.archive.parameterUnits });
  expect((await admitWorkspaceArchive(json, codecs)).preview.json).toBe(json);
  await expect(readLocalExperiment(json, codecs)).rejects.toThrow("local experiment source reference is unsupported");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses mismatched successful result identities and shapes from a genuinely executed original attempt", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "native source identity", localExperimentCodecs, "observed browser");
  const plan = await prepareExperimentPlan(await readLocalExperiment(archive.json, localExperimentCodecs), kernel);
  const runId = crypto.randomUUID(), attemptId = crypto.randomUUID(), events: KernelWorkerEvent[] = [];
  const worker = createOwnedKuramotoRun({ runId, revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint,
    request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs,
    workerFactory: () => new BuiltKernelWorker(), onEvent: event => events.push(event) });
  const outcome = await worker.result;
  if (!outcome.ok) throw new Error("genuine native result required");
  for (const altered of [
    { ...outcome, runId: crypto.randomUUID() }, { ...outcome, revisionHash: "a".repeat(64) },
    { ...outcome, planHash: "a".repeat(64) }, { ...outcome, buildFingerprint: "a".repeat(64) },
    { ...outcome, run: { ...outcome.run, orderParameter: new Float64Array() } },
    { ...outcome, run: { ...outcome.run, thetaFinal: new Float64Array() } },
  ]) await expect(appendExperimentAttempt(plan, runId, attemptId, events, altered, localExperimentCodecs)).rejects.toThrow("successful result does not belong");
  await expect(appendExperimentAttempt({ ...plan, revisionHash: "a".repeat(64) }, runId, attemptId, events, outcome, localExperimentCodecs)).rejects.toThrow("attempt revision differs");
  await expect(appendExperimentAttempt(plan, runId, attemptId, [], outcome, localExperimentCodecs)).rejects.toThrow("disposed original attempt diagnostics");
  await expect(Reflect.apply(appendExperimentAttempt, undefined, [plan, runId, attemptId, events, { ...outcome, disposed: false }, localExperimentCodecs])).rejects.toThrow("disposed original attempt diagnostics");
  await expect(appendExperimentAttempt(plan, runId, "invalid attempt UUID", events, outcome, localExperimentCodecs)).rejects.toThrow("original workspace document refused");
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("admits a real original-kernel sample with immutable specs, source bytes and explicit model units", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalSample(kernel, localExperimentCodecs, "observed test browser");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const graph = source.archive;
  expect(source.revisionHash).toBe(graph.manifest.body["draft_ref"] && (graph.manifest.body["draft_ref"] as { sha256: string }).sha256);
  expect(source.request.omega.length).toBe(12);
  expect(source.request.steps).toBe(300);
  expect(graph.parameterUnits["theta0"]).toBe("rad");
  expect(graph.parameterUnits["dt"]).toBe("model-time");
  expect(graph.parameterUnits["omega"]).toBe("rad/model-time");
  expect(source.environment["browser_user_agent"]).toBe("observed test browser");
  expect(source.kernelBytes).toEqual(wasm);
  expect(kernel.sourceBytes).toEqual(wasm);
  expect(graph.members.some(member => member.schema === "local_run_record.v1")).toBe(false);
});

it("the public archive producer preserves the independent two-oscillator source without executing it", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const request = { mode: "mean-field" as const, omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 };
  const archive = await createLocalExperiment(kernel, request, "analytic equal-frequency two-oscillator fixture", localExperimentCodecs, "observed test browser");
  expect((await readLocalExperiment(archive.json, localExperimentCodecs)).request).toEqual(request);
  await expect(readLocalExperiment(archive.json, new Map())).rejects.toThrow("unsupported raw producer");
  await expect(createLocalExperiment({ ...kernel, sourceBytes: new Uint8Array() }, request, "fixture", localExperimentCodecs, "browser")).rejects.toThrow();
});

it("the original parameter owner creates an immutable effective input instead of rewriting source bytes", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const original = await createLocalSample(kernel, localExperimentCodecs, "test runtime");
  const source = (await parameterSourceFromArchive(original.json, localExperimentCodecs))!;
  const draft = parameterDraftReducer(createParameterDraft(source), { type: "value", key: "coupling", index: 0, text: "1.4", unit: "rad/model-time" });
  expect(draft.refusal).toBeNull();
  const child = await appendParameterRevision(original.json, source, draft.snapshot, localExperimentCodecs, new Date().toISOString());
  const current = await readLocalExperiment(child.archive.json, localExperimentCodecs);
  const parent = await readLocalExperiment(original.json, localExperimentCodecs);
  expect(current.request.coupling).toBe(1.4);
  expect(current.revision.body["parent_revision_hashes"]).toEqual([parent.revisionHash]);
  expect(current.revisionHash).not.toBe(parent.revisionHash);
  expect((await prepareExperimentPlan(current, kernel)).planHash).not.toBe((await prepareExperimentPlan(parent, kernel)).planHash);
  for (const member of parent.archive.members) expect(current.archive.members.find(row => row.sha256 === member.sha256)).toEqual(member);
});

it("preserves the original directed network matrix and refuses unsupported native source requests", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const request = { mode: "networked" as const, omega: [0, 0], theta0: [-0, 0.5], coupling: 1, dt: 0.1, steps: 2, kNm: [0, -1, 2, 0] };
  const archive = await createLocalExperiment(kernel, request, "directed signed independent input", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  expect(source.request).toEqual(request);
  expect(source.archive.parameterUnits["k_nm"]).toBe("rad/model-time");
  for (const altered of [{ ...request, steps: 0 }, { ...request, dt: NaN }, { ...request, kNm: [0] }]) await expect(createLocalExperiment(kernel, altered, "fixture", localExperimentCodecs, "browser")).rejects.toThrow("source request refused");
  for (const id of ["", "x".repeat(4097)]) await expect(createLocalExperiment(kernel, request, id, localExperimentCodecs, "browser")).rejects.toThrow("source identity required");
});

async function changedRevision(json: string, change: Readonly<Record<string, unknown>>, unitChanges: Readonly<Record<string, string>> = {}) {
  const original = await readLocalExperiment(json, localExperimentCodecs);
  const revision = { ...original.revision, ...change } as WorkspaceDocument;
  const digest = await documentDigest(revision);
  const ref = { schema: revision.schema, sha256: digest, media_type: "application/json" };
  return writeJson({ schema: original.archive.preview.schema, manifest: { ...original.archive.manifest, body: { ...original.archive.manifest.body, revision_refs: [ref], draft_ref: ref } },
    members: [...original.archive.members.filter(member => member.sha256 !== original.revisionHash), { name: `documents/${digest}.json`, kind: "document", schema: revision.schema, sha256: digest, content: writeJson(revision) }], parameter_units: { ...original.archive.parameterUnits, ...unitChanges } });
}

/** Rehash an incompatible source and its original references; only negative semantic probes use it. */
async function incompatibleSource(json: string, change: {
  readonly parameters?: Readonly<Record<string, unknown>>;
  readonly specs?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly units?: Readonly<Record<string, string>>;
  readonly environment?: Readonly<Record<string, unknown>>;
  readonly settings?: Readonly<Record<string, unknown>>;
}): Promise<string> {
  const source = await readLocalExperiment(json, localExperimentCodecs);
  let members = [...source.archive.members];
  const reference = (member: { readonly schema: string; readonly sha256: string }) => ({ schema: member.schema, sha256: member.sha256, media_type: "application/json" });
  const replaceDocument = async (oldHash: string, document: WorkspaceDocument) => {
    const sha256 = await documentDigest(document);
    const member = { name: `documents/${sha256}.json`, kind: "document" as const, schema: document.schema, sha256, content: writeJson(document) };
    members = [...members.filter(member => member.sha256 !== oldHash), member];
    return member;
  };
  let inputRefs = [...source.revision.body["input_refs"] as readonly { readonly schema: string; readonly sha256: string; readonly media_type: string }[]];
  for (const [key, altered] of Object.entries(change.specs ?? {})) {
    const original = source.archive.parameterSpecs[key]!;
    const oldHash = await documentDigest(original);
    const replacement = await replaceDocument(oldHash, { ...original, body: { ...original.body, ...altered } });
    inputRefs = inputRefs.map(ref => ref.sha256 === oldHash ? reference(replacement) : ref);
  }
  const settingsRef = source.revision.body["semantic_settings_ref"] as { readonly sha256: string };
  const originalSettings = readJson(source.archive.members.find(member => member.sha256 === settingsRef.sha256)!.content) as WorkspaceDocument;
  const settingsBody = { ...originalSettings.body, ...change.settings };
  if (change.environment) {
    const environment = await makeArtifact("environment", { ...source.environment, ...change.environment });
    members = [...members.filter(member => member.sha256 !== source.environmentMember.sha256), environment];
    settingsBody["environment_ref"] = reference(environment);
  }
  const newSettings = await replaceDocument(settingsRef.sha256, { ...originalSettings, body: settingsBody });
  const revision = await replaceDocument(source.revisionHash, { ...source.revision, body: { ...source.revision.body,
    input_refs: inputRefs, semantic_settings_ref: reference(newSettings),
    ...(change.parameters === undefined ? {} : { parameters: change.parameters }) } });
  return writeJson({ schema: source.archive.preview.schema, manifest: { ...source.archive.manifest, body: { ...source.archive.manifest.body,
    revision_refs: [reference(revision)], draft_ref: reference(revision) } }, members, parameter_units: { ...source.archive.parameterUnits, ...change.units } });
}

it("refuses admitted archives whose selected source version, units or shape differ", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalSample(kernel, localExperimentCodecs, "browser");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  for (const local of [undefined, { version: 2n, model: "classical Kuramoto" }, { version: 1n, model: "quantum spin" }]) {
    const changed = await changedRevision(archive.json, { extensions: local === undefined ? {} : { local_experiment: local } });
    await expect(readLocalExperiment(changed, localExperimentCodecs)).rejects.toThrow("source version or model");
  }
  const root = readJson(archive.json) as { manifest: { body: Record<string, unknown> } };
  root.manifest.body["draft_ref"] = null;
  await expect(readLocalExperiment(writeJson(root), localExperimentCodecs)).rejects.toThrow("no local experiment revision");
  await expect(readLocalExperiment(await changedRevision(archive.json, {}, { dt: "s" }), localExperimentCodecs)).rejects.toThrow();
  const parameters = source.revision.body["parameters"] as Record<string, unknown>;
  await expect(readLocalExperiment(await changedRevision(archive.json, { body: { ...source.revision.body, parameters: { ...parameters, unexpected: parameters["dt"] } } }), localExperimentCodecs)).rejects.toThrow();
});

it.each([
  { key: "omega", dtype: "float64", shape: [1n], values: [encodeFloat64(0)], unit: "rad/model-time", message: "source dtype, shape or unit" },
  { key: "theta0", dtype: "int64", shape: [2n], values: ["0", "1"], unit: "rad", message: "source dtype, shape or unit" },
  { key: "coupling", dtype: "float64", shape: [], values: [encodeFloat64(1)], unit: "Hz", message: "source dtype, shape or unit" },
  { key: "steps", dtype: "float64", shape: [], values: [encodeFloat64(2)], unit: "1", message: "step count dtype, shape or unit" },
  { key: "steps", dtype: "uint64", shape: [1n], values: ["2"], unit: "1", message: "step count dtype, shape or unit" },
  { key: "steps", dtype: "uint64", shape: [], values: ["2"], unit: "s", message: "step count dtype, shape or unit" },
  { key: "steps", dtype: "uint64", shape: [], values: ["0"], unit: "1", message: "bounded original step count" },
  { key: "steps", dtype: "uint64", shape: [], values: ["4097"], unit: "1", message: "bounded original step count" },
  { key: "dt", dtype: "float64", shape: [], values: [encodeFloat64(-0.1)], unit: "model-time", message: "effective original source request refused" },
])("refuses graph-admitted incompatible source parameter $key/$dtype/$unit/$values", async fault => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "original typed source", localExperimentCodecs, "observed browser");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const parameters = source.revision.body["parameters"] as Record<string, unknown>;
  const changed = await incompatibleSource(archive.json, { parameters: { ...parameters, [fault.key]: { dtype: fault.dtype, shape: fault.shape, values: fault.values } },
    specs: { [fault.key]: { dtype: fault.dtype, shape: fault.shape, unit: fault.unit, domain: { kind: "finite" } } }, units: { [fault.key]: fault.unit } });
  expect((await admitWorkspaceArchive(changed, localExperimentCodecs)).preview.json).toBe(changed);
  await expect(readLocalExperiment(changed, localExperimentCodecs)).rejects.toThrow(fault.message);
});

it("refuses admitted sources with missing method fields, a different source program or incompatible settings", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 40 }, "original semantic source", localExperimentCodecs, "observed browser");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const params = source.revision.body["parameters"] as Record<string, unknown>;
  const { coupling: removed, ...incomplete } = params;
  expect(removed).toBeDefined();
  await expect(readLocalExperiment(await incompatibleSource(archive.json, { parameters: incomplete }), localExperimentCodecs)).rejects.toThrow("parameter fields differ");
  await expect(readLocalExperiment(await incompatibleSource(archive.json, { environment: { kernel_sha256: "a".repeat(64) } }), localExperimentCodecs)).rejects.toThrow("environment and original source program differ");
  const original = { backend: source.environment["backend"], method: "mean-field", precision: "float64", seed: null, time_unit: "model-time" };
  for (const settings of [{ effective: { ...original, precision: "float32" } }, { requested: { ...original, time_unit: "s" } }, { rejected_fields: ["method"] }]) {
    await expect(readLocalExperiment(await incompatibleSource(archive.json, { settings }), localExperimentCodecs)).rejects.toThrow("requested/effective semantic settings");
  }
  await expect(createLocalExperiment({ ...kernel, bounds: { ...kernel.bounds, maxOscillators: 1 } }, source.request, "original source", localExperimentCodecs, "browser")).rejects.toThrow("request exceeds declared kernel bounds");
});

it("refuses a real native module whose original retained and transferred bytes exceed the declared source ceiling", async () => {
  // A standard empty-name custom section retains every original code/export byte.
  const larger = new Uint8Array(2 * 1024 * 1024);
  larger.set(wasm);
  const payloadLength = larger.length - wasm.length - 4;
  larger.set([0, (payloadLength & 0x7f) | 0x80, ((payloadLength >>> 7) & 0x7f) | 0x80, payloadLength >>> 14], wasm.length);
  const kernel = await instantiateKuramoto(larger);
  expect(kernel.sourceBytes?.length).toBe(larger.length);
  await expect(createLocalExperiment(kernel, { mode: "mean-field", omega: [0, 0], theta0: [0, 1], coupling: 1, dt: 0.1, steps: 2 }, "actual native module with original custom-section bytes", localExperimentCodecs, "browser")).rejects.toThrow("sample budget refused");
});

it("offline lifecycle admission rejects wrong identity, ordering and fabricated terminal success", () => {
  const binding = { revision_hash: "a".repeat(64), plan_hash: "b".repeat(64), build_fingerprint: "c".repeat(64) };
  const accepted: ArchivedExperimentEvent = { version: 1n, run_id: "run", sequence: 1n, kind: "accepted", payload: binding };
  const progress: ArchivedExperimentEvent = { ...accepted, sequence: 2n, kind: "progress" };
  const result: ArchivedExperimentEvent = { ...accepted, sequence: 3n, kind: "result", payload: { ...binding, disposed: true } };
  const admit = (events: readonly ArchivedExperimentEvent[]) => validateExperimentLifecycle(events, "run", binding.revision_hash, binding.plan_hash, binding.build_fingerprint, "result");
  expect(() => admit([accepted, progress, result])).not.toThrow();
  for (const events of [[], [result], [progress, result], [accepted, accepted, result], [accepted, { ...accepted, sequence: 2n }, result], [accepted, result, { ...result, sequence: 4n }],
    [accepted, { ...result, sequence: 0n }], [accepted, { ...result, version: 2n }], [accepted, { ...result, run_id: "other" }],
    [accepted, { ...result, payload: { ...result.payload, revision_hash: "d".repeat(64) } }], [accepted, { ...result, payload: { ...result.payload, plan_hash: "d".repeat(64) } }],
    [accepted, { ...result, payload: { ...result.payload, build_fingerprint: "d".repeat(64) } }], [accepted, { ...result, payload: { ...result.payload, disposed: false } }]]) expect(() => admit(events)).toThrow();
});
