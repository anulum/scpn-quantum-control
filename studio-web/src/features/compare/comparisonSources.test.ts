// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original archive and genuine WASM comparison binding

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createOwnedKuramotoRun, instantiateKuramoto } from "../../panel/kuramoto";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import {
  appendExperimentAttempt,
  createLocalExperiment,
  readLocalExperiment,
} from "../experiments/experimentArchive";
import { prepareExperimentPlan } from "../experiments/experimentPlan";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import type { KernelWorkerEvent } from "../../workers/kernelProtocol";
import { projectComparisonSource, readComparisonArchive } from "./comparisonSources";
import type { ComparisonArchive } from "./comparisonSources";
import { compareImmutableRuns } from "./comparisonModel";
import { documentDigest, readJson, writeJson } from "../../shared/contracts";
import type { WorkspaceDocument } from "../../shared/contracts";
import { createParameterDraft, parameterDraftReducer } from "../parameters/parameterDraft";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import {
  appendParameterRevision,
  parameterSourceFromArchive,
} from "../parameters/parameterRevision";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  vi.unstubAllGlobals();
});
/** Require the actual source-owned recorded run produced by the real native fixture. */
function recordedRun(admitted: ComparisonArchive) {
  const run = admitted.runs.at(0);
  if (run === undefined) throw new Error("Source-owned explicitly indexed run document required");
  return run;
}

const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);

/** Build and run the actual original stationary two-node kernel; no mock output producer. */
async function stationaryArchive() {
  const kernel = await instantiateKuramoto(wasm);
  const draft = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.125, steps: 2 },
    "independent stationary phase oracle",
    localExperimentCodecs,
    "observed native test runtime",
  );
  const source = await readLocalExperiment(draft.json, localExperimentCodecs);
  const plan = await prepareExperimentPlan(source, kernel);
  const events: KernelWorkerEvent[] = [],
    runId = crypto.randomUUID();
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
    workerFactory: () => new BuiltKernelWorker(),
    onEvent: (event) => events.push(event),
  });
  const outcome = await owned.result;
  if (!outcome.ok) throw new Error("Genuine original native stationary result required");
  expect([...outcome.run.orderParameter]).toEqual([1, 1, 1]);
  expect([...outcome.run.thetaFinal]).toEqual([0, 0]);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  return {
    kernel,
    draft,
    source,
    plan,
    archive: await appendExperimentAttempt(
      plan,
      runId,
      crypto.randomUUID(),
      events,
      outcome,
      localExperimentCodecs,
    ),
  };
}

it("projects original immutable revision, disposed run and exact stationary output without rewriting source bytes", async () => {
  const original = await stationaryArchive();
  const admitted = await readComparisonArchive(original.archive.json, localExperimentCodecs);
  expect(admitted.revisions.map((row) => row.hash)).toEqual([original.plan.revisionHash]);
  expect(admitted.runs).toHaveLength(1);
  const run = recordedRun(admitted);
  expect(run.state).toBe("result");
  const projection = await projectComparisonSource(admitted, original.plan.revisionHash, run.hash);
  expect(projection.unavailableReason).toBeNull();
  expect(projection.observations).toEqual([
    { key: "order-parameter", time: 0, value: 1 },
    { key: "order-parameter", time: 0.125, value: 1 },
    { key: "order-parameter", time: 0.25, value: 1 },
    { key: "oscillator/0", time: 0.25, value: 0 },
    { key: "oscillator/1", time: 0.25, value: 0 },
  ]);
  const result = compareImmutableRuns(projection, projection);
  expect(result.rows.map((row) => row.delta)).toEqual([0, 0, 0, 0, 0]);
  expect(result.blockers).toEqual([]);
  expect(admitted.archive.preview.json).toBe(original.archive.json);
  expect(projection.semantics["program"]).toEqual(original.source.revision.body["program_ref"]);
  expect(projection.semantics["effective_settings.backend"]).toBe(
    "shipped-kuramoto-wasm-float64-owned-worker",
  );
  expect(Object.isFrozen(projection.observations[0])).toBe(true);
});

it("retains original revision and raw members when selecting another immutable child or no successful run", async () => {
  const original = await stationaryArchive();
  const parameterSource = await parameterSourceFromArchive(
    original.archive.json,
    localExperimentCodecs,
  );
  if (parameterSource === null) throw new Error("Original immutable parameter source required");
  const state = parameterDraftReducer(createParameterDraft(parameterSource), {
    type: "value",
    key: "dt",
    index: 0,
    text: "0.25",
    unit: "model-time",
  });
  const child = await appendParameterRevision(
    original.archive.json,
    parameterSource,
    state.snapshot,
    localExperimentCodecs,
    new Date().toISOString(),
  );
  const admitted = await readComparisonArchive(child.archive.json, localExperimentCodecs);
  const old = await projectComparisonSource(
    admitted,
    original.plan.revisionHash,
    recordedRun(admitted).hash,
  );
  const next = await projectComparisonSource(admitted, child.revisionHash, null);
  expect(next.unavailableReason).toBe("No run selected");
  expect(compareImmutableRuns(old, next).semanticDiff.map((row) => row.field)).toContain(
    "parameters.dt",
  );
  expect(old.observations[1]?.time).toBe(0.125);
  expect(
    (
      await readLocalExperiment(
        child.archive.json,
        localExperimentCodecs,
        original.plan.revisionHash,
      )
    ).request.dt,
  ).toBe(0.125);
  await expect(projectComparisonSource(admitted, "f".repeat(64), null)).rejects.toThrow(
    "Selected immutable revision is absent",
  );
  await expect(
    projectComparisonSource(admitted, child.revisionHash, recordedRun(admitted).hash),
  ).rejects.toThrow("Selected run belongs to another immutable revision");
  await expect(
    projectComparisonSource(admitted, original.plan.revisionHash, "f".repeat(64)),
  ).rejects.toThrow("Selected recorded run is absent");
  const before = (await readComparisonArchive(original.archive.json, localExperimentCodecs)).archive
    .members;
  for (const member of before)
    expect(
      admitted.archive.members.find((candidate) => candidate.sha256 === member.sha256),
    ).toEqual(member);
});

it("refuses unsupported archive versions, digest tampering and unavailable verifiers without touching the admitted original", async () => {
  const original = await stationaryArchive();
  const text = original.archive.json;
  await expect(
    readComparisonArchive(
      text.replace("quantum_workspace_archive.v1", "quantum_workspace_archive.v2"),
      localExperimentCodecs,
    ),
  ).rejects.toThrow("Unsupported archive schema");
  const wire = readJson(text) as { members: Array<{ content: string }> };
  const first = wire.members.at(0);
  if (first === undefined) throw new Error("Original kernel input bytes required");
  first.content += "00";
  await expect(readComparisonArchive(writeJson(wire), localExperimentCodecs)).rejects.toThrow();
  await expect(readComparisonArchive(text, new Map())).rejects.toThrow("unsupported raw producer");
  expect(original.archive.json).toBe(text);
});

it("preserves explicit recorded unavailable attempts with original semantic identity", async () => {
  const original = await stationaryArchive();
  const runId = crypto.randomUUID();
  const events: KernelWorkerEvent[] = [
    {
      version: 1,
      run_id: runId,
      sequence: 1,
      kind: "cancelled",
      payload: {
        revision_hash: original.plan.revisionHash,
        plan_hash: original.plan.planHash,
        build_fingerprint: original.plan.buildFingerprint,
        disposed: true,
        reason: "explicit original cancellation",
      },
    },
  ];
  const archive = await appendExperimentAttempt(
    original.plan,
    runId,
    crypto.randomUUID(),
    events,
    { ok: false, code: "cancelled", reason: "explicit original cancellation", disposed: true },
    localExperimentCodecs,
  );
  const admitted = await readComparisonArchive(archive.json, localExperimentCodecs);
  const projection = await projectComparisonSource(
    admitted,
    original.plan.revisionHash,
    recordedRun(admitted).hash,
  );
  expect(projection.unavailableReason).toBe("Recorded run is cancelled; no completed output");
  expect(projection.observations).toEqual([]);
});

it("keeps unsupported model metadata inspectable while refusing numerical comparison", async () => {
  const original = await stationaryArchive();
  const document: WorkspaceDocument = {
    ...original.source.revision,
    extensions: { local_experiment: { version: 1n, model: "unqualified alternative" } },
  };
  const hash = await documentDigest(document),
    ref = { schema: document.schema, sha256: hash, media_type: "application/json" };
  const wire = readJson(original.draft.json) as {
    manifest: { body: Record<string, unknown> };
    members: unknown[];
  };
  wire.manifest.body["revision_refs"] = [ref];
  wire.manifest.body["draft_ref"] = ref;
  wire.members = [
    ...wire.members,
    {
      name: `documents/${hash}.json`,
      kind: "document",
      schema: document.schema,
      sha256: hash,
      content: writeJson(document),
    },
  ];
  const admitted = await readComparisonArchive(writeJson(wire), localExperimentCodecs);
  const snapshot = await projectComparisonSource(admitted, hash, null);
  expect(snapshot.semantics["revision_extensions"]).toEqual(document.extensions);
  expect(snapshot.unavailableReason).toBe("No run selected");
});

it("keeps explicitly synthetic conformance metadata unqualified, including absent model/precision and an empty event ledger", async () => {
  const archive = await conformanceArchive(corpusText, false);
  const wire = readJson(archive.json) as {
    manifest: { body: Record<string, unknown> };
    members: Array<{ schema: string; sha256: string }>;
  };
  const originalRun = wire.members.find((member) => member.schema === "local_run_record.v1");
  if (originalRun === undefined)
    throw new Error("Original synthetic empty-ledger fixture required");
  wire.manifest.body["artefact_refs"] = [
    { schema: originalRun.schema, sha256: originalRun.sha256, media_type: "application/json" },
  ];
  const indexedJson = writeJson(wire);
  const admitted = await readComparisonArchive(indexedJson, conformanceCodecs);
  const run = recordedRun(admitted);
  expect(run.state).toBe("no terminal event");
  const source = await projectComparisonSource(admitted, run.revisionHash, run.hash);
  expect(source.unavailableReason).toBe(
    "Recorded run is missing a terminal event; no completed output",
  );
  expect(source.key.backend).toBe("null");
  expect(source.key.precision).toBe("null");
  expect(source.observations).toEqual([]);
  expect(source.semantics["effective_settings.shots"]).toBe("4096");
  expect(admitted.archive.preview.json).toBe(indexedJson);
});

/** Re-address a deliberately corrupted recorded plan/output while retaining the real original archive outside the negative fixture. */
async function corruptedRecord(
  json: string,
  change: {
    readonly plan?: Readonly<Record<string, unknown>>;
    readonly output?: Readonly<Record<string, unknown>>;
    readonly removeOutput?: boolean;
    readonly wrongOutputSchema?: boolean;
    readonly policySource?: string;
  },
) {
  const { artifactContent, makeArtifact, readExperimentArtifact, experimentSchemas } = await import(
    "../experiments/kuramotoArtifacts"
  );
  const { admitOwnedKuramotoResources } = await import("../../shared/resources/kuramotoResources");
  const admitted = await readComparisonArchive(json, localExperimentCodecs);
  const run = recordedRun(admitted),
    archive = admitted.archive;
  const document = archive.documents[run.hash];
  if (document === undefined) throw new Error("Actual original run record required");
  const planHash = document.body["plan_hash"] as string;
  const originalPlan = archive.members.find((member) => member.sha256 === planHash);
  if (originalPlan === undefined) throw new Error("Actual original numerical plan required");
  const payload = await readExperimentArtifact("plan", artifactContent(originalPlan));
  const changedPlan = { ...payload, ...change.plan };
  const shape = changedPlan["shape"] as {
    n: number;
    steps: number;
    mode: "mean-field" | "networked";
  };
  changedPlan["admission"] = admitOwnedKuramotoResources(
    shape,
    changedPlan["bounds"] as { maxOscillators: number; maxSteps: number },
    changedPlan["binary_bytes"] as number,
    changedPlan["policy"] as Parameters<typeof admitOwnedKuramotoResources>[3],
  );
  const plan = await makeArtifact("plan", changedPlan);
  const references = document.body["output_refs"] as Array<{ sha256: string }>;
  const outputMember = archive.members.find((member) => member.sha256 === references[0]?.sha256);
  if (outputMember === undefined) throw new Error("Actual original numerical output required");
  const outputPayload = await readExperimentArtifact("output", artifactContent(outputMember));
  const output = await makeArtifact("output", {
    ...outputPayload,
    plan_hash: plan.sha256,
    ...change.output,
  });
  let additional = [plan, output];
  if (change.policySource !== undefined) {
    const policy = await makeArtifact("policy", {
      ...(changedPlan["policy"] as Record<string, unknown>),
      source: change.policySource,
    });
    const replacement = await makeArtifact("plan", {
      ...changedPlan,
      policy_sha256: policy.sha256,
    });
    additional = [
      replacement,
      policy,
      await makeArtifact("output", {
        ...outputPayload,
        plan_hash: replacement.sha256,
        ...change.output,
      }),
    ];
  }
  const chosenPlan = additional[0];
  const chosenOutput = additional.at(-1);
  if (chosenPlan === undefined || chosenOutput === undefined)
    throw new Error("Declared negative custody fixture required");
  const changed: WorkspaceDocument = {
    ...document,
    body: {
      ...document.body,
      plan_hash: chosenPlan.sha256,
      events: (document.body["events"] as Array<{ payload: Record<string, unknown> }>).map(
        (event) => ({ ...event, payload: { ...event.payload, plan_hash: chosenPlan.sha256 } }),
      ),
      output_refs: change.removeOutput
        ? []
        : [
            {
              schema: change.wrongOutputSchema ? experimentSchemas.policy : chosenOutput.schema,
              sha256: change.wrongOutputSchema ? payload["policy_sha256"] : chosenOutput.sha256,
              media_type: "application/json",
            },
          ],
    },
  };
  const hash = await documentDigest(changed),
    reference = { schema: changed.schema, sha256: hash, media_type: "application/json" };
  const originalHashes = new Set(archive.members.map((member) => member.sha256));
  const appended = additional.filter((member) => !originalHashes.has(member.sha256));
  const recordMember = {
    name: `documents/${hash}.json`,
    kind: "document",
    schema: changed.schema,
    sha256: hash,
    content: writeJson(changed),
  };
  return {
    hash,
    revisionHash: run.revisionHash,
    json: writeJson({
      schema: archive.preview.schema,
      manifest: {
        ...archive.manifest,
        body: { ...archive.manifest.body, artefact_refs: [reference] },
      },
      members: [
        ...archive.members.filter((member) => member.sha256 !== run.hash),
        ...appended,
        recordMember,
      ],
      parameter_units: archive.parameterUnits,
    }),
    original: json,
  };
}

it.each([
  { revision_hash: "a".repeat(64) },
  { kernel_sha256: "a".repeat(64) },
  { environment_sha256: "a".repeat(64) },
  { input_sha256: "a".repeat(64) },
])(
  "refuses re-addressed recorded plan bindings that differ from the actual immutable source",
  async (plan) => {
    const original = await stationaryArchive();
    const wrong = await corruptedRecord(original.archive.json, { plan });
    const admitted = await readComparisonArchive(wrong.json, localExperimentCodecs);
    const source = await projectComparisonSource(admitted, wrong.revisionHash, wrong.hash);
    expect(source.unavailableReason).toBe(
      "Recorded numerical plan differs from the original immutable source",
    );
    expect(source.observations).toEqual([]);
    expect(original.archive.json).toBe(wrong.original);
  },
);

it.each([
  {
    output: { revision_hash: "a".repeat(64) },
    reason: "Recorded output belongs to another immutable source or plan",
  },
  {
    output: { plan_hash: "a".repeat(64) },
    reason: "Recorded output belongs to another immutable source or plan",
  },
  {
    output: { kernel_sha256: "a".repeat(64) },
    reason: "Recorded output belongs to another immutable source or plan",
  },
  {
    output: { order_parameter: ["3ff0000000000000"] },
    reason: "Recorded numerical output shape differs from its source",
  },
  { removeOutput: true, reason: "One original recorded numerical output required" },
  { wrongOutputSchema: true, reason: "One original recorded numerical output required" },
  {
    policySource: "deliberately changed negative custody label",
    reason: "Recorded numerical policy differs from its original artifact",
  },
])(
  "refuses incompatible recorded output/policy custody without comparing another source",
  async ({ reason, ...change }) => {
    const original = await stationaryArchive();
    const wrong = await corruptedRecord(original.archive.json, change);
    const admitted = await readComparisonArchive(wrong.json, localExperimentCodecs);
    const source = await projectComparisonSource(admitted, wrong.revisionHash, wrong.hash);
    expect(source.unavailableReason).toBe(reason);
    expect(source.observations).toEqual([]);
    expect(original.archive.json).toBe(wrong.original);
  },
);

it("refuses a different trusted plan namespace even when its original byte verifier confirms the real recorded plan", async () => {
  const original = await stationaryArchive();
  const admitted = await readComparisonArchive(original.archive.json, localExperimentCodecs);
  const run = recordedRun(admitted),
    document = admitted.archive.documents[run.hash];
  if (document === undefined) throw new Error("Actual original run record required");
  const planHash = document.body["plan_hash"] as string;
  const originalPlan = admitted.archive.members.find((member) => member.sha256 === planHash);
  if (originalPlan === undefined) throw new Error("Actual original plan bytes required");
  const verifier = localExperimentCodecs.get(originalPlan.schema);
  if (verifier === undefined) throw new Error("Actual original plan verifier required");
  const alternate = "external.recorded-plan.v1",
    codecs = new Map(localExperimentCodecs);
  codecs.set(alternate, async (bytes) => ({ ...(await verifier(bytes)), schema: alternate }));
  const wire = readJson(original.archive.json) as {
    members: Array<{ sha256: string; schema: string }>;
  };
  const replacement = wire.members.find((member) => member.sha256 === planHash);
  if (replacement === undefined) throw new Error("Original indexed plan member required");
  replacement.schema = alternate;
  const source = await readComparisonArchive(writeJson(wire), codecs);
  const projection = await projectComparisonSource(source, run.revisionHash, run.hash);
  expect(projection.unavailableReason).toBe(
    "Original comparison source member is absent or unsupported",
  );
  expect(projection.observations).toEqual([]);
  expect(original.archive.json).toBe(admitted.archive.preview.json);
});

it("does not turn a genuine verifier runtime fault into admitted numerical comparison", async () => {
  const original = await stationaryArchive();
  let refuse = false;
  const codecs = new Map(
    [...localExperimentCodecs].map(
      ([schema, verify]) =>
        [
          schema,
          async (bytes: Uint8Array) => {
            const identity = await verify(bytes);
            if (refuse) throw new Error("injected original verifier runtime fault");
            return identity;
          },
        ] as const,
    ),
  );
  const admitted = await readComparisonArchive(original.archive.json, codecs);
  const run = recordedRun(admitted),
    started = BuiltKernelWorker.started;
  refuse = true;
  await expect(projectComparisonSource(admitted, run.revisionHash, run.hash)).rejects.toThrow(
    "injected original verifier runtime fault",
  );
  expect(admitted.archive.preview.json).toBe(original.archive.json);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});
