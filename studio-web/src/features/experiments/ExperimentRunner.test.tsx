// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public local experiment journey

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { Workbench } from "../../app/Workbench";
import { instantiateKuramoto } from "../../panel/kuramoto";
import type { KuramotoKernel } from "../../panel/kuramoto";
import { createOwnedKuramotoRun } from "../../panel/kuramoto";
import { documentDigest, readJson, writeJson } from "../../shared/contracts";
import type { WorkspaceDocument } from "../../shared/contracts";
import type { KernelWorkerEvent } from "../../workers/kernelProtocol";
import type { KernelWorkerPort } from "../../workers/kernelClient";
import { WorkspaceEditor } from "../workspace/WorkspacePanel";
import { useWorkspace } from "../workspace/useWorkspace";
import { ExperimentRunner } from "./ExperimentRunner";
import { localExperimentCodecs, makeArtifact } from "./kuramotoArtifacts";
import { appendExperimentAttempt, createLocalExperiment, readLocalExperiment } from "./experimentArchive";
import { prepareExperimentPlan } from "./experimentPlan";
import { useExperimentRun } from "./useExperimentRun";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.history.replaceState(null, "", "/"); });
const wasm = new Uint8Array(readFileSync(process.env["STUDIO_EXPERIMENT_WASM_PATH"] ?? "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));

/** Exercise the actual archive/controller and native worker, including honest unavailable browser storage. */
function NativeExperiment({ loadKernel = () => instantiateKuramoto(wasm), workerFactory = () => new BuiltKernelWorker() }: { readonly loadKernel?: () => Promise<KuramotoKernel>; readonly workerFactory?: () => KernelWorkerPort }) {
  const workspace = useWorkspace(localExperimentCodecs);
  const run = useExperimentRun(workspace.draft, true, { workerFactory });
  return <><WorkspaceEditor workspace={workspace} rawCodecs={localExperimentCodecs} /><ExperimentRunner workspace={workspace} run={run} rawCodecs={localExperimentCodecs} loadKernel={loadKernel} /></>;
}

/** Await actual asynchronous admission before activating a production control. */
async function clickReady(name: string): Promise<void> {
  const button = screen.getByRole("button", { name }) as HTMLButtonElement;
  await waitFor(() => { expect(button.disabled).toBe(false); });
  fireEvent.click(button);
}

/** Open and validate the real committed sample through original user controls. */
async function openSample(): Promise<string> {
  await clickReady("Open Kuramoto sample");
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment operation" }).textContent).toContain("Original sample opened"); });
  const draft = (screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value;
  await clickReady("Validate experiment archive");
  await waitFor(() => { expect((screen.getByRole("button", { name: "Prepare numerical plan" }) as HTMLButtonElement).disabled).toBe(false); });
  return draft;
}

it("test_local_experiment_journey_01: the production workbench opens the local sample boundary", async () => {
  window.history.replaceState(null, "", "#/experiments");
  render(<Workbench>{() => null}</Workbench>);
  await screen.findByRole("heading", { name: "Local experiment" });
  expect(screen.getByRole("button", { name: "Open Kuramoto sample" })).toBeTruthy();
  expect(screen.queryByText("Experiment succeeded")).toBeNull();
});

/** Import original producer bytes through the same editable and validated workspace boundary. */
async function importDraft(json: string): Promise<void> {
  const editor = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  await waitFor(() => expect(editor.disabled).toBe(false));
  fireEvent.change(editor, { target: { value: json } });
  await clickReady("Validate experiment archive");
  await waitFor(() => expect(screen.getByRole("button", { name: "Prepare numerical plan" })).toMatchObject({ disabled: false }));
}

it("shows actual unknown imported policy fields and refuses native allocation without changing its source", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.01, steps: 1 }, "unknown imported capacity", localExperimentCodecs, "test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const policy = await makeArtifact("policy", { ...source.policy, memoryBytes: null, workUnits: null, overheadBytes: null });
  const originalSettings = source.archive.members.find(member => member.schema === "resolved_settings.v1")!;
  const settings = readJson(originalSettings.content) as WorkspaceDocument;
  const settingsDocument = { ...settings, body: { ...settings.body, policy_ref: { schema: policy.schema, sha256: policy.sha256, media_type: "application/json" } } };
  const settingsHash = await documentDigest(settingsDocument);
  const revisionDocument = { ...source.revision, body: { ...source.revision.body, semantic_settings_ref: { schema: settingsDocument.schema, sha256: settingsHash, media_type: "application/json" } } };
  const revisionHash = await documentDigest(revisionDocument);
  const revisionRef = { schema: revisionDocument.schema, sha256: revisionHash, media_type: "application/json" };
  const policyHash = (settings.body["policy_ref"] as { readonly sha256: string }).sha256;
  const json = writeJson({ schema: archive.schema, manifest: { ...source.archive.manifest, body: { ...source.archive.manifest.body, revision_refs: [revisionRef], draft_ref: revisionRef } },
    members: [...source.archive.members.filter(member => ![source.revisionHash, originalSettings.sha256, policyHash].includes(member.sha256)), policy,
      { name: `documents/${settingsHash}.json`, kind: "document", schema: settingsDocument.schema, sha256: settingsHash, content: writeJson(settingsDocument) },
      { name: `documents/${revisionHash}.json`, kind: "document", schema: revisionDocument.schema, sha256: revisionHash, content: writeJson(revisionDocument) }], parameter_units: source.archive.parameterUnits });
  expect((await readLocalExperiment(json, localExperimentCodecs)).policy.memoryBytes).toBeNull();
  render(<NativeExperiment loadKernel={async () => kernel} />);
  await importDraft(json);
  const started = BuiltKernelWorker.started;
  await clickReady("Prepare numerical plan");
  await waitFor(() => expect(screen.getByRole("status", { name: "Experiment operation" }).textContent).toContain("Numerical plan refused:"));
  await waitFor(() => expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain("memory_limit_unknown"));
  expect(within(screen.getByLabelText("Numerical plan")).getAllByText("Unknown")).toHaveLength(2);
  expect(screen.getByRole("button", { name: "Run experiment" })).toMatchObject({ disabled: true });
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(json);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("replays an actual saved explicit budget and displays the genuine original negative-zero phase bits", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [-0, -0], theta0: [-0, -0], coupling: -0, dt: 0.01, steps: 1 }, "IEEE signed-zero source", localExperimentCodecs, "test runtime");
  const plan = await prepareExperimentPlan(await readLocalExperiment(archive.json, localExperimentCodecs), kernel, { memoryBudget: "1048576" });
  const runId = crypto.randomUUID(), events: KernelWorkerEvent[] = [];
  const owned = createOwnedKuramotoRun({ runId, revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint,
    request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs,
    workerFactory: () => new BuiltKernelWorker(), onEvent: event => events.push(event) });
  const outcome = await owned.result;
  if (!outcome.ok) throw new Error("actual native signed-zero result required");
  expect(Array.from(outcome.run.thetaFinal, value => Object.is(value, -0))).toEqual([true, true]);
  const saved = await appendExperimentAttempt(plan, runId, crypto.randomUUID(), events, outcome, localExperimentCodecs);
  render(<NativeExperiment loadKernel={async () => kernel} />);
  await importDraft(saved.json);
  await clickReady("Prepare saved replay");
  await waitFor(() => expect(screen.getByLabelText("Run numeric byte budget")).toMatchObject({ value: "1048576" }));
  await clickReady("Run experiment");
  await screen.findByText("Experiment succeeded · Original float64 replay verified");
  expect(within(screen.getByLabelText("Original final phases")).getAllByText("-0")).toHaveLength(2);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("shows a genuine native disposal fault, preserves actual failure diagnostics and blocks another run", async () => {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.01, steps: 1 }, "native host disposal fault", localExperimentCodecs, "test runtime");
  const native = new BuiltKernelWorker();
  const port: KernelWorkerPort = { onmessage: null, onerror: null, postMessage: (message, transfers) => native.postMessage(message, transfers), terminate() { throw new Error("negative native host disposal fault"); } };
  native.onmessage = event => port.onmessage?.(event); native.onerror = event => port.onerror?.(event);
  try {
    render(<NativeExperiment loadKernel={async () => kernel} workerFactory={() => port} />);
    await importDraft(archive.json);
    await clickReady("Prepare numerical plan");
    await clickReady("Run experiment");
    await screen.findByRole("alert");
    expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain("worker disposal failed");
    expect(screen.getByLabelText("Original attempt diagnostics").textContent).toContain("negative native host disposal fault");
    expect(screen.getByRole("button", { name: "Run experiment" })).toMatchObject({ disabled: true });
    expect(screen.getByRole("button", { name: "Export experiment attempt" })).toMatchObject({ disabled: true });
    expect(screen.queryByText("Experiment succeeded")).toBeNull();
    expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(archive.json);
  } finally { await native.terminate(); }
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it.each([new TypeError("private loader path and interpreter details"), "private transport contents"])("keeps unexpected loader faults out of the actual UI and retains the original draft", async fault => {
  render(<NativeExperiment loadKernel={async () => { throw fault; }} />);
  const editor = screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement;
  await waitFor(() => { expect(editor.disabled).toBe(false); });
  fireEvent.change(editor, { target: { value: "original unsaved bytes" } });
  await clickReady("Open Kuramoto sample");
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment operation" }).textContent).toBe("Local experiment operation refused; current draft and saved archive retained"); });
  expect(editor.value).toBe("original unsaved bytes");
  expect(screen.queryByText(/private loader|private transport/)).toBeNull();
  expect(screen.queryByText("Experiment succeeded")).toBeNull();
});

it("shows authored refusal and zero-budget admission without allocating a native thread", async () => {
  render(<NativeExperiment />);
  const original = await openSample();
  const started = BuiltKernelWorker.started;
  fireEvent.change(screen.getByLabelText("Run numeric byte budget"), { target: { value: "01" } });
  await clickReady("Prepare numerical plan");
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment operation" }).textContent).toBe("canonical nonnegative bounded byte count required"); });
  fireEvent.change(screen.getByLabelText("Run numeric byte budget"), { target: { value: "0" } });
  await clickReady("Prepare numerical plan");
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain("refused"); });
  expect((screen.getByRole("button", { name: "Run experiment" }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(original);
  expect(BuiltKernelWorker.started).toBe(started);
});

it("runs the original native worker only for the visibly prepared budget and keeps browser-save claims unavailable", async () => {
  render(<NativeExperiment />);
  const original = await openSample();
  await clickReady("Prepare numerical plan");
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment lifecycle" }).textContent).toContain("planned"); });
  fireEvent.change(screen.getByLabelText("Run numeric byte budget"), { target: { value: "1" } });
  expect((screen.getByRole("button", { name: "Run experiment" }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.getByText("Run budget changed. Prepare a new numerical plan before execution.")).toBeTruthy();
  fireEvent.change(screen.getByLabelText("Run numeric byte budget"), { target: { value: "" } });
  await clickReady("Run experiment");
  await screen.findByText("Experiment succeeded");
  expect(screen.getByLabelText("Current experiment result")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Save experiment attempt" })).toMatchObject({ disabled: true });
  expect(screen.getByRole("button", { name: "Export experiment attempt" })).toMatchObject({ disabled: false });
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe(original);
  expect(BuiltKernelWorker.activeCount).toBe(0);
});

it("refuses a delayed original sample when the real workspace draft changes meanwhile", async () => {
  const kernel = await instantiateKuramoto(wasm);
  let finish: ((value: KuramotoKernel) => void) | undefined;
  const pending = new Promise<KuramotoKernel>(resolve => { finish = resolve; });
  render(<NativeExperiment loadKernel={() => pending} />);
  await clickReady("Open Kuramoto sample");
  fireEvent.change(screen.getByLabelText("Workspace archive JSON"), { target: { value: "new selected draft bytes" } });
  finish!(kernel);
  await waitFor(() => { expect(screen.getByRole("status", { name: "Experiment operation" }).textContent).toContain("source changed while preparing"); });
  expect((screen.getByLabelText("Workspace archive JSON") as HTMLTextAreaElement).value).toBe("new selected draft bytes");
  expect(screen.queryByText("Experiment succeeded")).toBeNull();
});
