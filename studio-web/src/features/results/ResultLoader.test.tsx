// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual result import and asynchronous source custody

import { createHash, webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { createOwnedKuramotoRun, instantiateKuramoto } from "../../panel/kuramoto";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { prepareExperimentPlan } from "../experiments/experimentPlan";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { ResultLoader } from "./ResultLoader";
import * as sources from "./resultSources";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const fixturePath = process.env["STUDIO_ANALYSE_RESULT_PATH"];
if (!fixturePath) throw new Error("Actual original CLI export required");
const json = readFileSync(fixturePath, "utf8");
const importText = (text: string) => { fireEvent.change(screen.getByLabelText("Result producer JSON"), { target: { value: text } }); fireEvent.click(screen.getByRole("button", { name: "Inspect producer result" })); };

it("reads actual original CLI values through public controls and preserves the admitted result on refusal", async () => {
  render(<ResultLoader />);
  expect(screen.getByRole("button", { name: "Inspect producer result" })).toMatchObject({ disabled: true });
  importText(json);
  await screen.findByRole("heading", { name: "Phase-cloud synchronisation witness" });
  expect(screen.getByRole("table", { name: "Betti H0 raw values" }).textContent).toContain("0.125");
  expect(screen.getByRole("status", { name: "Result import status" }).textContent).toContain("saved workspace unchanged");
  importText("{");
  await waitFor(() => expect(screen.getByRole("status", { name: "Result import status" }).textContent).toContain("previous result"));
  expect(screen.getByRole("heading", { name: "Phase-cloud synchronisation witness" })).toBeTruthy();
  importText(json.replace('"analysis_schema": "studio.sync-analysis.v1"', '"analysis_schema": "studio.sync-analysis.v2"'));
  await waitFor(() => expect(screen.getByRole("status", { name: "Result import status" }).textContent).toContain("Unsupported"));
  expect(screen.getByRole("table", { name: "Betti H0 raw values" })).toBeTruthy();
});

it("retains newest actual source bytes when an older real digest completion arrives later", async () => {
  const original = sources.inspectAnalyseProducer;
  let release!: () => void;
  const hold = new Promise<void>(resolve => { release = resolve; });
  vi.spyOn(sources, "inspectAnalyseProducer").mockImplementationOnce(async text => { const result = await original(text); await hold; return result; });
  render(<ResultLoader />);
  importText(json);
  await waitFor(() => expect(screen.getByRole("button", { name: "Inspect producer result" })).toMatchObject({ disabled: true }));
  const newer = json + "\n";
  importText(newer);
  await screen.findByRole("heading", { name: "Phase-cloud synchronisation witness" });
  const digest = createHash("sha256").update(newer).digest("hex");
  expect(screen.getByText(digest)).toBeTruthy();
  await act(async () => { release(); await hold; });
  expect(screen.getByText(digest)).toBeTruthy();
  expect(screen.queryByText(createHash("sha256").update(json).digest("hex"))).toBeNull();
});

it("ignores late real completion and late refusal after unmount without a state write", async () => {
  for (const fail of [false, true]) {
    const original = sources.inspectAnalyseProducer;
    let release!: () => void;
    const hold = new Promise<void>(resolve => { release = resolve; });
    const spy = vi.spyOn(sources, "inspectAnalyseProducer").mockImplementationOnce(async text => { const result = await original(text); await hold; if (fail) throw new Error("delayed transport refusal"); return result; });
    const rendered = render(<ResultLoader />);
    importText(json); rendered.unmount();
    await act(async () => { release(); await hold; });
    expect(rendered.container.textContent).toBe("");
    spy.mockRestore();
  }
});

it("ignores an older failed import after current text changes while retaining the original result", async () => {
  render(<ResultLoader />);
  importText(json); await screen.findByRole("heading", { name: "Phase-cloud synchronisation witness" });
  let reject!: (cause: Error) => void;
  vi.spyOn(sources, "inspectAnalyseProducer").mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail; }));
  importText(json + "\n");
  fireEvent.change(screen.getByLabelText("Result producer JSON"), { target: { value: "current unsent text" } });
  await act(async () => { reject(new Error("stale refusal")); });
  expect(screen.getByRole("heading", { name: "Phase-cloud synchronisation witness" })).toBeTruthy();
  expect(screen.getByRole("status", { name: "Result import status" }).textContent).not.toContain("refused");
});


/** Use a genuinely executed stationary native source for lifecycle fault companions. */
async function nativeRun() {
  const wasmPath = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
  if (!wasmPath) throw new Error("Actual original WASM required");
  const kernel = await instantiateKuramoto(new Uint8Array(readFileSync(wasmPath)));
  const archive = await createLocalExperiment(kernel, { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.125, steps: 2 }, "stationary native oracle", localExperimentCodecs, "native test runtime");
  const source = await readLocalExperiment(archive.json, localExperimentCodecs), plan = await prepareExperimentPlan(source, kernel);
  const handle = createOwnedKuramotoRun({ runId: crypto.randomUUID(), revisionHash: plan.revisionHash, planHash: plan.planHash, buildFingerprint: plan.buildFingerprint, request: plan.request, wasmBytes: plan.wasmBytes, bounds: plan.bounds, resourcePolicy: plan.policy, deadlineMs: plan.deadlineMs, workerFactory: () => new BuiltKernelWorker() });
  const outcome = await handle.result;
  if (!outcome.ok) throw new Error(outcome.reason);
  return { archive, source, plan, outcome };
}

it("presents a real disposed native run and honest mismatch/unexpected-source refusals without a new worker", async () => {
  const { archive, source, plan, outcome } = await nativeRun();
  const started = BuiltKernelWorker.started;
  const rendered = render(<ResultLoader plan={plan} outcome={outcome} />);
  await screen.findByRole("heading", { name: "Original classical Kuramoto run" });
  expect(screen.getByRole("table", { name: "Original order parameter raw values" }).textContent).toContain("0.25");
  rendered.rerender(<ResultLoader plan={plan} outcome={{ ok: false, code: "cancelled", reason: "injected terminal refusal", disposed: true }} />);
  await screen.findByText("Disposed current-source successful original run required");
  rendered.rerender(<ResultLoader plan={plan} outcome={{ ...outcome, run: { ...outcome.run, orderParameter: new Float64Array([NaN, 1, 1]) } }} />);
  await screen.findByText("Original run result unavailable; source and workspace retained.");
  expect(BuiltKernelWorker.started).toBe(started); expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(source.archive.preview.json).toBe(archive.json);
});

it("ignores stale native projection success and failure after changing the original source props", async () => {
  const actual = await nativeRun();
  for (const fail of [false, true]) {
    const original = sources.inspectKuramotoResult;
    let finish!: () => void;
    const held = new Promise<ReturnType<typeof sources.inspectKuramotoResult> extends Promise<infer T> ? T : never>((resolve, reject) => {
      finish = () => { void original(actual.plan, actual.outcome).then(result => { if (fail) reject(new Error("late native refusal")); else resolve(result); }, reject); };
    });
    const spy = vi.spyOn(sources, "inspectKuramotoResult").mockImplementationOnce(() => held);
    // Delay the real adapter over a genuine native result; late output has no current-source authority.
    const rendered = render(<ResultLoader plan={actual.plan} outcome={actual.outcome} />);
    rendered.rerender(<ResultLoader plan={null} outcome={null} />);
    await act(async () => { finish(); try { await held; } catch { /* Expected deliberately delayed refusal. */ } });
    expect(screen.queryByRole("heading", { name: "Original classical Kuramoto run" })).toBeNull();
    expect(screen.queryByText("Original run result unavailable; source and workspace retained.")).toBeNull();
    spy.mockRestore(); rendered.unmount();
  }
});
