// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public original revision and run comparison controls

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
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
import { RunComparison } from "./RunComparison";
import {
  appendParameterRevision,
  parameterSourceFromArchive,
} from "../parameters/parameterRevision";
import { createParameterDraft, parameterDraftReducer } from "../parameters/parameterDraft";
import { readComparisonArchive } from "./comparisonSources";

let json = "";
beforeAll(async () => {
  vi.stubGlobal("crypto", webcrypto);
  const wasm = new Uint8Array(
    readFileSync(
      process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
        "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
    ),
  );
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0, 0], theta0: [0, 0], coupling: 0, dt: 0.125, steps: 24 },
    "stationary comparison pagination oracle",
    localExperimentCodecs,
    "observed native test runtime",
  );
  const plan = await prepareExperimentPlan(
    await readLocalExperiment(archive.json, localExperimentCodecs),
    kernel,
  );
  const runId = crypto.randomUUID(),
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
    workerFactory: () => new BuiltKernelWorker(),
    onEvent: (event) => events.push(event),
  });
  const outcome = await owned.result;
  if (!outcome.ok) throw new Error("Genuine stationary original WASM result required");
  expect([...outcome.run.orderParameter]).toEqual(Array(25).fill(1));
  json = (
    await appendExperimentAttempt(
      plan,
      runId,
      crypto.randomUUID(),
      events,
      outcome,
      localExperimentCodecs,
    )
  ).json;
}, 15_000);
beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

/** Read both original sources through visible production controls. */
async function admitBoth() {
  fireEvent.click(screen.getByRole("button", { name: "Read baseline archive" }));
  await screen.findByLabelText("Baseline revision");
  fireEvent.click(screen.getByRole("button", { name: "Read candidate archive" }));
  await screen.findByLabelText("Candidate revision");
}

/** Complete an explicit read-only comparison through its public control. */
async function compare() {
  fireEvent.click(screen.getByRole("button", { name: "Compare selected immutable sources" }));
  await screen.findByRole("table", { name: "Original matched and unmatched values" });
}

it("test_immutable_run_comparison_03: compare, back and reopen retain original archives and raw values without executing another worker", async () => {
  const started = BuiltKernelWorker.started;
  let view = render(<RunComparison sourceJson={json} />);
  await admitBoth();
  await compare();
  expect(screen.getByText("No original semantic fields differ.")).toBeTruthy();
  expect(screen.getByText(/Matched: 27; baseline only: 0; candidate only: 0/)).toBeTruthy();
  const table = screen.getByRole("table", { name: "Original matched and unmatched values" });
  expect(within(table).getAllByRole("row")).toHaveLength(21);
  const row = within(table).getAllByRole("row")[1];
  if (row === undefined) throw new Error("Original stationary value row required");
  expect(
    within(row)
      .getAllByRole("cell")
      .map((cell) => cell.textContent),
  ).toEqual(["0", "matched", "1", "1", "0", "available"]);
  fireEvent.click(screen.getByRole("button", { name: "Next comparison values" }));
  expect(within(table).getAllByRole("row")).toHaveLength(8);
  expect(screen.getByText(/Original observations 21–27 of 27/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Previous comparison values" }));
  expect(screen.getByLabelText("Baseline archive JSON")).toHaveProperty("value", json);
  expect(screen.getByLabelText("Candidate archive JSON")).toHaveProperty("value", json);
  view.unmount();
  view = render(<RunComparison sourceJson={json} rawCodecs={localExperimentCodecs} />);
  await admitBoth();
  await compare();
  expect(screen.getByText(/Matched: 27/)).toBeTruthy();
  expect(BuiltKernelWorker.started).toBe(started);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(screen.getByLabelText("Baseline archive JSON")).toHaveProperty("value", json);
  view.unmount();
});

it("keeps semantic evidence differences and blocked unavailable results visible before raw values", async () => {
  render(<RunComparison sourceJson={json} />);
  await admitBoth();
  fireEvent.change(screen.getByLabelText("Candidate recorded run"), { target: { value: "" } });
  await compare();
  expect(screen.getByRole("alert").textContent).toContain("Candidate: No run selected");
  const differences = screen.getByRole("table", { name: "Original semantic differences" });
  expect(differences.textContent).toContain("output_evidence");
  expect(differences.textContent).toContain("(absent)");
  expect(
    screen.getByRole("table", { name: "Original matched and unmatched values" }).textContent,
  ).toContain("baseline-only");
  expect(
    screen.getByRole("table", { name: "Original matched and unmatched values" }).textContent,
  ).toContain("blocked");
  fireEvent.change(screen.getByLabelText("Baseline recorded run"), { target: { value: "" } });
  await compare();
  await screen.findByText(/Original observations 0–0 of 0/);
  expect(screen.getByText("No original semantic fields differ.")).toBeTruthy();
});

it("reads original local files and can explicitly restore current source after a refused edit", async () => {
  render(<RunComparison sourceJson={json} />);
  await admitBoth();
  await compare();
  const baseline = screen.getByLabelText("Baseline archive JSON");
  fireEvent.change(baseline, { target: { value: "{" } });
  fireEvent.click(screen.getByRole("button", { name: "Read baseline archive" }));
  await screen.findByText(
    "Comparison refused; previous admitted comparison and saved workspace retained.",
  );
  expect(screen.getByRole("table", { name: "Original matched and unmatched values" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Use current workspace as baseline" }));
  fireEvent.click(screen.getByRole("button", { name: "Use current workspace as candidate" }));
  expect(baseline).toHaveProperty("value", json);
  fireEvent.change(screen.getByLabelText("Baseline archive file"), { target: { files: [] } });
  fireEvent.change(screen.getByLabelText("Candidate archive file"), {
    target: { files: [new File([`${json}\n`], "original.json")] },
  });
  await screen.findByText(
    "Original source bytes read locally. Complete archive admission is required; saved data unchanged.",
  );
  expect(screen.getByLabelText("Candidate archive JSON")).toHaveProperty("value", `${json}\n`);
});

it("starts with no fabricated source or execution authority", () => {
  render(<RunComparison />);
  expect(screen.getByRole("button", { name: "Compare selected immutable sources" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(screen.getByRole("button", { name: "Use current workspace as baseline" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(screen.queryByRole("table", { name: "Original matched and unmatched values" })).toBeNull();
});

it("selects an original parent and actual recorded run from a child archive without rebinding the saved draft", async () => {
  const source = await parameterSourceFromArchive(json, localExperimentCodecs);
  if (source === null) throw new Error("Original stationary immutable revision required");
  const changed = parameterDraftReducer(createParameterDraft(source), {
    type: "value",
    key: "dt",
    index: 0,
    text: "0.25",
    unit: "model-time",
  });
  const child = await appendParameterRevision(
    json,
    source,
    changed.snapshot,
    localExperimentCodecs,
    new Date().toISOString(),
  );
  const original = await readComparisonArchive(json, localExperimentCodecs);
  const run = original.runs.at(0);
  if (run === undefined) throw new Error("Actual original stationary recorded run required");
  render(<RunComparison sourceJson={child.archive.json} />);
  await admitBoth();
  fireEvent.change(screen.getByLabelText("Baseline revision"), {
    target: { value: run.revisionHash },
  });
  fireEvent.change(screen.getByLabelText("Candidate revision"), {
    target: { value: run.revisionHash },
  });
  fireEvent.change(screen.getByLabelText("Candidate recorded run"), { target: { value: "" } });
  fireEvent.change(screen.getByLabelText("Candidate recorded run"), {
    target: { value: run.hash },
  });
  await compare();
  expect(screen.getByText(/Matched: 27; baseline only: 0; candidate only: 0/)).toBeTruthy();
  expect(screen.getByLabelText("Candidate archive JSON")).toHaveProperty(
    "value",
    child.archive.json,
  );
  fireEvent.change(screen.getByLabelText("Candidate revision"), {
    target: { value: child.revisionHash },
  });
  fireEvent.click(screen.getByRole("button", { name: "Compare selected immutable sources" }));
  await screen.findByRole("table", { name: "Original semantic differences" });
  expect(
    screen.getByRole("table", { name: "Original semantic differences" }).textContent,
  ).toContain("parameters.dt");
  expect(screen.getByRole("alert").textContent).toContain("No run selected");
  expect(screen.getByLabelText("Baseline archive JSON")).toHaveProperty(
    "value",
    child.archive.json,
  );
});
