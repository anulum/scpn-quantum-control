// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — genuine workflow native worker tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import * as preF09 from "../../../test-support/workflow-history/pre-f09.mjs";
import * as relocatedF09 from "../../../test-support/workflow-history/relocated-f09.mjs";
import { instantiateKuramoto } from "../browser/kuramoto";
import { canonicalBytes, canonicalDigest, readJson, writeJson } from "../contracts";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { readWorkflowArchive } from "./workflowArchive";
import { runLocalWorkflow } from "./workflowExecution";
import type { WorkflowExecutionOptions } from "./workflowExecution";
import { parseWorkflow } from "./workflowModel";

beforeEach(() => vi.stubGlobal("crypto", webcrypto));
afterEach(() => {
  expect(BuiltKernelWorker.activeCount).toBe(0);
  vi.unstubAllGlobals();
});

const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);

function graph(kind: "preview" | "projection") {
  const type = { schema: "studio.workflow-run-reference.v1", dtype: "json", shape: [], unit: "1" };
  const stage = {
    id: "validate",
    adapter: "local-kuramoto",
    verb: "validate",
    backend: "shipped-kuramoto-wasm-float64",
    parameters: {},
    inputs: [],
    outputs: [],
    depends_on: [],
  };
  const simulate = {
    ...stage,
    id: "simulate",
    verb: "simulate",
    outputs: [{ name: "run", path: ["run_ref"], type }],
  };
  const analyse = {
    ...stage,
    id: "analyse",
    verb: "analyse",
    inputs: [{ parameter: "run_ref", source_stage: "simulate", source_port: "run", type }],
  };
  return parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: `authentic-workflow-history-${kind}`,
      stages: kind === "preview" ? [stage] : [simulate, analyse],
      sweep: {
        axes: [],
        seeds: ["0"],
        seed_binding: null,
        evaluation_budget: kind === "preview" ? 1n : 2n,
      },
    },
    extensions: {
      source: "Original classical Kuramoto native fixture, no quantum-model equivalence claim",
    },
  });
}

it.each([
  { version: "pre-F09", executor: preF09, prefix: "features", kind: "preview" },
  { version: "pre-F09", executor: preF09, prefix: "features", kind: "projection" },
  { version: "relocated-F09", executor: relocatedF09, prefix: "shared", kind: "preview" },
  { version: "relocated-F09", executor: relocatedF09, prefix: "shared", kind: "projection" },
] as const)(
  "resumes authentic $version $kind history without rewriting or replacement execution",
  async ({ executor, prefix, kind }) => {
    const kernel = await instantiateKuramoto(wasm);
    const initial = await createLocalExperiment(
      kernel,
      {
        mode: "mean-field",
        omega: [0.2, 0.2],
        theta0: [0, 0.8],
        coupling: 1.4,
        dt: 0.01,
        steps: 4,
      },
      "authentic workflow history",
      localExperimentCodecs,
      "native workflow history test",
    );
    const source = await readLocalExperiment(initial.json, localExperimentCodecs);
    let saved = initial;
    const options: Omit<WorkflowExecutionOptions, "sourceJson"> = {
      baseRevision: source.revisionHash,
      definition: graph(kind),
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      workerFactory: () => new BuiltKernelWorker(),
      save: async (candidate, prior) => {
        expect(prior).toBe(saved.json);
        saved = candidate;
      },
    };
    const started = BuiltKernelWorker.started;
    const original = await executor.runLocalWorkflow({ ...options, sourceJson: initial.json });
    expect(original.journal.state).toBe("complete");
    expect(original.journal.evaluations).toBe(kind === "preview" ? 1n : 2n);
    expect(original.journal.entries).toHaveLength(kind === "preview" ? 1 : 2);
    expect(BuiltKernelWorker.started - started).toBe(kind === "preview" ? 0 : 1);
    const producer =
      "studio-web/src/" +
      prefix +
      "/" +
      (kind === "preview"
        ? "experiments/experimentPlan.ts#prepareExperimentPlan"
        : "results/resultSources.ts#inspectKuramotoResult");
    const terminal = original.journal.entries.at(-1);
    if (terminal === undefined) throw new Error("Actual historical terminal stage is absent");
    expect((terminal.output as Record<string, unknown>)["producer_identity"]).toBe(producer);
    const before = canonicalBytes("original-workflow-journal.v1", original.journal);
    const members = (await readWorkflowArchive(original.archive.json, localExperimentCodecs)).source
      .members;
    const workers = BuiltKernelWorker.started;
    for (const resume of [executor.runLocalWorkflow, runLocalWorkflow, executor.runLocalWorkflow]) {
      saved = original.archive;
      const progress: boolean[] = [];
      const result = await resume({
        ...options,
        sourceJson: original.archive.json,
        onProgress: (row) => progress.push(row.reused),
      });
      expect(result.journal.state).toBe("complete");
      expect(canonicalBytes("original-workflow-journal.v1", result.journal)).toEqual(before);
      expect(
        (await readWorkflowArchive(result.archive.json, localExperimentCodecs)).source.members,
      ).toEqual(members);
      expect(progress).toEqual(original.journal.entries.map(() => true));
      expect(BuiltKernelWorker.started).toBe(workers);
    }
    for (const fault of ["unknown-producer", "changed-payload"]) {
      const wire = readJson(original.archive.json) as Record<string, unknown>;
      const extension = (
        (wire["manifest"] as Record<string, unknown>)["extensions"] as Record<string, unknown>
      )["experiment_workflows"] as Record<string, unknown>;
      const item = (extension["items"] as Record<string, unknown>[])[0] as Record<string, unknown>;
      const entries = (
        (item["journal"] as Record<string, unknown>)["body"] as Record<string, unknown>
      )["entries"] as Record<string, unknown>[];
      const entry = entries.at(-1) as Record<string, unknown>,
        output = entry["output"] as Record<string, unknown>;
      output[
        fault === "unknown-producer"
          ? "producer_identity"
          : kind === "preview"
            ? "preview_only"
            : "recorded_source"
      ] = fault === "unknown-producer" ? "unrecognised producer" : false;
      entry["output_digest"] = await canonicalDigest("studio.workflow-output.v1", output);
      const corrupted = writeJson(wire);
      await expect(
        runLocalWorkflow({
          ...options,
          sourceJson: corrupted,
          save: async () => {
            throw new Error("Changed original output must never be saved");
          },
        }),
      ).rejects.toThrow(
        kind === "preview"
          ? "original cached preview differs"
          : "original cached projection differs",
      );
      expect(BuiltKernelWorker.started).toBe(workers);
      expect(writeJson(readJson(corrupted))).toBe(corrupted);
    }
    expect(canonicalBytes("original-workflow-journal.v1", original.journal)).toEqual(before);
  },
  30000,
);
