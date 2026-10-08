// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — genuine workflow native worker tests

import { webcrypto } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { BuiltKernelWorker } from "../../../test-support/kernelWorker";
import { instantiateKuramoto } from "../../panel/kuramoto";
import { canonicalBytes, canonicalDigest, readJson, writeJson } from "../../shared/contracts";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { runLocalWorkflow } from "./workflowExecution";
import type { WorkflowExecutionResult, WorkflowStageProgress } from "./workflowExecution";
import type { WorkflowAttempt } from "./workflowJournal";
import { readWorkflowArchive } from "./workflowArchive";
import { parseWorkflow, workflowDocument } from "./workflowModel";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
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

function graph() {
  const type = { schema: "studio.workflow-run-reference.v1", dtype: "json", shape: [], unit: "1" };
  return parseWorkflow({
    schema: "experiment_workflow.v1",
    body: {
      workflow_id: "classical-native-sweep",
      stages: [
        {
          id: "validate",
          adapter: "local-kuramoto",
          verb: "validate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [],
          depends_on: [],
        },
        {
          id: "simulate",
          adapter: "local-kuramoto",
          verb: "simulate",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [],
          outputs: [{ name: "run", path: ["run_ref"], type }],
          depends_on: ["validate"],
        },
        {
          id: "analyse",
          adapter: "local-kuramoto",
          verb: "analyse",
          backend: "shipped-kuramoto-wasm-float64",
          parameters: {},
          inputs: [{ parameter: "run_ref", source_stage: "simulate", source_port: "run", type }],
          outputs: [],
          depends_on: [],
        },
      ],
      sweep: {
        axes: [
          { stage_id: "simulate", parameter: "coupling", values: [1.2, 1.4] },
          { stage_id: "simulate", parameter: "dt", values: [0.01, 0.02, 0.04] },
        ],
        seeds: ["0"],
        seed_binding: null,
        evaluation_budget: 18n,
      },
    },
    extensions: {
      source: "Original classical Kuramoto native fixture, no quantum-model equivalence claim",
    },
  });
}
function graphStage(id: string) {
  const stage = graph().stages.find((item) => item.id === id);
  if (stage === undefined) throw new Error("Original native fixture stage is absent");
  return stage;
}
function producedOutput(entry: WorkflowAttempt | undefined): Record<string, unknown> {
  const value = entry?.output;
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("Actual original workflow output object is absent");
  return value as Record<string, unknown>;
}

async function original() {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "native workflow source",
    localExperimentCodecs,
    "native worker test",
  );
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  return { kernel, archive, baseRevision: source.revisionHash };
}

let fixture: Awaited<ReturnType<typeof original>> & {
  readonly result: WorkflowExecutionResult;
  readonly started: number;
  readonly saves: number;
  readonly progress: readonly WorkflowStageProgress[];
};
beforeAll(async () => {
  vi.stubGlobal("crypto", webcrypto);
  try {
    const initial = await original();
    let current = initial.archive;
    const started = BuiltKernelWorker.started;
    let saves = 0;
    const progress: WorkflowStageProgress[] = [];
    const startedAt = performance.now();
    const timings: {
      milliseconds: number;
      entries: number;
      evaluations: string;
      archiveCharacters: number;
      nativeWorkers: number;
    }[] = [];
    const result = await runLocalWorkflow({
      sourceJson: initial.archive.json,
      baseRevision: initial.baseRevision,
      definition: graph(),
      kernel: initial.kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      workerFactory: () => new BuiltKernelWorker(),
      onProgress: (stage) => {
        progress.push(stage);
      },
      onCheckpoint: (checkpoint) => {
        timings.push({
          milliseconds: performance.now() - startedAt,
          entries: checkpoint.journal.entries.length,
          evaluations: checkpoint.journal.evaluations.toString(),
          archiveCharacters: checkpoint.archive.json.length,
          nativeWorkers: BuiltKernelWorker.started - started,
        });
        const destination = process.env["STUDIO_WORKFLOW_TIMINGS"];
        if (destination !== undefined) writeFileSync(destination, JSON.stringify(timings));
      },
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
        saves++;
      },
    });
    fixture = { ...initial, result, started, saves, progress };
  } finally {
    vi.unstubAllGlobals();
  }
}, 30000);

it("executes six actual native cells and preserves eighteen original stage outcomes", async () => {
  const { result, baseRevision, started, saves } = fixture;
  expect(result.journal.state, writeJson(result.journal.entries)).toBe("complete");
  expect(result.journal.evaluations).toBe(18n);
  expect(result.journal.entries).toHaveLength(18);
  expect(result.disposalConfirmed).toBe(true);
  expect(BuiltKernelWorker.started - started).toBe(6);
  expect(fixture.progress.map((stage) => stage.settledStages)).toEqual(
    Array.from({ length: 18 }, (_, index) => index + 1),
  );
  expect(
    fixture.progress.every(
      (stage) =>
        stage.totalStages === 18 &&
        !stage.reused &&
        stage.workflowDigest === result.journal.workflow_digest,
    ),
  ).toBe(true);
  expect(fixture.progress.map((stage) => [stage.cellId, stage.stageId])).toEqual(
    result.journal.entries.map((entry) => [entry.cell_id, entry.stage_id]),
  );
  const admitted = await readWorkflowArchive(result.archive.json, localExperimentCodecs);
  const records = Object.values(admitted.source.documents).filter(
    (document) => document.schema === "local_run_record.v1",
  );
  expect(records).toHaveLength(6);
  const source = await readLocalExperiment(
    result.archive.json,
    localExperimentCodecs,
    baseRevision,
  );
  expect(source.request.coupling).toBe(1.4);
  expect(source.request.dt).toBe(0.01);
  expect(saves).toBe(37);
});

it("resumes the six-cell original archive without allocating another worker or duplicating records", async () => {
  const { result, baseRevision, kernel } = fixture;
  const started = BuiltKernelWorker.started;
  let current = result.archive;
  const progress: WorkflowStageProgress[] = [];
  const resumed = await runLocalWorkflow({
    sourceJson: result.archive.json,
    onProgress: (stage) => {
      progress.push(stage);
    },
    baseRevision,
    definition: graph(),
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    workerFactory: () => new BuiltKernelWorker(),
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  expect(resumed.journal.state).toBe("complete");
  expect(progress.map((stage) => stage.settledStages)).toEqual(
    Array.from({ length: 18 }, (_, index) => index + 1),
  );
  expect(progress.every((stage) => stage.reused && stage.totalStages === 18)).toBe(true);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(canonicalBytes("original-native-journal.v1", resumed.journal)).toEqual(
    canonicalBytes("original-native-journal.v1", result.journal),
  );
  expect(
    (await readWorkflowArchive(resumed.archive.json, localExperimentCodecs)).source.members,
  ).toEqual((await readWorkflowArchive(result.archive.json, localExperimentCodecs)).source.members);
}, 30000);

it("allows a scheduled user cancellation between revalidated cached stages", async () => {
  const { result, baseRevision, kernel } = fixture;
  const started = BuiltKernelWorker.started;
  const cancellation = new AbortController();
  const progress: WorkflowStageProgress[] = [];
  let timer: ReturnType<typeof setTimeout> | undefined;
  let current = result.archive;
  try {
    const resumed = await runLocalWorkflow({
      sourceJson: result.archive.json,
      baseRevision,
      definition: graph(),
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: cancellation.signal,
      workerFactory: () => new BuiltKernelWorker(),
      onProgress: (stage) => {
        progress.push(stage);
        if (progress.length === 1) timer = setTimeout(() => cancellation.abort(), 0);
      },
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
    });
    expect(resumed.journal.state).toBe("cancelled");
    expect(progress).toHaveLength(1);
    expect(progress[0]?.reused).toBe(true);
    expect(resumed.journal.entries).toEqual(result.journal.entries);
    expect(resumed.journal.evaluations).toBe(18n);
    expect(resumed.disposalConfirmed).toBe(true);
    expect(BuiltKernelWorker.started).toBe(started);
  } finally {
    clearTimeout(timer);
  }
}, 30000);

it("retains actual native cancellation only after observed worker disposal", async () => {
  const { kernel, archive, baseRevision } = await original();
  let current = archive;
  const abort = new AbortController();
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition: graph(),
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: abort.signal,
    workerFactory: () => new BuiltKernelWorker(),
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
    onWorker: (handle) => {
      if (handle !== null) abort.abort();
    },
  });
  expect(result.journal.state).toBe("cancelled");
  expect(result.disposalConfirmed).toBe(true);
  expect(result.journal.entries.map((entry) => entry.status)).toEqual(["complete", "cancelled"]);
  expect(result.journal.entries[1]?.output).not.toBeNull();
  const admitted = await readWorkflowArchive(result.archive.json, localExperimentCodecs);
  const record = Object.values(admitted.source.documents).find(
    (document) => document.schema === "local_run_record.v1",
  );
  const events = record?.body["events"] as readonly {
    readonly kind: string;
    readonly payload: Record<string, unknown>;
  }[];
  expect(events.at(-1)?.kind).toBe("cancelled");
  expect(events.at(-1)?.payload["disposed"]).toBe(true);
});

it("blocks dependent stages when the original source-domain validation fails", async () => {
  const { kernel, archive, baseRevision } = await original();
  const document = readJson(writeJson(workflowDocument(graph()))) as Record<string, unknown>,
    body = document["body"] as Record<string, unknown>;
  const stages = body["stages"] as Record<string, unknown>[];
  (stages[0] as Record<string, unknown>)["parameters"] = { steps: 0n };
  const definition = parseWorkflow(document);
  let current = archive;
  const started = BuiltKernelWorker.started;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
    workerFactory: () => new BuiltKernelWorker(),
  });
  expect(result.journal.state).toBe("partial");
  expect(result.journal.evaluations).toBe(6n);
  expect(result.journal.entries.map((entry) => entry.status)).toEqual(
    Array.from({ length: 6 }, () => ["failed", "blocked", "blocked"]).flat(),
  );
  expect(BuiltKernelWorker.started).toBe(started);
});

it("refuses unsupported executive graphs before any native allocation or save", async () => {
  const { kernel, archive, baseRevision } = await original();
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  let saved = false;
  const started = BuiltKernelWorker.started;
  await expect(
    runLocalWorkflow({
      sourceJson: archive.json,
      baseRevision,
      definition: parseWorkflow(corpus["workflow"]),
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      save: async () => {
        saved = true;
      },
    }),
  ).rejects.toThrow("local CLI");
  expect(saved).toBe(false);
  expect(BuiltKernelWorker.started).toBe(started);
});

it.each([
  ["vector override", { omega: [0.3, 0.4], steps: 4n }, "complete"],
  ["vector shape", { omega: [0.3] }, "failed"],
  ["vector scalar", { omega: 0.3 }, "failed"],
  ["integer dtype", { steps: 4.0 }, "failed"],
  ["unknown specification", { unknown_parameter: 1.0 }, "failed"],
  ["invalid complete binding", { parameters: null }, "failed"],
] as const)("keeps the original source contract for %s", async (_name, overrides, expected) => {
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    stages: [{ ...graphStage("validate"), parameters: overrides }],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = archive;
  const started = BuiltKernelWorker.started;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  expect(result.journal.entries[0]?.status).toBe(expected);
  expect(result.journal.state).toBe(expected === "complete" ? "complete" : "partial");
  expect(result.journal.evaluations).toBe(1n);
  expect(BuiltKernelWorker.started).toBe(started);
  expect((await readLocalExperiment(archive.json, localExperimentCodecs)).request.omega).toEqual([
    0.2, 0.2,
  ]);
});

it("feeds actual source-owned parameter bindings through a typed workflow output port", async () => {
  const { kernel, archive, baseRevision } = await original();
  const type = {
    schema: "studio.parameter-bindings.v1",
    dtype: "json" as const,
    shape: [],
    unit: "1",
  };
  const first = {
    ...graphStage("validate"),
    parameters: {},
    outputs: [{ name: "parameters", path: ["parameters"], type }],
  };
  const second = {
    ...first,
    id: "bound-child",
    outputs: [],
    inputs: [{ parameter: "parameters", source_stage: first.id, source_port: "parameters", type }],
  };
  const definition = {
    ...graph(),
    stages: [first, second],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
  };
  let current = archive;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  expect(result.journal.state).toBe("complete");
  expect(result.journal.entries.map((entry) => entry.status)).toEqual(["complete", "complete"]);
  expect(producedOutput(result.journal.entries[1])["parameters"]).toEqual(
    producedOutput(result.journal.entries[0])["parameters"],
  );
});

it.each(["missing", "scalar-parent"] as const)(
  "records failure of the actual %s produced port and blocks its original dependent",
  async (fault) => {
    const { kernel, archive, baseRevision } = await original();
    const type = {
      schema: "studio.parameter-bindings.v1",
      dtype: "json" as const,
      shape: [],
      unit: "1",
    };
    const first = {
      ...graphStage("validate"),
      parameters: {},
      outputs: [
        {
          name: "parameters",
          path: fault === "missing" ? ["absent"] : ["preview_only", "absent"],
          type,
        },
      ],
    };
    const second = {
      ...first,
      id: "bound-child",
      outputs: [],
      inputs: [
        { parameter: "parameters", source_stage: first.id, source_port: "parameters", type },
      ],
    };
    const definition = {
      ...graph(),
      stages: [first, second],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
    };
    let current = archive;
    const started = BuiltKernelWorker.started;
    const result = await runLocalWorkflow({
      sourceJson: archive.json,
      baseRevision,
      definition,
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
    });
    expect(result.journal.entries[0]?.reason).toContain(
      fault === "missing" ? "output port is absent" : "original workflow object required",
    );
    const retained = (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]
      ?.journal;
    expect(retained?.state).toBe("partial");
    expect(retained?.entries.map((entry) => entry.status)).toEqual(["failed", "blocked"]);
    expect(retained?.evaluations).toBe(1n);
    expect(BuiltKernelWorker.started).toBe(started);
  },
);

it.each(["validate", "simulate", "analyse", "refused-plan"] as const)(
  "refuses a changed recorded %s output even after its declaration digest is updated",
  async (stage) => {
    const { result, kernel, baseRevision } = fixture;
    const wire = readJson(result.archive.json) as Record<string, unknown>;
    const extension = (
      (wire["manifest"] as Record<string, unknown>)["extensions"] as Record<string, unknown>
    )["experiment_workflows"] as Record<string, unknown>;
    const item = (extension["items"] as Record<string, unknown>[])[0] as Record<string, unknown>;
    const journal = (item["journal"] as Record<string, unknown>)["body"] as Record<string, unknown>;
    const entries = journal["entries"] as Record<string, unknown>[];
    const sid = stage === "refused-plan" ? "validate" : stage;
    const first = entries.find((entry) => entry["stage_id"] === sid) as Record<string, unknown>;
    const output = first["output"] as Record<string, unknown>;
    if (stage === "refused-plan") {
      first["fingerprint"] = await canonicalDigest("studio.workflow-stage.v1", {
        cell_id: first["cell_id"],
        stage_id: sid,
        dependencies: first["dependencies"],
        source: journal["source_fingerprint"],
        runtime: journal["runtime_fingerprint"],
        parameters: {},
        plan_hash: null,
      });
    } else if (stage === "simulate") {
      const other = entries.find(
        (row) => row["stage_id"] === stage && row["cell_id"] !== first["cell_id"],
      );
      if (other === undefined) throw new Error("Other genuine native cell is absent");
      output["run_ref"] = (other["output"] as Record<string, unknown>)["run_ref"];
    } else output[stage === "validate" ? "preview_only" : "recorded_source"] = false;
    first["output_digest"] = await canonicalDigest("studio.workflow-output.v1", first["output"]);
    for (const entry of entries) {
      if (entry["cell_id"] === first["cell_id"]) {
        const dependencies = entry["dependencies"] as Record<string, unknown>;
        if (Object.hasOwn(dependencies, stage)) dependencies[stage] = first["output_digest"];
      }
    }
    const corrupted = writeJson(wire);
    const before = BuiltKernelWorker.started;
    let approvals = 0;
    const planOptions = {
      get deadlineMs() {
        return approvals++ === 0 ? 5000 : 0;
      },
    };
    await expect(
      runLocalWorkflow({
        sourceJson: corrupted,
        baseRevision,
        definition: graph(),
        kernel,
        ...(stage === "refused-plan" ? { planOptions } : {}),
        rawCodecs: localExperimentCodecs,
        signal: new AbortController().signal,
        save: async () => {
          throw new Error("Changed cached preview must never be saved");
        },
        workerFactory: () => new BuiltKernelWorker(),
      }),
    ).rejects.toThrow(
      stage === "refused-plan"
        ? "completed stage no longer has a current plan"
        : stage === "validate"
          ? "cached preview differs"
          : stage === "simulate"
            ? "cached run plan differs"
            : "cached projection differs",
    );
    expect(BuiltKernelWorker.started).toBe(before);
    expect(writeJson(readJson(corrupted))).toBe(corrupted);
  },
);

it.each(["null", "missing-field", "wrong-plan", "wrong-build", "wrong-revision"] as const)(
  "refuses an imported %s run reference without allocating a replacement",
  async (fault) => {
    const { result, kernel, baseRevision } = fixture;
    const simulations = result.journal.entries.filter((entry) => entry.stage_id === "simulate");
    const reference = producedOutput(simulations[0])["run_ref"] as Record<string, unknown>;
    const bad: Record<string, unknown> = { ...reference };
    if (fault === "missing-field") delete bad["build_fingerprint"];
    else if (fault === "wrong-plan") bad["plan_hash"] = "a".repeat(64);
    else if (fault === "wrong-build") bad["build_fingerprint"] = "a".repeat(64);
    else if (fault === "wrong-revision")
      bad["revision_hash"] = (
        producedOutput(simulations.at(-1))["run_ref"] as Record<string, unknown>
      )["revision_hash"];
    const stage = {
      ...graphStage("analyse"),
      id: "imported-analysis",
      parameters: { run_ref: fault === "null" ? null : bad },
      inputs: [],
      depends_on: [],
    };
    const definition = {
      ...graph(),
      workflow_id: `refused-reference-${fault}`,
      stages: [stage],
      sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
    };
    const started = BuiltKernelWorker.started;
    let current = result.archive;
    const refusal = await runLocalWorkflow({
      sourceJson: current.json,
      baseRevision,
      definition,
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      save: async (candidate, prior) => {
        expect(prior).toBe(current.json);
        current = candidate;
      },
    });
    expect(refusal.journal.state).toBe("partial");
    expect(refusal.journal.entries[0]?.status).toBe("failed");
    expect(BuiltKernelWorker.started).toBe(started);
    const retained = await readWorkflowArchive(current.json, localExperimentCodecs);
    expect(retained.workflows[0]?.journal).toEqual(result.journal);
    expect(
      Object.keys(retained.source.documents).filter(
        (hash) => retained.source.documents[hash]?.schema === "local_run_record.v1",
      ),
    ).toHaveLength(6);
  },
);

it("rejects extra analysis inputs before using a genuine indexed result", async () => {
  const { result, kernel, baseRevision } = fixture;
  const entry = result.journal.entries.find((row) => row.stage_id === "simulate");
  const reference = producedOutput(entry)["run_ref"];
  const stage = {
    ...graphStage("analyse"),
    id: "extra-analysis-input",
    parameters: { run_ref: reference, unexpected: 1n },
    inputs: [],
    depends_on: [],
  };
  const definition = {
    ...graph(),
    workflow_id: "extra-analysis-input",
    stages: [stage],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = result.archive;
  const refusal = await runLocalWorkflow({
    sourceJson: current.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate) => {
      current = candidate;
    },
  });
  expect(refusal.journal.state).toBe("partial");
  expect(refusal.journal.entries[0]?.status).toBe("failed");
});

it.each(["source_fingerprint", "runtime_fingerprint"] as const)(
  "preserves the original archive when imported %s no longer matches",
  async (field) => {
    const { result, kernel, baseRevision } = fixture;
    const wire = readJson(result.archive.json) as Record<string, unknown>;
    const extension = (
      (wire["manifest"] as Record<string, unknown>)["extensions"] as Record<string, unknown>
    )["experiment_workflows"] as Record<string, unknown>;
    const item = (extension["items"] as Record<string, unknown>[])[0] as Record<string, unknown>;
    ((item["journal"] as Record<string, unknown>)["body"] as Record<string, unknown>)[field] =
      "a".repeat(64);
    const changed = writeJson(wire);
    const started = BuiltKernelWorker.started;
    let saves = 0;
    await expect(
      runLocalWorkflow({
        sourceJson: changed,
        baseRevision,
        definition: graph(),
        kernel,
        rawCodecs: localExperimentCodecs,
        signal: new AbortController().signal,
        save: async () => {
          saves++;
        },
      }),
    ).rejects.toThrow("source or runtime changed");
    expect(saves).toBe(0);
    expect(BuiltKernelWorker.started).toBe(started);
    expect(writeJson(readJson(changed))).toBe(changed);
  },
);

it("retains an explicitly refused resource plan without allocating a worker", async () => {
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    stages: [graphStage("validate")],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = archive;
  const started = BuiltKernelWorker.started;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    planOptions: { memoryBudget: "0" },
    save: async (candidate) => {
      current = candidate;
    },
  });
  expect(result.journal.state).toBe("partial");
  expect(result.journal.entries[0]?.reason).toBe("original resource plan refused");
  expect(producedOutput(result.journal.entries[0])["preview_only"]).toBe(true);
  expect(BuiltKernelWorker.started).toBe(started);
  expect(
    (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(result.journal);
});

it("retains real worker events and refuses further traversal when the disposal receipt fails", async () => {
  class RefusedReceiptWorker extends BuiltKernelWorker {
    override async terminate(): Promise<void> {
      await super.terminate();
      throw new Error("native port receipt refused after real thread exit");
    }
  }
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    stages: [
      graphStage("simulate"),
      { ...graphStage("validate"), id: "after-native", depends_on: ["simulate"] },
    ].map((stage) => ({ ...stage, outputs: [], inputs: [] })),
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
  };
  definition.stages[0] = { ...graphStage("simulate"), depends_on: [], outputs: [], inputs: [] };
  let current = archive;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    workerFactory: () => new RefusedReceiptWorker(),
    save: async (candidate) => {
      current = candidate;
    },
  });
  expect(result.disposalConfirmed).toBe(false);
  expect(result.journal.state).toBe("partial");
  expect(result.journal.entries).toHaveLength(1);
  expect(result.journal.entries[0]?.status).toBe("failed");
  expect(producedOutput(result.journal.entries[0])["disposed"]).toBe(false);
  expect(BuiltKernelWorker.activeCount).toBe(0);
  expect(
    (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(result.journal);
});

it("refuses a saved graph rebound to another genuine immutable source revision", async () => {
  const { result, kernel, baseRevision } = fixture;
  const entry = result.journal.entries.find((row) => row.stage_id === "simulate");
  const ref = producedOutput(entry)["run_ref"] as Record<string, unknown>;
  const wire = readJson(result.archive.json) as Record<string, unknown>;
  const metadata = (
    (wire["manifest"] as Record<string, unknown>)["extensions"] as Record<string, unknown>
  )["experiment_workflows"] as Record<string, unknown>;
  const item = (metadata["items"] as Record<string, unknown>[])[0];
  if (item === undefined) throw new Error("Genuine saved workflow is absent");
  item["base_revision_hash"] = ref["revision_hash"];
  const prior = writeJson(wire);
  let saved = false;
  await expect(
    runLocalWorkflow({
      sourceJson: prior,
      baseRevision,
      definition: graph(),
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      save: async () => {
        saved = true;
      },
    }),
  ).rejects.toThrow("workflow baseline changed");
  expect(saved).toBe(false);
});

it("recovers a genuinely reserved checkpoint after its host loses the commit receipt", async () => {
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    workflow_id: "reserved-metadata-recovery",
    stages: [graphStage("validate")],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
  };
  let current = archive;
  await expect(
    runLocalWorkflow({
      sourceJson: archive.json,
      baseRevision,
      definition,
      kernel,
      rawCodecs: localExperimentCodecs,
      signal: new AbortController().signal,
      save: async (candidate) => {
        current = candidate;
        throw new Error("host commit receipt lost after reserved source was stored");
      },
    }),
  ).rejects.toThrow("commit receipt lost");
  const reserved = (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]
    ?.journal;
  expect(reserved?.entries[0]?.status).toBe("running");
  const result = await runLocalWorkflow({
    sourceJson: current.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  expect(result.journal.state).toBe("complete");
  expect(result.journal.entries.map((row) => row.status)).toEqual(["interrupted", "complete"]);
  expect(result.journal.evaluations).toBe(2n);
});

it("keeps cancellation between the actual reserved checkpoint and stage execution", async () => {
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    workflow_id: "reserved-cancel",
    stages: [graphStage("validate")],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  const signal = new AbortController();
  let current = archive;
  const started = BuiltKernelWorker.started;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: signal.signal,
    save: async (candidate) => {
      current = candidate;
      signal.abort();
    },
  });
  expect(result.journal.state).toBe("cancelled");
  expect(result.journal.entries[0]?.reason).toBe("Cancelled before original stage execution");
  expect(result.journal.entries[0]?.output).toBeNull();
  expect(BuiltKernelWorker.started).toBe(started);
  expect(
    (await readWorkflowArchive(current.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(result.journal);
});

it("keeps the prior blocked child once when the same genuine parent refusal is retried", async () => {
  const { kernel, archive, baseRevision } = await original();
  const parent = { ...graphStage("validate"), parameters: { steps: 0n } };
  const child = { ...graphStage("validate"), id: "blocked-child", depends_on: [parent.id] };
  const definition = {
    ...graph(),
    workflow_id: "repeated-parent-refusal",
    stages: [parent, child],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 2n },
  };
  let current = archive;
  const options = {
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate: typeof archive, prior: string) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  };
  const first = await runLocalWorkflow({ ...options, sourceJson: current.json });
  const next = await runLocalWorkflow({ ...options, sourceJson: current.json });
  expect(first.journal.entries.map((row) => row.status)).toEqual(["failed", "blocked"]);
  expect(next.journal.entries.map((row) => row.status)).toEqual(["failed", "blocked", "failed"]);
  expect(next.journal.entries[1]).toEqual(first.journal.entries[1]);
  expect(next.journal.evaluations).toBe(2n);
});

it("refuses a new current plan after the saved evaluation budget is exhausted", async () => {
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    workflow_id: "changed-plan-budget",
    stages: [graphStage("validate")],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = archive;
  const options = {
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate: typeof archive, prior: string) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  };
  const first = await runLocalWorkflow({ ...options, sourceJson: current.json });
  const refused = await runLocalWorkflow({
    ...options,
    sourceJson: current.json,
    planOptions: { deadlineMs: 4000 },
  });
  expect(first.journal.state).toBe("complete");
  expect(refused.journal.state).toBe("partial");
  expect(refused.journal.entries).toEqual(first.journal.entries);
  expect(refused.journal.evaluations).toBe(1n);
});

it("retains an explicit original refusal when the native browser Worker API is absent", async () => {
  expect(typeof Worker).toBe("undefined");
  const { kernel, archive, baseRevision } = await original();
  const definition = {
    ...graph(),
    workflow_id: "native-worker-unavailable",
    stages: [{ ...graphStage("simulate"), depends_on: [], outputs: [] }],
    sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 1n },
  };
  let current = archive;
  const started = BuiltKernelWorker.started;
  const result = await runLocalWorkflow({
    sourceJson: archive.json,
    baseRevision,
    definition,
    kernel,
    rawCodecs: localExperimentCodecs,
    signal: new AbortController().signal,
    save: async (candidate, prior) => {
      expect(prior).toBe(current.json);
      current = candidate;
    },
  });
  expect(result.disposalConfirmed).toBe(true);
  expect(result.journal.state).toBe("partial");
  expect(result.journal.entries[0]?.status).toBe("failed");
  expect(result.journal.entries[0]?.reason).toContain("Worker");
  expect(producedOutput(result.journal.entries[0])["outcome"]).toBe("refused");
  expect(BuiltKernelWorker.started).toBe(started);
  const retained = await readWorkflowArchive(current.json, localExperimentCodecs);
  expect(retained.workflows[0]?.journal).toEqual(result.journal);
  expect(
    Object.values(retained.source.documents).filter(
      (item) => item.schema === "local_run_record.v1",
    ),
  ).toHaveLength(1);
});
