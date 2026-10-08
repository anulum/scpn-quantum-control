// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public typed workflow graph contract tests

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { canonicalBytes, readJson, writeJson } from "../../shared/contracts";
import {
  parseWorkflow,
  maxWorkflowBytes,
  topologicalOrder,
  validatePortValue,
  workflowDocument,
  WorkflowRefusal,
} from "./workflowModel";
import type { WorkflowOutput, WorkflowStage } from "./workflowModel";

function fixture(): Record<string, unknown> {
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  return corpus["workflow"] as Record<string, unknown>;
}

/** Decode the original shared metadata oracle without changing integer or binary64 types. */
function numericTokens(): Record<string, unknown> {
  const corpus = readJson(
    readFileSync("../tests/data/studio_workspace/transport.json", "utf8"),
  ) as Record<string, unknown>;
  const cases = corpus["cases"] as Record<string, unknown>[];
  const original = cases.find((item) => item["id"] === "typed_numeric_tokens");
  if (original === undefined || typeof original["input_json"] !== "string")
    throw new Error("Original cross-language numeric-token fixture is unavailable");
  return readJson(original["input_json"]) as Record<string, unknown>;
}

describe("original workflow graph admission", () => {
  it("preserves shared exact JSON, source order and immutable parameters", () => {
    const original = fixture();
    const before = canonicalBytes("fixture.v1", original);
    const definition = parseWorkflow(original);
    expect(topologicalOrder(definition)).toEqual(["source", "trace"]);
    expect(canonicalBytes("fixture.v1", workflowDocument(definition))).toEqual(before);
    const recovered = parseWorkflow(readJson(writeJson(workflowDocument(definition))));
    expect(canonicalBytes("fixture.v1", workflowDocument(recovered))).toEqual(before);
    (original["body"] as Record<string, unknown>)["workflow_id"] = "changed";
    expect(definition.workflow_id).toBe("compile-sweep");
    expect(definition.stages).toHaveLength(2);
    const trace = definition.stages[0] as WorkflowStage;
    expect(Reflect.set(trace.parameters, "compiler_trace", false)).toBe(false);
  });

  it.each([
    "cycle",
    "unit",
    "schema",
    "dtype",
    "shape",
    "dangling",
    "duplicate",
    "approval",
    "version",
  ])("refuses %s before producing an executable graph", (fault) => {
    const original = fixture();
    const body = original["body"] as Record<string, unknown>;
    const stages = body["stages"] as [Record<string, unknown>, Record<string, unknown>];
    expect(stages).toHaveLength(2);
    const [incoming] = stages[0]["inputs"] as [Record<string, unknown>];
    const port = incoming["type"] as Record<string, unknown>;
    if (fault === "cycle") stages[1]["depends_on"] = ["trace"];
    else if (["unit", "schema", "dtype", "shape"].includes(fault))
      port[fault] = { unit: "rad", schema: "other-source.v1", dtype: "json", shape: [1n] }[
        fault as "unit" | "schema" | "dtype" | "shape"
      ];
    else if (fault === "dangling") incoming["source_stage"] = "missing";
    else if (fault === "duplicate") stages.push(stages[0]);
    else if (fault === "approval") stages[0]["approved"] = true;
    else original["schema"] = "experiment_workflow.v2";
    expect(() => parseWorkflow(original)).toThrow();
  });

  it.each([
    ["body-array", "workflow body: object required"],
    ["empty-graph", "nonempty unique workflow stages"],
    ["missing-control-parent", "workflow dependency is absent"],
    ["adapter", "unsupported workflow adapter or verb"],
    ["local-verb", "local classical adapter does not implement"],
    ["backend-control-text", "backend: bounded nonempty text"],
    ["port-dtype", "unsupported port schema or dtype"],
    ["json-port-shape", "unsupported port shape"],
    ["port-dimension", "port dimension: integer between"],
    ["port-shape-product", "unsupported port shape"],
    ["output-path", "nonempty original output path"],
    ["duplicate-input", "duplicate stage port or dependency"],
    ["constant-input", "a bound input cannot also have a constant"],
    ["input-parameter-key", "original ASCII parameter key"],
    ["empty-axis", "nonempty unique sweep coordinates"],
    ["duplicate-axis", "nonempty unique sweep coordinates"],
    ["bound-axis", "sweep target is absent or already bound"],
    ["empty-seeds", "unique canonical uint64 seeds"],
    ["overflow-seed", "unique canonical uint64 seeds"],
    ["axis-seed-binding", "seed target is absent or already has a value"],
    ["missing-seed-target", "seed target is absent or already has a value"],
    ["zero-budget", "evaluation budget: integer between"],
    ["extensions-array", "extensions: object required"],
    ["stage-identifier", "lowercase ASCII identifier"],
    ["definition-byte-bound", "workflow definition exceeds byte bound"],
    ["stages-object", "stages: bounded array"],
    ["stage-count", "stages: bounded array"],
    ["constant-seed-target", "seed target is absent or already has a value"],
    ["bound-seed-target", "seed target is absent or already has a value"],
    ["insufficient-budget", "sweep exceeds cell or stage evaluation budget"],
    ["edge-count", "workflow edge bound exceeded"],
  ])("refuses malformed %s while retaining the original caller document", (fault, reason) => {
    const original = fixture();
    const body = original["body"] as Record<string, unknown>;
    const stages = body["stages"] as [Record<string, unknown>, Record<string, unknown>];
    const inputs = stages[0]["inputs"] as [Record<string, unknown>];
    const outputs = stages[1]["outputs"] as [Record<string, unknown>];
    const port = inputs[0]["type"] as Record<string, unknown>;
    const sweep = body["sweep"] as Record<string, unknown>;
    const axes = sweep["axes"] as [Record<string, unknown>, Record<string, unknown>];
    if (fault === "body-array") original["body"] = [];
    else if (fault === "empty-graph") body["stages"] = [];
    else if (fault === "missing-control-parent") stages[0]["depends_on"] = ["missing"];
    else if (fault === "adapter") stages[0]["adapter"] = "provider";
    else if (fault === "local-verb") {
      stages[0]["adapter"] = "local-kuramoto";
      stages[0]["verb"] = "compile";
    } else if (fault === "backend-control-text") stages[0]["backend"] = "invalid\nbackend";
    else if (fault === "port-dtype") port["dtype"] = "complex";
    else if (fault === "json-port-shape") {
      port["dtype"] = "json";
      port["shape"] = [1n];
    } else if (fault === "port-dimension") port["shape"] = [4097n];
    else if (fault === "port-shape-product") port["shape"] = [64n, 65n];
    else if (fault === "output-path") outputs[0]["path"] = [];
    else if (fault === "duplicate-input") inputs.push(inputs[0]);
    else if (fault === "constant-input")
      (stages[0]["parameters"] as Record<string, unknown>)[inputs[0]["parameter"] as string] =
        "already-bound";
    else if (fault === "input-parameter-key") inputs[0]["parameter"] = "invalid-key";
    else if (fault === "empty-axis") axes[0]["values"] = [];
    else if (fault === "duplicate-axis") axes.push(axes[0]);
    else if (fault === "bound-axis") {
      axes[0]["stage_id"] = "trace";
      axes[0]["parameter"] = inputs[0]["parameter"];
    } else if (fault === "empty-seeds") sweep["seeds"] = [];
    else if (fault === "overflow-seed") sweep["seeds"] = ["18446744073709551616"];
    else if (fault === "axis-seed-binding")
      sweep["seed_binding"] = {
        stage_id: axes[0]["stage_id"],
        parameter: axes[0]["parameter"],
      };
    else if (fault === "missing-seed-target")
      sweep["seed_binding"] = { stage_id: "missing", parameter: "seed" };
    else if (fault === "zero-budget") sweep["evaluation_budget"] = 0n;
    else if (fault === "extensions-array") original["extensions"] = [];
    else if (fault === "stage-identifier") stages[0]["id"] = "Mixed";
    else if (fault === "stages-object") body["stages"] = {};
    else if (fault === "stage-count") body["stages"] = Array.from({ length: 65 }, () => stages[1]);
    else if (fault === "constant-seed-target")
      sweep["seed_binding"] = { stage_id: "trace", parameter: "compiler_trace" };
    else if (fault === "bound-seed-target")
      sweep["seed_binding"] = { stage_id: "trace", parameter: "program_source" };
    else if (fault === "insufficient-budget") sweep["evaluation_budget"] = 1n;
    else if (fault === "edge-count")
      body["stages"] = Array.from({ length: 32 }, (_, index) => ({
        ...stages[1],
        id: `stage-${index}`,
        depends_on: Array.from({ length: 5 }, (_, parent) => `stage-${parent}`),
      }));
    else original["extensions"] = { source_text: "a".repeat(maxWorkflowBytes) };
    const before = writeJson(original);
    expect(() => parseWorkflow(original)).toThrow(WorkflowRefusal);
    expect(() => parseWorkflow(original)).toThrow(reason);
    expect(writeJson(original)).toBe(before);
  });

  it("binds explicit seeds to the original compiler optimisation setting", () => {
    const original = fixture();
    const body = original["body"] as Record<string, unknown>;
    const sweep = body["sweep"] as Record<string, unknown>;
    const axes = sweep["axes"] as [Record<string, unknown>, Record<string, unknown>];
    sweep["axes"] = [axes[0]];
    sweep["seeds"] = ["0", "1", "2"];
    sweep["seed_binding"] = { stage_id: "trace", parameter: "optimisation_level" };
    const before = writeJson(original);
    const admitted = parseWorkflow(original);
    expect(admitted.sweep.seed_binding).toEqual({
      stage_id: "trace",
      parameter: "optimisation_level",
    });
    expect(writeJson(workflowDocument(admitted))).toBe(before);
  });

  it("retains the actual compiler output as independent immutable JSON port data", () => {
    const corpus = readJson(
      readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
    ) as Record<string, unknown>;
    const checkpoint = corpus["original_checkpoint"] as Record<string, unknown>;
    const body = checkpoint["body"] as Record<string, unknown>;
    const [entry] = body["entries"] as [Record<string, unknown>];
    const output = entry["output"] as Record<string, unknown>;
    const admitted = validatePortValue(
      { schema: "studio.workflow-output.v1", dtype: "json", shape: [], unit: "1" },
      output,
    );
    expect(admitted).toEqual(output);
    expect(admitted).not.toBe(output);
    expect(Reflect.set(admitted as object, "digest", "changed")).toBe(false);
    const result = (admitted as Record<string, unknown>)["result"] as Record<string, unknown>;
    expect((result["outputs"] as Record<string, unknown>)["execution_status"]).toBe(
      "emitted_not_executed",
    );
  });

  it("roundtrips opaque large integers, binary64 and signed zero", () => {
    const original = fixture();
    original["extensions"] = { integer: 9007199254740993n, float: 1, zero: -0 };
    const first = parseWorkflow(original);
    const second = parseWorkflow(readJson(writeJson(workflowDocument(first))));
    expect(canonicalBytes("fixture.v1", workflowDocument(second))).toEqual(
      canonicalBytes("fixture.v1", workflowDocument(first)),
    );
    expect(second.extensions["integer"]).toBe(9007199254740993n);
    expect(Object.is(second.extensions["zero"], -0)).toBe(true);
  });

  it("checks actual original port values without converting types", () => {
    const definition = parseWorkflow(fixture());
    expect(definition.stages).toHaveLength(2);
    const source = definition.stages[1] as WorkflowStage;
    expect(source.outputs).toHaveLength(1);
    const type = (source.outputs[0] as WorkflowOutput).type;
    const original = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0];';
    expect(validatePortValue(type, original)).toBe(original);
    expect(() => validatePortValue(type, 1n)).toThrow();
  });

  it("preserves original binary64 scalar and vector bits and refuses incompatible shapes", () => {
    const tokens = numericTokens();
    const type = {
      schema: "experiment_revision.v1",
      dtype: "float64",
      shape: [],
      unit: "1",
    } as const;
    expect(validatePortValue(type, tokens["f"])).toBe(tokens["f"]);
    const vector = validatePortValue({ ...type, shape: [2n] }, [
      tokens["f"],
      tokens["z"],
    ]) as readonly unknown[];
    expect(vector[0]).toBe(tokens["f"]);
    expect(Object.is(vector[1], -0)).toBe(true);
    expect(Object.isFrozen(vector)).toBe(true);
    expect(() => validatePortValue({ ...type, shape: [2n] }, tokens["f"])).toThrow(
      "original port shape differs",
    );
    expect(() => validatePortValue({ ...type, shape: [2n] }, [tokens["f"]])).toThrow(
      "original port shape differs",
    );
    expect(() => validatePortValue(type, tokens["i"])).toThrow("original port dtype differs");
    expect(() => validatePortValue(type, Number.NaN)).toThrow();
    expect(() => validatePortValue(type, Number.POSITIVE_INFINITY)).toThrow();
  });

  it("preserves compiler booleans and exact signed/unsigned integers without numeric coercion", () => {
    const tokens = numericTokens();
    const graph = parseWorkflow(fixture());
    const traced = graph.stages[0]?.parameters["compiler_trace"];
    expect(traced).toBe(true);
    const type = { schema: "experiment_revision.v1", shape: [], unit: "1" } as const;
    expect(validatePortValue({ ...type, dtype: "bool" }, traced)).toBe(true);
    expect(validatePortValue({ ...type, dtype: "int64" }, tokens["i"])).toBe(1n);
    expect(validatePortValue({ ...type, dtype: "uint64" }, tokens["big"])).toBe(9007199254740993n);
    expect(() => validatePortValue({ ...type, dtype: "bool" }, tokens["i"])).toThrow(
      "original port dtype differs",
    );
    for (const value of [traced, -(2n ** 63n) - 1n, 2n ** 63n])
      expect(() => validatePortValue({ ...type, dtype: "int64" }, value)).toThrow(
        "original port dtype differs",
      );
    for (const value of [tokens["f"], -1n, 2n ** 64n])
      expect(() => validatePortValue({ ...type, dtype: "uint64" }, value)).toThrow(
        "original port dtype differs",
      );
  });

  it("refuses a caller-created duplicate instead of silently dropping a stage", () => {
    const definition = parseWorkflow(fixture());
    expect(definition.stages).toHaveLength(2);
    const first = definition.stages[0] as WorkflowStage;
    expect(() => topologicalOrder({ ...definition, stages: [first, first] })).toThrow("unique");
  });

  it("preserves the original executive K_nm parameter in sweep bindings", () => {
    const source = fixture();
    const body = source["body"] as Record<string, unknown>;
    body["stages"] = [
      {
        id: "network",
        adapter: "executive",
        verb: "compile",
        backend: "python",
        parameters: {
          K_nm: [
            [0, 0.1],
            [0.1, 0],
          ],
          omega: [0, 0],
          time: 0.1,
          trotter_steps: 1n,
          trotter_order: 1n,
        },
        inputs: [],
        outputs: [],
        depends_on: [],
      },
    ];
    body["sweep"] = {
      axes: [
        {
          stage_id: "network",
          parameter: "K_nm",
          values: [
            [
              [0, 0.1],
              [0.1, 0],
            ],
            [
              [0, 0.2],
              [0.2, 0],
            ],
          ],
        },
      ],
      seeds: ["0"],
      seed_binding: null,
      evaluation_budget: 2n,
    };
    const definition = parseWorkflow(source);
    expect(definition.sweep.axes.at(0)?.parameter).toBe("K_nm");
    expect(Object.hasOwn((definition.stages[0] as WorkflowStage).parameters, "K_nm")).toBe(true);
  });
});
