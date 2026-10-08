// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable typed experiment workflow admission

import { canonicalBytes, readJson, writeJson } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";

/** Exact additive graph version; no existing workspace format is replaced. */
export const workflowSchema = "experiment_workflow.v1";
/** Maximum encoded definition size before graph admission. */
export const maxWorkflowBytes = 16 * 1024 * 1024;
/** Maximum stages before graph/plan allocation. */
export const maxWorkflowStages = 64;
/** Maximum Cartesian coordinates including seed identities. */
export const maxWorkflowCells = 256;
/** Total stage-evaluation ceiling; not a compute-duration promise. */
export const maxWorkflowEvaluations = 4096;

/** Typed graph or original runtime value refused without executing or saving. */
export class WorkflowRefusal extends Error {}
/** Original source scalar dtype; no implicit conversions between them. */
export type WorkflowDtype = "json" | "float64" | "int64" | "uint64" | "bool" | "utf8";
/** Explicitly distinct original executive/quantum and local/classical adapters. */
export type WorkflowAdapter = "executive" | "local-kuramoto";

/** Exact declared format of one unchanged producer datum; not scientific equivalence. */
export interface WorkflowPortType {
  /** Explicit original source-format identity. */ readonly schema: string;
  /** Scalar type, or one opaque JSON datum. */ readonly dtype: WorkflowDtype;
  /** Exact nested array shape; empty means scalar/opaque datum. */ readonly shape: readonly bigint[];
  /** Exact original unit, with no automatic conversion. */ readonly unit: string;
}
/** Bind one typed original output to a handler parameter. */
export interface WorkflowInput {
  /** Original target parameter; cannot also be constant or swept. */ readonly parameter: string;
  /** Existing producer stage. */ readonly source_stage: string;
  /** Existing original producer output name. */ readonly source_port: string;
  /** Exact consumer format, dtype, shape and unit. */ readonly type: WorkflowPortType;
}
/** Select unchanged original output with an object-key path. */
export interface WorkflowOutput {
  /** Unique stage-local output name. */ readonly name: string;
  /** Original object keys; no expression or transformation is evaluated. */ readonly path: readonly string[];
  /** Exact declared original output type. */ readonly type: WorkflowPortType;
}
/** One original operation with its source parameters and dependency bindings. */
export interface WorkflowStage {
  /** Unique stable ASCII identity. */ readonly id: string;
  /** Explicit model/runtime owner; adapters cannot be silently substituted. */ readonly adapter: WorkflowAdapter;
  /** Original operation; actual availability is checked by its owner. */ readonly verb: string;
  /** Explicit declared backend; no fallback is inferred. */ readonly backend: string;
  /** Immutable original constant values, preserving integer/float identity. */ readonly parameters: Readonly<
    Record<string, unknown>
  >;
  /** Typed original producer bindings. */ readonly inputs: readonly WorkflowInput[];
  /** Named original output selections. */ readonly outputs: readonly WorkflowOutput[];
  /** Explicit control edges in addition to data edges. */ readonly depends_on: readonly string[];
}
/** One ordered original parameter grid; no random or expression-based values. */
export interface WorkflowAxis {
  /** Original target stage. */ readonly stage_id: string;
  /** Original parameter key. */ readonly parameter: string;
  /** Exact unique ordered values. */ readonly values: readonly unknown[];
}
/** Exact seed identities and bounded total evaluation count. */
export interface WorkflowSweep {
  /** Last-axis-fastest Cartesian dimensions. */ readonly axes: readonly WorkflowAxis[];
  /** Canonical uint64 decimal seed identities. */ readonly seeds: readonly string[];
  /** Explicit original seed parameter, or null when the model does not consume it. */ readonly seed_binding: {
    /** Original stage. */ readonly stage_id: string;
    /** Original parameter. */ readonly parameter: string;
  } | null;
  /** Total permitted stage evaluations; original runtime limits also apply. */ readonly evaluation_budget: bigint;
}
/** Completely admitted immutable graph; no execution or provider authority granted. */
export interface WorkflowDefinition {
  /** Stable original workflow identity. */ readonly workflow_id: string;
  /** Original stage order, distinct from execution order. */ readonly stages: readonly WorkflowStage[];
  /** Exact bounded parameter coordinates. */ readonly sweep: WorkflowSweep;
  /** Opaque metadata roundtrips; cannot install code or approve a job. */ readonly extensions: Readonly<
    Record<string, unknown>
  >;
}

const verbs = new Set([
  "compile",
  "simulate",
  "analyse",
  "validate",
  "benchmark",
  "replay",
  "differentiate",
  "mitigate",
  "execute",
]);
const dtypes: readonly string[] = ["json", "float64", "int64", "uint64", "bool", "utf8"];

function object(value: unknown, name: string, keys?: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new WorkflowRefusal(`${name}: object required`);
  const result = Object.fromEntries(dataEntries(value));
  if (
    keys !== undefined &&
    (Object.keys(result).length !== keys.length || keys.some((k) => !Object.hasOwn(result, k)))
  )
    throw new WorkflowRefusal(`${name}: incomplete or unsupported fields`);
  return result;
}
function array(value: unknown, name: string, maximum: number): readonly unknown[] {
  if (!Array.isArray(value) || value.length > maximum)
    throw new WorkflowRefusal(`${name}: bounded array required`);
  return value;
}
function text(value: unknown, name: string, maximum = 128): string {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    Array.from(value).length > maximum ||
    Array.from(value).some((c) => c.charCodeAt(0) < 32)
  )
    throw new WorkflowRefusal(`${name}: bounded nonempty text required`);
  return value;
}
function identifier(value: unknown, name: string): string {
  const result = text(value, name, 64);
  if (!/^[a-z][a-z0-9_-]{0,63}$/.test(result))
    throw new WorkflowRefusal(`${name}: lowercase ASCII identifier required`);
  return result;
}
function parameterKey(value: unknown, name: string): string {
  const result = text(value, name, 64);
  if (!/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(result))
    throw new WorkflowRefusal(`${name}: original ASCII parameter key required`);
  return result;
}
function integer(value: unknown, name: string, low: bigint, high: bigint): bigint {
  if (typeof value !== "bigint" || value < low || value > high)
    throw new WorkflowRefusal(`${name}: integer between ${low} and ${high} required`);
  return value;
}
function capture(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.map(capture));
  if (typeof value === "object" && value !== null)
    return Object.freeze(
      Object.fromEntries(dataEntries(value).map(([key, item]) => [key, capture(item)])),
    );
  return value;
}
function captureRecord(value: Record<string, unknown>): Readonly<Record<string, unknown>> {
  return Object.freeze(
    Object.fromEntries(Object.entries(value).map(([key, item]) => [key, capture(item)])),
  );
}
function port(value: unknown): WorkflowPortType {
  const p = object(value, "port type", ["schema", "dtype", "shape", "unit"]);
  const schema = text(p["schema"], "port schema"),
    dtype = text(p["dtype"], "port dtype");
  if (!/^[a-z][a-z0-9_.-]{0,127}$/.test(schema) || !dtypes.includes(dtype))
    throw new WorkflowRefusal("unsupported port schema or dtype");
  const shape = array(p["shape"], "port shape", 4).map((d) =>
    integer(d, "port dimension", 0n, 4096n),
  );
  if (shape.reduce((size, d) => size * d, 1n) > 4096n || (dtype === "json" && shape.length !== 0))
    throw new WorkflowRefusal("unsupported port shape");
  return Object.freeze({
    schema,
    dtype: dtype as WorkflowDtype,
    shape: Object.freeze(shape),
    unit: text(p["unit"], "port unit"),
  });
}
function stage(value: unknown): WorkflowStage {
  const p = object(value, "stage", [
    "id",
    "adapter",
    "verb",
    "backend",
    "parameters",
    "inputs",
    "outputs",
    "depends_on",
  ]);
  const adapter = text(p["adapter"], "adapter"),
    verb = text(p["verb"], "verb");
  if ((adapter !== "executive" && adapter !== "local-kuramoto") || !verbs.has(verb))
    throw new WorkflowRefusal("unsupported workflow adapter or verb");
  if (adapter === "local-kuramoto" && !["validate", "simulate", "analyse"].includes(verb))
    throw new WorkflowRefusal("local classical adapter does not implement this verb");
  const parameters = object(p["parameters"], "parameters");
  const inputs = array(p["inputs"], "inputs", 16).map((item): WorkflowInput => {
    const q = object(item, "input", ["parameter", "source_stage", "source_port", "type"]);
    return Object.freeze({
      parameter: parameterKey(q["parameter"], "input parameter"),
      source_stage: identifier(q["source_stage"], "source stage"),
      source_port: identifier(q["source_port"], "source port"),
      type: port(q["type"]),
    });
  });
  const outputs = array(p["outputs"], "outputs", 16).map((item): WorkflowOutput => {
    const q = object(item, "output", ["name", "path", "type"]);
    const path = array(q["path"], "output path", 16).map((key) =>
      text(key, "output path key", 256),
    );
    if (path.length === 0) throw new WorkflowRefusal("nonempty original output path required");
    return Object.freeze({
      name: identifier(q["name"], "output name"),
      path: Object.freeze(path),
      type: port(q["type"]),
    });
  });
  const incoming = inputs.map((i) => i.parameter),
    outgoing = outputs.map((o) => o.name);
  const depends_on = array(p["depends_on"], "dependencies", maxWorkflowStages).map((id) =>
    identifier(id, "dependency"),
  );
  if (
    new Set(incoming).size !== incoming.length ||
    new Set(outgoing).size !== outgoing.length ||
    new Set(depends_on).size !== depends_on.length
  )
    throw new WorkflowRefusal("duplicate stage port or dependency");
  if (incoming.some((key) => Object.hasOwn(parameters, key)))
    throw new WorkflowRefusal("a bound input cannot also have a constant parameter");
  return Object.freeze({
    id: identifier(p["id"], "stage id"),
    adapter,
    verb,
    backend: text(p["backend"], "backend"),
    parameters: captureRecord(parameters),
    inputs: Object.freeze(inputs),
    outputs: Object.freeze(outputs),
    depends_on: Object.freeze(depends_on),
  });
}
function sweep(value: unknown, stages: readonly WorkflowStage[]): WorkflowSweep {
  const p = object(value, "sweep", ["axes", "seeds", "seed_binding", "evaluation_budget"]);
  const byId = new Map(stages.map((s) => [s.id, s]));
  const targets = new Set<string>();
  let cells = 1;
  const axes = array(p["axes"], "axes", 4).map((item): WorkflowAxis => {
    const q = object(item, "axis", ["stage_id", "parameter", "values"]);
    const stage_id = identifier(q["stage_id"], "axis stage"),
      parameter = parameterKey(q["parameter"], "axis parameter");
    const target = JSON.stringify([stage_id, parameter]);
    const values = array(q["values"], "axis values", 64);
    const identities = values.map((v) =>
      Array.from(canonicalBytes("studio.workflow-axis-value.v1", v)).join(","),
    );
    if (values.length === 0 || new Set(identities).size !== values.length || targets.has(target))
      throw new WorkflowRefusal("nonempty unique sweep coordinates required");
    const targetStage = byId.get(stage_id);
    if (targetStage === undefined || targetStage.inputs.some((i) => i.parameter === parameter))
      throw new WorkflowRefusal("sweep target is absent or already bound to a source");
    targets.add(target);
    cells *= values.length;
    return Object.freeze({ stage_id, parameter, values: Object.freeze(values.map(capture)) });
  });
  const seeds = array(p["seeds"], "seeds", 64).map((v) => text(v, "seed", 20));
  if (
    seeds.length === 0 ||
    new Set(seeds).size !== seeds.length ||
    seeds.some((s) => !/^(?:0|[1-9][0-9]{0,19})$/.test(s) || BigInt(s) > 2n ** 64n - 1n)
  )
    throw new WorkflowRefusal("unique canonical uint64 seeds required");
  let seed_binding: WorkflowSweep["seed_binding"] = null;
  if (p["seed_binding"] !== null) {
    const q = object(p["seed_binding"], "seed binding", ["stage_id", "parameter"]);
    const stage_id = identifier(q["stage_id"], "seed stage"),
      parameter = parameterKey(q["parameter"], "seed parameter");
    const targetStage = byId.get(stage_id);
    if (
      targets.has(JSON.stringify([stage_id, parameter])) ||
      targetStage === undefined ||
      Object.hasOwn(targetStage.parameters, parameter) ||
      targetStage.inputs.some((i) => i.parameter === parameter)
    )
      throw new WorkflowRefusal("seed target is absent or already has a value");
    seed_binding = Object.freeze({ stage_id, parameter });
  }
  const evaluation_budget = integer(
    p["evaluation_budget"],
    "evaluation budget",
    1n,
    BigInt(maxWorkflowEvaluations),
  );
  cells *= seeds.length;
  if (cells > maxWorkflowCells || BigInt(cells * stages.length) > evaluation_budget)
    throw new WorkflowRefusal("sweep exceeds cell or stage evaluation budget");
  return Object.freeze({
    axes: Object.freeze(axes),
    seeds: Object.freeze(seeds),
    seed_binding,
    evaluation_budget,
  });
}

/** Return every stage once in stable ASCII lexical order, respecting data and control edges.
 * Throws for missing dependencies or any graph cycle. No operation executes.
 */
export function topologicalOrder(definition: WorkflowDefinition): readonly string[] {
  const pending = new Map(
    definition.stages.map((s) => [
      s.id,
      new Set([...s.depends_on, ...s.inputs.map((i) => i.source_stage)]),
    ]),
  );
  if (pending.size === 0 || pending.size !== definition.stages.length)
    throw new WorkflowRefusal("nonempty unique workflow stages required");
  if (Array.from(pending.values()).some((deps) => Array.from(deps).some((d) => !pending.has(d))))
    throw new WorkflowRefusal("workflow dependency is absent");
  const order: string[] = [];
  while (pending.size !== 0) {
    const ready = Array.from(pending)
      .filter(([, deps]) => deps.size === 0)
      .map(([id]) => id)
      .sort();
    const id = ready[0];
    if (id === undefined) throw new WorkflowRefusal("workflow graph contains a cycle");
    order.push(id);
    pending.delete(id);
    for (const dependencies of pending.values()) dependencies.delete(id);
  }
  return Object.freeze(order);
}
/** Admit a bounded immutable v1 graph before any request, worker or save exists.
 * Uses the original lossless codec; rejects versions, unknown fields, duplicate
 * identities, dangling edges, incompatible exact port declarations and budgets.
 */
export function parseWorkflow(payload: unknown): WorkflowDefinition {
  const json = writeJson(payload);
  if (new TextEncoder().encode(json).length > maxWorkflowBytes)
    throw new WorkflowRefusal("workflow definition exceeds byte bound");
  const document = object(readJson(json), "workflow", ["schema", "body", "extensions"]);
  if (document["schema"] !== workflowSchema)
    throw new WorkflowRefusal("unsupported workflow schema");
  const body = object(document["body"], "workflow body", ["workflow_id", "stages", "sweep"]);
  const stages = array(body["stages"], "stages", maxWorkflowStages).map(stage);
  if (stages.length === 0 || new Set(stages.map((s) => s.id)).size !== stages.length)
    throw new WorkflowRefusal("nonempty unique workflow stages required");
  const byId = new Map(stages.map((s) => [s.id, s]));
  if (stages.reduce((count, s) => count + s.inputs.length + s.depends_on.length, 0) > 128)
    throw new WorkflowRefusal("workflow edge bound exceeded");
  for (const s of stages)
    for (const input of s.inputs) {
      const producer = byId.get(input.source_stage);
      if (producer === undefined) throw new WorkflowRefusal("input producer stage is absent");
      const output = producer.outputs.find((o) => o.name === input.source_port);
      if (output === undefined || writeJson(output.type) !== writeJson(input.type))
        throw new WorkflowRefusal("input and original output port types are incompatible");
    }
  const definition = Object.freeze({
    workflow_id: identifier(body["workflow_id"], "workflow id"),
    stages: Object.freeze(stages),
    sweep: sweep(body["sweep"], stages),
    extensions: captureRecord(object(document["extensions"], "extensions")),
  });
  topologicalOrder(definition);
  return definition;
}
/** Produce an independent lossless wire document with original stage order and values. */
export function workflowDocument(
  definition: WorkflowDefinition,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: workflowSchema,
    body: Object.freeze({
      workflow_id: definition.workflow_id,
      stages: definition.stages,
      sweep: definition.sweep,
    }),
    extensions: definition.extensions,
  });
}
/** Check original producer values against exact declared dtype/shape, without converting.
 * Original handlers still validate domain and meaning. Returns an independent
 * immutable datum; format, finiteness, scalar type or shape mismatch refuses.
 */
export function validatePortValue(type: WorkflowPortType, value: unknown): unknown {
  canonicalBytes("studio.workflow-port-value.v1", value);
  port(type);
  const check = (item: unknown, shape: readonly bigint[]): void => {
    const [dimension, ...rest] = shape;
    if (dimension !== undefined) {
      if (!Array.isArray(item) || BigInt(item.length) !== dimension)
        throw new WorkflowRefusal("original port shape differs");
      for (const element of item) check(element, rest);
      return;
    }
    const valid =
      type.dtype === "json" ||
      (type.dtype === "utf8" && typeof item === "string") ||
      (type.dtype === "bool" && typeof item === "boolean") ||
      (type.dtype === "float64" && typeof item === "number" && Number.isFinite(item)) ||
      (type.dtype === "int64" &&
        typeof item === "bigint" &&
        item >= -(2n ** 63n) &&
        item < 2n ** 63n) ||
      (type.dtype === "uint64" && typeof item === "bigint" && item >= 0n && item < 2n ** 64n);
    if (!valid) throw new WorkflowRefusal("original port dtype differs");
  };
  check(value, type.shape);
  return capture(readJson(writeJson(value)));
}
