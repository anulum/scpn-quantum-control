// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — immutable workspace documents

import { canonicalBytes, canonicalDigest, dataEntries } from "./canonical";
import { readJson } from "./jsonTransport";

/** Supported metadata documents; none grants execution authority. */
export const workspaceSchemas = Object.freeze([
  "quantum_workspace.v1", "experiment_revision.v1", "parameter_spec.v1", "resolved_settings.v1", "local_run_record.v1",
] as const);
/** Exact versioned name of one supported metadata document. */
export type WorkspaceSchema = typeof workspaceSchemas[number];
/** Recursively frozen structural snapshot, requiring separate graph admission. */
export interface WorkspaceDocument<S extends WorkspaceSchema = WorkspaceSchema> {
  /** Exact versioned document kind. */ readonly schema: S;
  /** Validated named fields with exact scalar values. */ readonly body: Readonly<Record<string, unknown>>;
  /** Opaque optional metadata, preserved without numeric coercion. */ readonly extensions: Readonly<Record<string, unknown>>;
}
/** A workspace root. */
export type WorkspaceManifest = WorkspaceDocument<"quantum_workspace.v1">;
/** Immutable experiment inputs. */
export type ExperimentRevision = WorkspaceDocument<"experiment_revision.v1">;
/** Typed parameter domain and provenance. */
export type ParameterSpec = WorkspaceDocument<"parameter_spec.v1">;
/** Recorded requested/effective settings without policy execution. */
export type ResolvedSettings = WorkspaceDocument<"resolved_settings.v1">;
/** Recorded local events without a successful-execution claim. */
export type LocalRunRecord = WorkspaceDocument<"local_run_record.v1">;
/** Explicit structural acceptance or a field-addressed refusal. */
export type ParseResult<T> = {
  /** Structural validation succeeded; the value is available. */
  readonly ok: true;
  /** Fully validated immutable document or projection. */
  readonly value: T;
} | {
  /** Structural validation refused the supplied input. */
  readonly ok: false;
  /** Stable category identifying the contract refusal. */
  readonly code: string;
  /** Field address at which validation refused the input. */
  readonly path: string;
  /** Human-readable explanation of the refusal. */
  readonly message: string;
};

const fields: Record<WorkspaceSchema, readonly string[]> = {
  "quantum_workspace.v1": ["project_id", "revision_refs", "draft_ref", "created_at", "updated_at", "artefact_refs"],
  "experiment_revision.v1": ["project_id", "parent_revision_hashes", "problem_ref", "program_ref", "parameters", "semantic_settings_ref", "input_refs"],
  "parameter_spec.v1": ["key", "dtype", "shape", "unit", "domain", "default_source", "trainable", "dependency_keys"],
  "resolved_settings.v1": ["requested", "effective", "origins", "policy_ref", "environment_ref", "rejected_fields"],
  "local_run_record.v1": ["run_id", "attempt_id", "revision_hash", "plan_hash", "mode", "events", "output_refs"],
};
const safeInteger = 9007199254740991n;
class ContractError extends Error {
  constructor(readonly path: string, message: string) { super(message); }
}
function fail(path: string, message: string): never { throw new ContractError(path, message); }
function full(pattern: RegExp, text: string): boolean { return pattern.exec(text)?.[0] === text; }
function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return fail(path, "object required");
  return Object.fromEntries(dataEntries(value, path));
}
function keys(value: Record<string, unknown>, required: readonly string[], path: string, optional: readonly string[] = []): void {
  if (required.some(key => !Object.hasOwn(value, key)) || Object.keys(value).some(key => !required.includes(key) && !optional.includes(key))) fail(path, "missing or unknown field");
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || /^[\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u.test(value)) return fail(path, "nonempty string required");
  return value;
}
function list(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) return fail(path, "array required");
  return [...value] as unknown[];
}
function strings(value: unknown, path: string): string[] {
  const result = list(value, path).map(item => text(item, path));
  if (new Set(result).size !== result.length) fail(path, "duplicate entry");
  return result;
}
function integer(value: unknown, path: string): bigint {
  if (typeof value === "number" && Number.isSafeInteger(value) && !Object.is(value, -0)) value = BigInt(value);
  if (typeof value !== "bigint" || value < 0n || value > safeInteger) return fail(path, "safe nonnegative integer required");
  return value;
}
function shape(value: unknown, path: string): { dimensions: bigint[]; size: bigint } {
  const dimensions = list(value, path).map(item => integer(item, path));
  let size = dimensions.includes(0n) ? 0n : 1n;
  for (const dimension of dimensions) {
    size *= dimension;
    if (size > safeInteger) fail(path, "shape product overflow");
  }
  return { dimensions, size };
}
function hash(value: unknown, path: string): string {
  const result = text(value, path);
  if (!full(/[0-9a-f]{64}/, result)) fail(path, "lowercase SHA-256 required");
  return result;
}
function uuid(value: unknown, path: string): string {
  const result = text(value, path);
  if (!full(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/, result)) fail(path, "canonical UUID required");
  return result;
}
function reference(value: unknown, path: string): Record<string, unknown> {
  const ref = object(value, path);
  keys(ref, ["schema", "sha256", "media_type"], path, ["name"]);
  if (!full(/[A-Za-z0-9_.-]+\.v[1-9][0-9]*/, text(ref["schema"], path))) fail(path + ".schema", "versioned schema required");
  hash(ref["sha256"], path + ".sha256");
  if (!full(/[A-Za-z0-9.+-]+\/[A-Za-z0-9.+-]+/, text(ref["media_type"], path))) fail(path + ".media_type", "media type required");
  if ("name" in ref) {
    const name = text(ref["name"], path + ".name");
    if (/[\\:%?#@\x00-\x1f]/.test(name) || name.split("/").some(part => ["", ".", ".."].includes(part))) fail(path + ".name", "safe project-relative name required");
  }
  return ref;
}
function references(value: unknown, path: string): Record<string, unknown>[] {
  const refs = list(value, path).map(item => reference(item, path));
  if (new Set(refs.map(ref => `${String(ref["schema"])}:${String(ref["sha256"])}`)).size !== refs.length) fail(path, "duplicate reference");
  return refs;
}
function dtype(value: unknown, path: string): string {
  const result = text(value, path);
  if (!["float64", "int64", "uint64"].includes(result)) fail(path, "unsupported dtype");
  return result;
}
function element(value: unknown, kind: string, path: string): bigint | number {
  const literal = text(value, path);
  if (kind === "float64") {
    if (!full(/[0-9a-f]{16}/, literal)) fail(path, "16-digit lowercase IEEE hex required");
    const bytes = Uint8Array.from(literal.match(/../g)!, pair => parseInt(pair, 16));
    const result = new DataView(bytes.buffer).getFloat64(0, false);
    if (!Number.isFinite(result)) fail(path, "non-finite element");
    return result;
  }
  if (!full(/0|-?[1-9][0-9]{0,19}/, literal)) fail(path, "canonical bounded decimal required");
  const result = BigInt(literal);
  const [lower, upper] = kind === "int64" ? [-(2n ** 63n), 2n ** 63n - 1n] as const : [0n, 2n ** 64n - 1n] as const;
  if (result < lower || result > upper) fail(path, "integer element overflow");
  return result;
}
function typed(value: unknown, path: string): Record<string, unknown> {
  const result = object(value, path);
  keys(result, ["dtype", "shape", "values"], path);
  const kind = dtype(result["dtype"], path + ".dtype");
  const dimensions = shape(result["shape"], path + ".shape");
  const values = list(result["values"], path + ".values");
  if (BigInt(values.length) !== dimensions.size) fail(path, "shape/value cardinality mismatch");
  values.forEach((item, index) => element(item, kind, `${path}.values[${index}]`));
  return { dtype: kind, shape: dimensions.dimensions, values };
}
function domain(value: unknown, kind: string, path: string): Record<string, unknown> {
  const result = object(value, path);
  if (result["kind"] === "finite") keys(result, ["kind"], path);
  else if (result["kind"] === "closed_interval") {
    keys(result, ["kind", "lower", "upper"], path);
    if (element(result["lower"], kind, path) > element(result["upper"], kind, path)) fail(path, "reversed interval");
  } else if (result["kind"] === "enumerated") {
    keys(result, ["kind", "values"], path);
    const values = strings(result["values"], path);
    if (values.length === 0) fail(path, "empty domain");
    values.forEach(item => element(item, kind, path));
  } else fail(path, "unsupported domain");
  return result;
}
function timestamp(value: unknown, path: string): string {
  const literal = text(value, path);
  const pattern = /([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?Z/;
  const match = pattern.exec(literal);
  if (!match || match[0] !== literal) return fail(path, "UTC timestamp required");
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  const hour = Number(match[4]), minute = Number(match[5]), second = Number(match[6]);
  const date = new Date(0);
  date.setUTCFullYear(year, month - 1, day);
  date.setUTCHours(hour, minute, second, 0);
  if (year < 1 || date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day || date.getUTCHours() !== hour || date.getUTCMinutes() !== minute || date.getUTCSeconds() !== second) fail(path, "invalid calendar timestamp");
  return literal.slice(0, 19) + "." + (match[7] ?? "").padEnd(9, "0");
}

function validate(schema: WorkspaceSchema, body: Record<string, unknown>): void {
  keys(body, fields[schema], "$.body");
  if (schema === "quantum_workspace.v1" || schema === "experiment_revision.v1") uuid(body["project_id"], "$.body.project_id");
  if (schema === "quantum_workspace.v1") {
    body["revision_refs"] = references(body["revision_refs"], "$.body.revision_refs");
    body["artefact_refs"] = references(body["artefact_refs"], "$.body.artefact_refs");
    if (body["draft_ref"] !== null) body["draft_ref"] = reference(body["draft_ref"], "$.body.draft_ref");
    if (timestamp(body["updated_at"], "$.body.updated_at") < timestamp(body["created_at"], "$.body.created_at")) fail("$.body.updated_at", "precedes creation");
  } else if (schema === "experiment_revision.v1") {
    strings(body["parent_revision_hashes"], "$.body.parent_revision_hashes").forEach(parent => hash(parent, "$.body.parent_revision_hashes"));
    for (const key of ["problem_ref", "program_ref", "semantic_settings_ref"]) body[key] = reference(body[key], "$.body." + key);
    body["input_refs"] = references(body["input_refs"], "$.body.input_refs");
    const parameters = object(body["parameters"], "$.body.parameters");
    body["parameters"] = Object.fromEntries(Object.entries(parameters).map(([key, item]) => [text(key, "$.body.parameters"), typed(item, "$.body.parameters." + key)]));
  } else if (schema === "parameter_spec.v1") {
    for (const key of ["key", "unit", "default_source"]) text(body[key], "$.body." + key);
    const kind = dtype(body["dtype"], "$.body.dtype");
    body["shape"] = shape(body["shape"], "$.body.shape").dimensions;
    body["domain"] = domain(body["domain"], kind, "$.body.domain");
    if (typeof body["trainable"] !== "boolean") fail("$.body.trainable", "boolean required");
    strings(body["dependency_keys"], "$.body.dependency_keys");
  } else if (schema === "resolved_settings.v1") {
    object(body["requested"], "$.body.requested");
    const effective = object(body["effective"], "$.body.effective");
    const origins = object(body["origins"], "$.body.origins");
    if (Object.keys(origins).length !== Object.keys(effective).length || Object.keys(effective).some(key => !Object.hasOwn(origins, key))) fail("$.body.origins", "must cover effective fields exactly");
    for (const [key, origin] of Object.entries(origins)) {
      if (typeof origin === "string") text(origin, "$.body.origins." + key);
      else if (Object.keys(object(origin, "$.body.origins." + key)).length === 0) fail("$.body.origins", "empty provenance");
    }
    for (const key of ["policy_ref", "environment_ref"]) body[key] = reference(body[key], "$.body." + key);
    strings(body["rejected_fields"], "$.body.rejected_fields");
  } else {
    for (const key of ["run_id", "attempt_id"]) uuid(body[key], "$.body." + key);
    for (const key of ["revision_hash", "plan_hash"]) hash(body[key], "$.body." + key);
    if (body["mode"] !== "local") fail("$.body.mode", "unsupported mode");
    let previous = -1n;
    body["events"] = list(body["events"], "$.body.events").map(raw => {
      const event = object(raw, "$.body.events");
      keys(event, ["version", "run_id", "sequence", "kind", "payload"], "$.body.events");
      const version = integer(event["version"], "$.body.events.version");
      const sequence = integer(event["sequence"], "$.body.events.sequence");
      if (version !== 1n || event["run_id"] !== body["run_id"] || typeof event["kind"] !== "string" || !["accepted", "progress", "result", "failed", "cancelled"].includes(event["kind"])) fail("$.body.events", "version, run or event kind mismatch");
      if (sequence <= previous) fail("$.body.events.sequence", "must increase");
      previous = sequence;
      object(event["payload"], "$.body.events.payload");
      return { ...event, version, sequence };
    });
    body["output_refs"] = references(body["output_refs"], "$.body.output_refs");
  }
}

function snapshot(value: unknown, freeze: boolean): unknown {
  if (typeof value !== "object" || value === null) return value;
  const result: object = Array.isArray(value) ? value.map(item => snapshot(item, freeze)) : Object.fromEntries(dataEntries(value).map(([key, item]) => [key, snapshot(item, freeze)]));
  return freeze ? Object.freeze(result) : result;
}
function refusal(error: unknown): ParseResult<never> {
  return { ok: false, code: "invalid_document", path: error instanceof ContractError ? error.path : "$", message: error instanceof Error ? error.message : "Document refused" };
}

/** Parse one of five envelopes, validate exact fields and take a recursive snapshot. */
export function parseDocument(payload: unknown): ParseResult<WorkspaceDocument> {
  try {
    canonicalBytes("workspace_input.v1", payload);
    const record = object(payload, "$");
    keys(record, ["schema", "body", "extensions"], "$");
    const schema = text(record["schema"], "$.schema");
    if (!Object.hasOwn(fields, schema)) fail("$.schema", "unsupported workspace schema");
    const body = object(record["body"], "$.body");
    const extensions = object(record["extensions"], "$.extensions");
    validate(schema as WorkspaceSchema, body);
    return { ok: true, value: snapshot({ schema, body, extensions }, true) as WorkspaceDocument };
  } catch (error: unknown) { return refusal(error); }
}
function named<S extends WorkspaceSchema>(schema: S, payload: unknown): ParseResult<WorkspaceDocument<S>> {
  const parsed = parseDocument(payload);
  if (!parsed.ok) return parsed;
  if (parsed.value.schema !== schema) return { ok: false, code: "schema_mismatch", path: "$.schema", message: `${schema} required` };
  return { ok: true, value: parsed.value as WorkspaceDocument<S> };
}
/** Parse a workspace root without claiming reference admission. */
export function parseWorkspaceManifest(payload: unknown): ParseResult<WorkspaceManifest> { return named("quantum_workspace.v1", payload); }
/** Parse immutable experiment inputs without resolving parents. */
export function parseExperimentRevision(payload: unknown): ParseResult<ExperimentRevision> { return named("experiment_revision.v1", payload); }
/** Parse a typed parameter's exact unit, shape and domain. */
export function parseParameterSpec(payload: unknown): ParseResult<ParameterSpec> { return named("parameter_spec.v1", payload); }
/** Parse recorded setting provenance without executing policy. */
export function parseResolvedSettings(payload: unknown): ParseResult<ResolvedSettings> { return named("resolved_settings.v1", payload); }
/** Parse local event metadata without a successful-run claim. */
export function parseLocalRunRecord(payload: unknown): ParseResult<LocalRunRecord> { return named("local_run_record.v1", payload); }
/** Read portable JSON through the lossless reader before validating the document. */
export function parseDocumentJson(text: string): ParseResult<WorkspaceDocument> {
  try { return parseDocument(readJson(text)); } catch (error: unknown) { return refusal(error); }
}
/** Export a fresh mutable wire snapshot rather than exposing stored containers. */
export function documentToWire(document: WorkspaceDocument): Record<string, unknown> {
  const parsed = parseDocument(document);
  if (!parsed.ok) throw new Error(`${parsed.path}: ${parsed.message}`);
  return snapshot(parsed.value, false) as Record<string, unknown>;
}
/** Revalidate and address a whole immutable document, including its extensions. */
export function documentDigest(document: WorkspaceDocument): Promise<string> {
  return canonicalDigest(document.schema, documentToWire(document));
}
/** Check supplied typed values against a spec without implicit unit conversion. */
export function validateParameterBinding(spec: ParameterSpec, payload: unknown, unit: string): ParseResult<null> {
  try {
    const parsed = parseParameterSpec(spec);
    if (!parsed.ok) return parsed;
    const values = typed(payload, "$.parameters");
    const expected = parsed.value.body;
    const dimensions = values["shape"] as bigint[];
    const expectedDimensions = expected["shape"] as readonly bigint[];
    if (unit !== expected["unit"] || values["dtype"] !== expected["dtype"] || dimensions.length !== expectedDimensions.length || dimensions.some((value, index) => value !== expectedDimensions[index])) fail("$.parameters", "dtype, shape or unit mismatch");
    const kind = values["dtype"] as string;
    const restriction = expected["domain"] as Readonly<Record<string, unknown>>;
    for (const value of values["values"] as readonly unknown[]) {
      const number = element(value, kind, "$.parameters.values");
      if (restriction["kind"] === "closed_interval" && (number < element(restriction["lower"], kind, "$.domain.lower") || number > element(restriction["upper"], kind, "$.domain.upper"))) fail("$.parameters.values", "outside interval");
      if (restriction["kind"] === "enumerated" && !(restriction["values"] as readonly unknown[]).includes(value)) fail("$.parameters.values", "outside enumerated domain");
    }
    return { ok: true, value: null };
  } catch (error: unknown) { return refusal(error); }
}
