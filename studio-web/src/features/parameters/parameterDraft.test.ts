// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — parameter draft public reducer contract tests

import { expect, it } from "vitest";
import { parseExperimentRevision, parseParameterSpec, writeJson } from "../../shared/contracts";
import type { ParseResult } from "../../shared/contracts";
import { createParameterDraft, parameterDraftDigest, parameterDraftReducer, parameterElementText, parameterEditorHistoryLimit, parameterInputUnits, validateParameterSnapshot } from "./parameterDraft";
import type { ParameterDraftSource } from "./parameterDraft";

function take<T>(parsed: ParseResult<T>): T {
  if (!parsed.ok) throw new Error(parsed.message);
  return parsed.value;
}

function source(dtype = "float64", shape: readonly bigint[] = [], values = ["3ff0000000000000"], domain: Record<string, unknown> = { kind: "finite" }, trainable = true): ParameterDraftSource {
  const spec = take(parseParameterSpec({ schema: "parameter_spec.v1", body: {
    key: "parameter", dtype, shape, unit: "declared-unit", domain, default_source: "Independent exact workspace oracle", trainable, dependency_keys: [],
  }, extensions: {} }));
  const ref = { schema: "declared_test_input.v1", sha256: "a".repeat(64), media_type: "application/json" };
  const revision = take(parseExperimentRevision({ schema: "experiment_revision.v1", body: {
    project_id: "00000000-0000-4000-8000-000000000001", parent_revision_hashes: [],
    problem_ref: ref, program_ref: ref, semantic_settings_ref: { ...ref, schema: "resolved_settings.v1" },
    parameters: { parameter: { dtype, shape, values } }, input_refs: [],
  }, extensions: {} }));
  return { revision, specs: [spec], units: { parameter: "declared-unit" } };
}

it("captures independent immutable input and retains exact negative-zero semantics", async () => {
  const original = source();
  const before = writeJson(original);
  const initial = createParameterDraft(original);
  const changed = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text: "-0", unit: "declared-unit" });
  expect(changed.snapshot.parameters["parameter"]!.values).toEqual(["8000000000000000"]);
  expect(parameterElementText(changed.snapshot.parameters["parameter"]!, 0)).toBe("-0");
  expect(await parameterDraftDigest(changed)).not.toBe(await parameterDraftDigest(initial));
  expect(writeJson(original)).toBe(before);
  expect(Object.isFrozen(changed.snapshot.parameters["parameter"]!.values)).toBe(true);
});

it.each(["NaN", "Infinity", "1e999", "", " ", "1\n", "0x1"])("refuses unsupported float form %j with exact snapshot/history retention", text => {
  const initial = createParameterDraft(source());
  const refused = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text, unit: "declared-unit" });
  expect(refused.refusal).toContain("finite float64");
  expect(refused.snapshot).toBe(initial.snapshot);
  expect(refused.past).toBe(initial.past);
});

it("keeps an integer above 2^53 exact through edit and undo and rejects overflow", () => {
  const initial = createParameterDraft(source("uint64", [], ["9007199254740993"]));
  expect(parameterElementText(initial.snapshot.parameters["parameter"]!, 0)).toBe("9007199254740993");
  const changed = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text: "18446744073709551615", unit: "declared-unit" });
  expect(changed.refusal).toBeNull();
  expect(changed.snapshot.parameters["parameter"]!.values).toEqual(["18446744073709551615"]);
  const refused = parameterDraftReducer(changed, { type: "value", key: "parameter", index: 0, text: "18446744073709551616", unit: "declared-unit" });
  expect(refused.refusal).toContain("overflow");
  expect(refused.snapshot).toBe(changed.snapshot);
  expect(parameterDraftReducer(refused, { type: "undo" }).snapshot).toBe(initial.snapshot);
});

it.each(["-0", "+1", "01", "1.0", "1\n"])("refuses noncanonical integer form %j", text => {
  const initial = createParameterDraft(source("int64", [], ["0"]));
  expect(parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text, unit: "declared-unit" }).refusal).toContain("Canonical");
});

it("uses the original closed-interval and enumerated domain refusals", () => {
  for (const [domain, text] of [[{ kind: "closed_interval", lower: "0000000000000000", upper: "3ff0000000000000" }, "2"], [{ kind: "enumerated", values: ["3ff0000000000000"] }, "0"]] as const) {
    const initial = createParameterDraft(source("float64", [], ["3ff0000000000000"], domain));
    const changed = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text, unit: "declared-unit" });
    expect(changed.refusal).toContain("outside");
    expect(changed.snapshot).toBe(initial.snapshot);
  }
});

it("changes trainable subsets with exact undo/redo and preserves source eligibility", async () => {
  const initial = createParameterDraft(source());
  const changed = parameterDraftReducer(initial, { type: "mask", key: "parameter", index: 0, enabled: false });
  expect(changed.snapshot.trainableMasks["parameter"]).toEqual([false]);
  expect(await parameterDraftDigest(changed)).not.toBe(await parameterDraftDigest(initial));
  const restored = parameterDraftReducer(changed, { type: "undo" });
  expect(restored.snapshot).toBe(initial.snapshot);
  expect(parameterDraftReducer(restored, { type: "redo" }).snapshot).toBe(changed.snapshot);
  const fixed = createParameterDraft(source("float64", [], ["3ff0000000000000"], { kind: "finite" }, false));
  expect(parameterDraftReducer(fixed, { type: "mask", key: "parameter", index: 0, enabled: true }).refusal).toContain("eligibility");
});

it("applies symmetric mask edits to the selected pair without silently symmetrising the matrix", async () => {
  const initial = createParameterDraft(source("float64", [2n, 2n], ["0000000000000000", "3ff0000000000000", "c000000000000000", "0000000000000000"]));
  const policy = parameterDraftReducer(initial, { type: "policy", policy: "symmetric" });
  expect(await parameterDraftDigest(policy)).toBe(await parameterDraftDigest(initial));
  expect(policy.snapshot).toBe(initial.snapshot);
  const changed = parameterDraftReducer(policy, { type: "mask", key: "parameter", index: 1, enabled: false });
  expect(changed.snapshot.trainableMasks["parameter"]).toEqual([true, false, false, true]);
  expect(changed.snapshot.parameters).toEqual(initial.snapshot.parameters);
});

it("refuses symmetric edits on a rectangular matrix without changing either edge", () => {
  const initial = createParameterDraft(source("float64", [1n, 2n], ["3ff0000000000000", "c000000000000000"]));
  const policy = parameterDraftReducer(initial, { type: "policy", policy: "symmetric" });
  const changed = parameterDraftReducer(policy, { type: "value", key: "parameter", index: 1, text: "3", unit: "declared-unit" });
  expect(changed.refusal).toContain("square matrix");
  expect(changed.snapshot).toBe(initial.snapshot);
});

it.each([-1, 1, 0.5, NaN])("refuses invalid row-major selection %s", index => {
  const initial = createParameterDraft(source());
  expect(parameterDraftReducer(initial, { type: "select", key: "parameter", index }).refusal).toContain("out of range");
  expect(parameterDraftReducer(initial, { type: "select", key: "__proto__", index: 0 }).refusal).toContain("out of range");
});

it("selects real elements without changing semantic identity and keeps no-op history empty", async () => {
  const initial = createParameterDraft(source());
  const selected = parameterDraftReducer(initial, { type: "select", key: "parameter", index: 0 });
  expect(await parameterDraftDigest(selected)).toBe(await parameterDraftDigest(initial));
  const unchanged = parameterDraftReducer(selected, { type: "value", key: "parameter", index: 0, text: "1", unit: "declared-unit" });
  expect(unchanged.past).toHaveLength(0);
  expect(parameterDraftReducer(unchanged, { type: "undo" })).toBe(unchanged);
  expect(parameterDraftReducer(unchanged, { type: "redo" })).toBe(unchanged);
});

it("bounds undo history and clears redo only after a new semantic branch", () => {
  let state = createParameterDraft(source());
  for (let index = 0; index < parameterEditorHistoryLimit + 3; index++) state = parameterDraftReducer(state, { type: "value", key: "parameter", index: 0, text: String(index + 2), unit: "declared-unit" });
  expect(state.past).toHaveLength(parameterEditorHistoryLimit);
  state = parameterDraftReducer(state, { type: "undo" });
  expect(state.future).toHaveLength(1);
  state = parameterDraftReducer(state, { type: "value", key: "parameter", index: 0, text: "100", unit: "declared-unit" });
  expect(state.future).toHaveLength(0);
});

it("revalidates bulk replacements and refuses dtype, shape and unit drift", () => {
  const initial = createParameterDraft(source());
  for (const [payload, unit] of [[{ dtype: "float64", shape: [1n], values: ["3ff0000000000000"] }, "declared-unit"], [{ dtype: "int64", shape: [], values: ["1"] }, "declared-unit"], [initial.snapshot.parameters["parameter"]!, "other-unit"]] as const) {
    const refused = parameterDraftReducer(initial, { type: "replace", parameters: { parameter: payload }, units: { parameter: unit } });
    expect(refused.refusal).toContain("mismatch");
    expect(refused.snapshot).toBe(initial.snapshot);
  }
  const changed = parameterDraftReducer(initial, { type: "replace", parameters: { parameter: { dtype: "float64", shape: [], values: ["4000000000000000"] } }, units: initial.snapshot.units });
  expect(changed.refusal).toBeNull();
  expect(parameterElementText(changed.snapshot.parameters["parameter"]!, 0)).toBe("2");
});

it("retains empty shapes and refuses unsupported ranks and the declared element excess", () => {
  expect(createParameterDraft(source("float64", [0n], [])).selection).toBeNull();
  expect(() => createParameterDraft(source("float64", [1n, 1n, 1n]))).toThrow("scalar, vector and matrix");
  expect(() => createParameterDraft(source("float64", [4097n], Array(4097).fill("3ff0000000000000") as string[]))).toThrow("4096-element");
  expect(() => parameterElementText(createParameterDraft(source()).snapshot.parameters["parameter"]!, 1)).toThrow("out of range");
});

it("refuses missing or duplicated source declarations and incomplete unit indexes", () => {
  const original = source();
  expect(() => createParameterDraft({ ...original, specs: [] })).toThrow("Missing ParameterSpec");
  expect(() => createParameterDraft({ ...original, specs: [...original.specs, ...original.specs] })).toThrow("Duplicate");
  expect(() => createParameterDraft({ ...original, units: {} })).toThrow("cover");
  expect(() => validateParameterSnapshot(original, { ...createParameterDraft(original).snapshot, parameters: {} })).toThrow("keys must remain unchanged");
});

it("applies only explicit float64 SI-prefix conversions and preserves negative zero", () => {
  const input = source();
  const spec = take(parseParameterSpec({ ...input.specs[0]!, body: { ...input.specs[0]!.body, unit: "rad" } }));
  const initial = createParameterDraft({ ...input, specs: [spec], units: { parameter: "rad" } });
  expect(parameterInputUnits("rad")).toEqual(["rad", "mrad", "urad"]);
  expect(parameterInputUnits("__proto__")).toEqual(["__proto__"]);
  const implicit = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text: "3000", unit: "mrad" });
  expect(implicit.refusal).toContain("unit mismatch");
  const converted = parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text: "3000", unit: "mrad", convert: true });
  expect(converted.snapshot.parameters["parameter"]!.values).toEqual(["4008000000000000"]);
  expect(converted.snapshot.units["parameter"]).toBe("rad");
  const negativeZero = parameterDraftReducer(converted, { type: "value", key: "parameter", index: 0, text: "-0", unit: "urad", convert: true });
  expect(negativeZero.snapshot.parameters["parameter"]!.values).toEqual(["8000000000000000"]);
  expect(parameterDraftReducer(negativeZero, { type: "undo" }).snapshot).toBe(converted.snapshot);
  expect(parameterDraftReducer(initial, { type: "value", key: "parameter", index: 0, text: "1", unit: "seconds", convert: true }).refusal).toContain("Unsupported");
  const arbitrary = createParameterDraft(input);
  expect(parameterDraftReducer(arbitrary, { type: "value", key: "parameter", index: 0, text: "1", unit: "other", convert: true }).refusal).toContain("Unsupported");
  const integerInput = source("int64", [], ["1"]);
  expect(parameterDraftReducer(createParameterDraft(integerInput), { type: "value", key: "parameter", index: 0, text: "1", unit: "other", convert: true }).refusal).toContain("requires float64");
});

it.each([
  null, [], { version: 2n, trainable_masks: { parameter: [true] } },
  { version: 1n, trainable_masks: null }, { version: 1n, trainable_masks: [] },
  { version: 1n, trainable_masks: { parameter: true } },
  { version: 1n, trainable_masks: {} },
  { version: 1n, trainable_masks: { parameter: [] } },
  { version: 1n, trainable_masks: { parameter: ["true"] } },
])("refuses malformed or unsupported persisted mask metadata without changing the input [%#]", metadata => {
  const original = source();
  const revision = { ...original.revision, extensions: { parameter_editor: metadata } };
  const before = writeJson(revision);
  expect(() => createParameterDraft({ ...original, revision })).toThrow(/metadata|version|masks|mask/);
  expect(writeJson(revision)).toBe(before);
});

it("re-admits a saved trainable subset and refuses a mask beyond source eligibility", () => {
  const original = source();
  const revision = { ...original.revision, extensions: { parameter_editor: { version: 1n, trainable_masks: { parameter: [false] } } } };
  expect(createParameterDraft({ ...original, revision }).snapshot.trainableMasks["parameter"]).toEqual([false]);
  const fixed = source("float64", [], ["3ff0000000000000"], { kind: "finite" }, false);
  expect(() => createParameterDraft({ ...fixed, revision: { ...fixed.revision,
    extensions: { parameter_editor: { version: 1n, trainable_masks: { parameter: [true] } } },
  } })).toThrow("source eligibility");
});

it("edits a symmetric diagonal once and refuses a converted value that overflows binary64", () => {
  const initial = createParameterDraft(source("float64", [1n, 1n]));
  const symmetric = parameterDraftReducer(initial, { type: "policy", policy: "symmetric" });
  const changed = parameterDraftReducer(symmetric, { type: "value", key: "parameter", index: 0, text: "2", unit: "declared-unit" });
  expect(changed.snapshot.parameters["parameter"]!.values).toEqual(["4000000000000000"]);
  expect(changed.past).toHaveLength(1);
  const original = source();
  const spec = take(parseParameterSpec({ ...original.specs[0]!, body: { ...original.specs[0]!.body, unit: "rad/s" } }));
  const angular = createParameterDraft({ ...original, specs: [spec], units: { parameter: "rad/s" } });
  const refused = parameterDraftReducer(angular, { type: "value", key: "parameter", index: 0, text: "1e308", unit: "rad/ms", convert: true });
  expect(refused.refusal).toContain("finite float64");
  expect(refused.snapshot).toBe(angular.snapshot);
});

it("maps a hostile dynamic command to a fixed refusal without exposing thrown text", () => {
  const initial = createParameterDraft(source());
  const command = Object.create(null) as object;
  Object.defineProperty(command, "type", { get() { throw "untrusted dynamic command contents"; } });
  const refused = parameterDraftReducer(initial, command as import("./parameterDraft").ParameterAction);
  expect(refused.refusal).toBe("Parameter edit refused");
  expect(refused.snapshot).toBe(initial.snapshot);
  expect(refused.past).toBe(initial.past);
});
