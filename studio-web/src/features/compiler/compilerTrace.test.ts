// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native compiler metadata admission tests

import { expect, it } from "vitest";
import nativeTrace from "../../../../data/studio/compiler_trace_demo.json?raw";
import nativeCases from "../../../../data/studio/compiler_trace_cases.json?raw";
import { readJson, writeJson } from "../../shared/contracts/jsonTransport";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { parseCompilerTrace } from "./compilerTrace";
import { MAX_TRACE_BYTES } from "./compilerTrace";

/** Preserve exact native numeric token kinds while changing unsigned metadata. */
async function changed(change: (wire: Record<string, unknown>) => void): Promise<string> {
  const wire = readJson(nativeTrace) as Record<string, unknown>;
  change(wire);
  wire["sha256"] = await canonicalDigest("studio.compiler-trace.v1", { schema: wire["schema"], body: wire["body"], extensions: wire["extensions"] });
  return writeJson(wire);
}
it("admits original producer bytes and keeps both native source snapshots and record digests", async () => {
  const result = await parseCompilerTrace(nativeTrace);
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.message);
  expect(result.value.text).toBe(nativeTrace);
  expect(result.value.passes.length).toBe(2);
  expect(result.value.source).toContain("α🧪");
  const pass = result.value.passes[0]!;
  if (pass.state !== "qualified") throw new Error("Native pass absent");
  expect(pass.outputLayout).toEqual([1, 0]);
  expect(pass.observableMap).toEqual([[0, 1, 1, 1], [1, 0, 0, 0]]);
  expect(pass.input.operations[0]!.parameters).toEqual([0.41]);
  expect(Object.isFrozen(result.value)).toBe(true);
  expect(Object.isFrozen(pass.input.operations)).toBe(true);
});
it("refuses future schema and duplicate keys without fetching any reference", async () => {
  expect((await parseCompilerTrace('{"schema":"studio.compiler-trace.v2"}')).ok).toBe(false);
  expect((await parseCompilerTrace('{"schema":1,"schema":2}')).ok).toBe(false);
});
it("refuses a changed source even when the imported envelope is rebound", async () => {
  const text = await changed(wire => { (wire["body"] as Record<string, unknown>)["source"] = "tampered"; });
  expect((await parseCompilerTrace(text)).ok).toBe(false);
});
it("refuses a changed native snapshot identity and malformed pass layouts", async () => {
  for (const field of ["input_layout", "output_layout", "output_classical_layout"]) {
    const text = await changed(wire => {
      const body = wire["body"] as Record<string, unknown>;
      const pass = (body["passes"] as Record<string, unknown>[])[0]!;
      (pass["record"] as Record<string, unknown>)[field] = [0n, 0n];
    });
    expect((await parseCompilerTrace(text)).ok).toBe(false);
  }
  const text = await changed(wire => {
    const body = wire["body"] as Record<string, unknown>;
    const pass = (body["passes"] as Record<string, unknown>[])[0]!;
    (pass["input"] as Record<string, unknown>)["sha256"] = "0".repeat(64);
  });
  expect((await parseCompilerTrace(text)).ok).toBe(false);
});
it("preserves optional exact extensions without granting native verification", async () => {
  const text = await changed(wire => { wire["extensions"] = { integer: 9007199254740993n, negative_zero: -0 }; });
  const result = await parseCompilerTrace(text);
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.message);
  expect(result.value.text).toBe(text);
  expect(result.value.nativeVerification).toBe("producer_declaration");
});

/** Replace one restored wire field through its public import boundary. */
function replaceField(wire: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split(".");
  let row = wire;
  for (const key of keys.slice(0, -1)) row = row[key] as Record<string, unknown>;
  row[keys.at(-1)!] = value;
}
const input = "body.passes.0.input";
const record = "body.passes.0.record";
const metrics = "body.passes.0.metrics.input";
const originalBody = (readJson(nativeTrace) as Record<string, unknown>)["body"] as Record<string, unknown>;
const originalFirst = (originalBody["passes"] as Record<string, unknown>[])[0]!;

it.each([
  ["body", null, "Object required"],
  ["body", [], "Object required"],
  ["body", true, "Object required"],
  ["body", {}, "Required trace fields"],
  ["body.execution_status", "executed", "emitted-not-executed"],
  ["body.backend_snapshot.compiler", "other", "Unsupported compiler"],
  ["body.backend_snapshot.target", "hardware", "Unsupported compiler"],
  ["body.backend_snapshot.reference_backend", "other", "Unsupported compiler"],
  ["body.backend_snapshot.basis_convention", "big_endian", "Unsupported compiler"],
  ["body.backend_snapshot.compiler_version", 1n, "Bounded nonempty text"],
  ["body.backend_snapshot.compiler_version", "", "Bounded nonempty text"],
  ["body.backend_snapshot.compiler_version", "v".repeat(257), "Bounded nonempty text"],
  ["body.backend_snapshot.settings", [], "Object required"],
  ["body.passes", false, "Bounded array"],
  ["body.passes", Array(33).fill(null), "Bounded array"],
  ["body.passes", [], "At least one"],
  ["body.passes.0", { state: "missing", reason: "source absent" }, "source-bearing pass is missing"],
  ["body.passes.0.state", "unsupported", "Unsupported native pass"],
  [record + ".schema", "future", "Unsupported native reference"],
  [record + ".reference_backend", "other", "Unsupported native reference"],
  [record + ".basis_convention", "other", "Unsupported native reference"],
  [input + ".ir.num_qubits", 0n, "Exact bounded integer"],
  [input + ".ir.num_qubits", 9n, "Exact bounded integer"],
  [input + ".ir.num_qubits", 2.0, "Exact bounded integer"],
  [input + ".ir.global_phase", 0n, "Finite binary64"],
  [input + ".ir.quantum_registers.0", ["q"], "Register name/indices"],
  [input + ".ir.operations.0.source_span.line", 1n, "Source coordinates"],
  [input + ".ir.operations.0.parameters", [1n], "Finite binary64"],
  [input + ".ir.operations.2.qubits", [], "Measurement needs one"],
  [input + ".ir.operations.2.clbits", [], "Measurement needs one"],
  [input + ".ir.operations.2.parameters", [0.0], "Measurement needs one"],
  [input + ".source_sha256", "xyz", "SHA-256 hex"],
  [input + ".source_sha256", "0".repeat(64), "Original source digest"],
  [record + ".input_layout", [0n], "complete bijection"],
  [record + ".observable_map.0", [0n], "Four readout"],
  [metrics + ".operation_count", 0n, "Resource metadata"],
  [metrics + ".num_qubits", 1n, "Resource metadata"],
  [metrics + ".num_clbits", 1n, "Resource metadata"],
  [metrics + ".statevector_bytes", 16n, "Declared dense payload"],
  [metrics + ".operator_bytes", 16n, "Declared dense payload"],
  ["body.passes.0.metrics.delta.depth", 1n, "delta differs"],
  ["body.passes.0.effects.input_measurements", [[0n]], "Readout effect needs"],
  ["body.passes.0.effects.output_measurements", [], "Readout effects differ"],
  ["body.passes.0.effects.source_mapping", "guess", "Unsupported cross-pass"],
  [record + ".tolerance", 0.0, "Invalid original qualification"],
  [record + ".tolerance", 1e-11, "Invalid original qualification"],
  [record + ".operator_error", -0.1, "Invalid original qualification"],
  [record + ".operator_error", 1e-10, "Invalid original qualification"],
  [record + ".allow_global_phase", 1n, "Invalid original qualification"],
  ["body.source_sha256", "0".repeat(64), "Original pinned source"],
  ["body.passes.1.record.input_layout", [0n, 1n], "continuity differs"],
  ["body.complete", false, "Missing native artifacts"],
] as const)("refuses restored metadata at %s without substituting source or device", async (path, value, message) => {
  const text = await changed(wire => replaceField(wire, path, value));
  const result = await parseCompilerTrace(text);
  expect(result.ok).toBe(false);
  if (result.ok) throw new Error("Malformed trace was admitted");
  expect(result.message).toContain(message);
  expect(result.code).not.toBe("trace_unavailable");
});

it("refuses equal-sized field substitution and a changed envelope digest", async () => {
  const text = await changed(wire => {
    const body = wire["body"] as Record<string, unknown>;
    body["undeclared"] = body["complete"]; delete body["complete"];
  });
  expect((await parseCompilerTrace(text)).ok).toBe(false);
  const wire = readJson(nativeTrace) as Record<string, unknown>;
  wire["sha256"] = "0".repeat(64);
  expect(await parseCompilerTrace(writeJson(wire))).toMatchObject({ ok: false, code: "digest_mismatch" });
});

it("refuses disconnected native source even when local effects remain bound", async () => {
  const text = await changed(wire => {
    replaceField(wire, "body.passes.1.input", originalFirst["input"]);
    const effects = originalFirst["effects"] as Record<string, unknown>;
    replaceField(wire, "body.passes.1.effects.input_measurements", effects["input_measurements"]);
  });
  const result = await parseCompilerTrace(text);
  expect(result.ok).toBe(false);
  if (result.ok) throw new Error("Disconnected trace admitted");
  expect(result.message).toContain("continuity differs");
});

it("enforces distinct UTF8 and character ceilings before parsing", async () => {
  for (const text of ["x".repeat(MAX_TRACE_BYTES + 1), "α".repeat(MAX_TRACE_BYTES / 2 + 1)])
    expect(await parseCompilerTrace(text)).toMatchObject({ ok: false, code: "resource_limit" });
  for (const text of ["null", "[]", "true", "1e400", '{"x":"\\ud800"}'])
    expect((await parseCompilerTrace(text)).ok).toBe(false);
});

it("admits real native empty, readout-only, negative-zero and emitted MLIR fixtures", async () => {
  const cases = readJson(nativeCases) as Record<string, unknown>;
  for (const [name, wire] of Object.entries(cases)) {
    const result = await parseCompilerTrace(writeJson(wire));
    expect(result.ok, name).toBe(true);
    if (!result.ok) throw new Error(result.message);
    expect(result.value.passes[0].input.numQubits).toBe(1);
    if (name === "lowering") expect(result.value.emittedText).toContain("module");
    if (name === "empty") expect(result.value.passes[0].before.operationCount).toBe(0);
    if (name === "negative_zero") expect(Object.is(result.value.passes[0].input.operations[0]!.parameters[0], -0)).toBe(true);
  }
});

it("refuses altered textual IR status, identity and native representation widths", async () => {
  const cases = readJson(nativeCases) as Record<string, unknown>;
  for (const [path, value, message] of [
    ["body.emitted_ir.execution_status", "executed", "cannot declare native execution"],
    ["body.emitted_ir.sha256", "0".repeat(64), "byte identity differs"],
    ["body.passes.0.output.ir.num_clbits", 1n, "widths differ"],
  ] as const) {
    const wire = readJson(writeJson(cases["lowering"])) as Record<string, unknown>;
    replaceField(wire, path, value);
    if (path.includes("num_clbits")) {
      const snapshot = (wire["body"] as Record<string, unknown>)["passes"] as Record<string, unknown>[];
      const output = snapshot[0]!["output"] as Record<string, unknown>;
      output["sha256"] = await canonicalDigest("studio.circuit-snapshot.v1", output["ir"]);
    }
    const result = await parseCompilerTrace(writeJson(wire));
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error("Invalid native artifact admitted");
    expect(result.message).toContain(message);
  }
});

it("refuses oversized multibyte native source before hashing it", async () => {
  const wire = readJson(nativeTrace) as Record<string, unknown>;
  replaceField(wire, input + ".ir.source", "α".repeat(524289));
  expect(await parseCompilerTrace(writeJson(wire))).toMatchObject({ ok: false, message: "Source exceeds one MiB." });
});
