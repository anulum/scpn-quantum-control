// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact original experiment source artifacts

import { createHash, webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { encodeKuramotoInput } from "../../panel/kuramoto";
import { writeJson } from "../../shared/contracts";
import { admitOwnedKuramotoResources, browserResourcePolicy } from "../../shared/resources/kuramotoResources";
import { artifactContent, artifactBytesDigest, decodeKernelInput, decodeFloat64, encodeFloat64, makeArtifact, localExperimentCodecs, readExperimentArtifact } from "./kuramotoArtifacts";

beforeEach(() => { vi.stubGlobal("crypto", webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });

it("retains the original little-endian ABI and raw digest including negative zero", async () => {
  const request = { mode: "mean-field" as const, omega: [-0, 0.2], theta0: [0, 0.8], dt: 0.01, coupling: 1.4, steps: 40 };
  const bytes = encodeKuramotoInput(request)!;
  const view = new DataView(bytes.buffer);
  expect(bytes.length).toBe(64);
  expect(view.getUint32(0, true)).toBe(1);
  expect(view.getUint32(4, true)).toBe(0);
  expect(view.getUint32(8, true)).toBe(2);
  expect(view.getUint32(12, true)).toBe(40);
  expect(view.getBigUint64(32, true)).toBe(0x8000000000000000n);
  expect(decodeKernelInput(bytes)).toEqual(request);
  const artifact = await makeArtifact("input", bytes);
  expect(artifact.sha256).toBe(createHash("sha256").update(bytes).digest("hex"));
  expect(await localExperimentCodecs.get(artifact.schema)!(bytes)).toMatchObject({ kind: "problem", digest: artifact.sha256 });
  expect(Object.is(decodeFloat64(encodeFloat64(-0)), -0)).toBe(true);
});

it("refuses malformed ABI/version/shape/nonfinite bytes before releasing input", () => {
  const bytes = encodeKuramotoInput({ mode: "networked", omega: [0, 0], theta0: [0, 1], steps: 2, dt: 0.1, coupling: 0, kNm: [0, 1, 1, 0] })!;
  expect(decodeKernelInput(bytes).kNm).toEqual([0, 1, 1, 0]);
  for (const [offset, value] of [[0, 2], [4, 9], [8, 0], [12, 4097]] as const) {
    const bad = bytes.slice(); new DataView(bad.buffer).setUint32(offset, value, true);
    expect(() => decodeKernelInput(bad)).toThrow();
  }
  expect(() => decodeKernelInput(bytes.slice(1))).toThrow();
  expect(() => decodeKernelInput(bytes.slice(0, -8))).toThrow("kernel input byte shape mismatch");
  const nonfinite = bytes.slice(); new DataView(nonfinite.buffer).setFloat64(16, Infinity, true);
  expect(() => decodeKernelInput(nonfinite)).toThrow();
  expect(() => decodeFloat64("7ff0000000000000")).toThrow();
  expect(() => decodeFloat64("-0")).toThrow();
  expect(() => decodeFloat64(0)).toThrow();
  expect(() => encodeFloat64(NaN)).toThrow();
  expect(() => decodeKernelInput(new Uint8Array(0))).toThrow();
  expect(() => decodeKernelInput(new Uint8Array(2 * 1024 * 1024 + 1))).toThrow();
  for (const [offset, value] of [[8, 129], [12, 0]] as const) {
    const bad = bytes.slice(); new DataView(bad.buffer).setUint32(offset, value, true);
    expect(() => decodeKernelInput(bad)).toThrow();
  }
});

const nativeBounds = { maxOscillators: 128, maxSteps: 4096 };
const sha = "a".repeat(64);
const observedEnvironment = { backend: "shipped-kuramoto-wasm-float64-owned-worker", integrator: "Rust fixed-step RK4", dtype: "float64",
  phase_unit: "rad", frequency_unit: "rad/model-time", time_unit: "model-time", seed: null, kernel_sha256: sha, bounds: nativeBounds, browser_user_agent: "observed test browser" };
const bytesOf = (value: unknown) => new TextEncoder().encode(writeJson(value));

it("offline verifiers retain every declared source role, bytes and exact integer policy", async () => {
  const declaredPolicy = browserResourcePolicy(nativeBounds);
  for (const [kind, body, role] of [
    ["kernel", new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]), "program"],
    ["policy", { ...declaredPolicy }, "policy"],
    ["environment", observedEnvironment, "environment"],
    ["output", { revision_hash: sha, plan_hash: sha, kernel_sha256: sha, order_parameter: [encodeFloat64(0.9)], theta_final: [encodeFloat64(-0)] }, "output"],
  ] as const) {
    const member = await makeArtifact(kind, body);
    const raw = artifactContent(member);
    expect(member.sha256).toBe(createHash("sha256").update(raw).digest("hex"));
    expect(await artifactBytesDigest(raw)).toBe(member.sha256);
    expect(await localExperimentCodecs.get(member.schema)!(raw)).toMatchObject({ kind: role, digest: member.sha256 });
    if (kind !== "kernel") expect(await readExperimentArtifact(kind, raw)).toEqual(body);
  }
  const admission = admitOwnedKuramotoResources({ n: 2, steps: 40, mode: "mean-field" }, nativeBounds, 8);
  const body = { revision_hash: sha, kernel_sha256: sha, input_sha256: sha, environment_sha256: sha, policy_sha256: sha,
    shape: { n: 2, steps: 40, mode: "mean-field" }, bounds: nativeBounds, binary_bytes: 8, policy: admission.policy, requested_memory_bytes: null, deadline_ms: 5000, admission };
  expect(await readExperimentArtifact("plan", artifactContent(await makeArtifact("plan", body)))).toEqual(body);
  const explicit = { ...body, requested_memory_bytes: admission.policy.memoryBytes };
  expect((await readExperimentArtifact("plan", bytesOf(explicit)))["requested_memory_bytes"]).toBe(admission.policy.memoryBytes);
});

it("refuses unknown, oversized, nonfinite or malformed artifact metadata offline", async () => {
  for (const body of [null, [], {}, { ...observedEnvironment, future: true }]) await expect(readExperimentArtifact("environment", bytesOf(body))).rejects.toThrow();
  for (const [key, value] of [
    ["backend", "iqm"], ["integrator", "unknown"], ["dtype", "float32"], ["phase_unit", "degree"], ["frequency_unit", "Hz"], ["time_unit", "s"], ["seed", 0],
    ["kernel_sha256", "A".repeat(64)], ["kernel_sha256", "a".repeat(63)], ["kernel_sha256", 0], ["bounds", {}], ["bounds", { maxOscillators: 0, maxSteps: 4096 }],
    ["browser_user_agent", 0], ["browser_user_agent", "x".repeat(4097)],
  ] as const) await expect(readExperimentArtifact("environment", bytesOf({ ...observedEnvironment, [key]: value }))).rejects.toThrow();
  await expect(readExperimentArtifact("policy", bytesOf({ ...browserResourcePolicy(nativeBounds), memoryBytes: -1n }))).rejects.toThrow();
  for (const bytes of [new Uint8Array(), new Uint8Array(2 * 1024 * 1024 + 1), new Uint8Array([255])]) await expect(readExperimentArtifact("policy", bytes)).rejects.toThrow();
  for (const raw of [new Uint8Array(), new Uint8Array(2 * 1024 * 1024 + 1), new Uint8Array([0, 97, 115, 109, 2, 0, 0, 0])]) await expect(makeArtifact("kernel", raw)).rejects.toThrow();
  const output = { revision_hash: sha, plan_hash: sha, kernel_sha256: sha, order_parameter: [encodeFloat64(1)], theta_final: [encodeFloat64(0)] };
  for (const [key, value] of [["order_parameter", null], ["order_parameter", []], ["order_parameter", Array(4098).fill(encodeFloat64(0))], ["theta_final", Array(129).fill(encodeFloat64(0))], ["theta_final", ["fff0000000000000"]]] as const) await expect(readExperimentArtifact("output", bytesOf({ ...output, [key]: value }))).rejects.toThrow();
  const valid = await makeArtifact("kernel", new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
  for (const change of [{ kind: "document" as const }, { content: "a" }, { content: "AA" }, { content: "z0" }, { content: "00".repeat(2 * 1024 * 1024 + 1) }]) expect(() => artifactContent({ ...valid, ...change })).toThrow();
  expect(artifactContent({ ...valid, content: "" })).toEqual(new Uint8Array());
});

it("the plan verifier recomputes complete original admission instead of trusting recorded success", async () => {
  const admission = admitOwnedKuramotoResources({ n: 2, steps: 40, mode: "mean-field" }, nativeBounds, 8);
  const valid = { revision_hash: sha, kernel_sha256: sha, input_sha256: sha, environment_sha256: sha, policy_sha256: sha,
    shape: { n: 2, steps: 40, mode: "mean-field" }, bounds: nativeBounds, binary_bytes: 8, policy: admission.policy, requested_memory_bytes: null, deadline_ms: 5000, admission };
  for (const change of [{ admission: { ...admission, bytesRequired: 0n } }, { requested_memory_bytes: 0n }, { requested_memory_bytes: -1n }, { requested_memory_bytes: "0" },
    { deadline_ms: 0 }, { deadline_ms: 60_001 }, { deadline_ms: 1.5 }, { deadline_ms: "5000" }, { policy: {} }, { shape: {} }]) await expect(readExperimentArtifact("plan", bytesOf({ ...valid, ...change }))).rejects.toThrow();
});
