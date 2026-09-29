// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { bindKuramoto, readBounds } from "../../panel/kuramoto";
import type { KuramotoExports, KuramotoRequest } from "../../panel/kuramoto";
import { admitKuramotoResources, browserResourcePolicy, smallerKuramotoRequest } from "./kuramotoResources";
import type { ResourcePolicy } from "./admission";

let exports: KuramotoExports;
beforeAll(async () => {
  const bytes = readFileSync(resolve("..", "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm"));
  const loaded = await WebAssembly.instantiate(bytes, {});
  exports = loaded.instance.exports as unknown as KuramotoExports;
});
function request(): KuramotoRequest {
  return { mode: "mean-field", omega: [0, 0.2], theta0: [0, 0.3], steps: 10, dt: 0.05, coupling: 1 };
}
function policy(change: Partial<ResourcePolicy> = {}): ResourcePolicy {
  return { ...browserResourcePolicy(readBounds(exports)), ...change };
}

describe("original Kuramoto entry with declared resource policy", () => {
  it("accounts for ABI, parser, RK4 and retained output from source-owned counts", () => {
    const row = admitKuramotoResources({ n: 2, steps: 10, mode: "mean-field" }, readBounds(exports));
    // 4 caller scalars, two 64-byte ABI buffers, 4 parsed scalars,
    // 12 stage scalars and three 13-scalar output buffers.
    expect(row.estimate.components.map(component => component.bytes)).toEqual([32n, 128n, 32n, 96n, 312n]);
    expect(row.bytesRequired).toBe(600n);
    expect(row.estimate.workUnits).toBe(80n);
    const networked = admitKuramotoResources({ n: 2, steps: 10, mode: "networked" }, readBounds(exports));
    expect(networked.bytesRequired).toBe(728n);
    expect(networked.estimate.workUnits).toBe(160n);
    expect(networked.estimate.components.every(component => component.dtype === "float64" || component.dtype === "uint8")).toBe(true);
  });

  it("refuses before the real native allocator is called and recovers at exact boundary", () => {
    let allocations = 0;
    const observed = { ...exports, scpn_alloc(length: number) { allocations++; return exports.scpn_alloc(length); } };
    const denied = bindKuramoto(observed, policy({ memoryBytes: 599n }))(request());
    expect(denied.ok).toBe(false);
    expect(allocations).toBe(0);
    const accepted = bindKuramoto(observed, policy({ memoryBytes: 600n, workUnits: 80n }))(request());
    expect(accepted.ok).toBe(true);
    expect(allocations).toBe(2);
    if (accepted.ok) {
      expect(accepted.run.orderParameter.length).toBe(11);
      expect(Array.from(accepted.run.orderParameter).every(Number.isFinite)).toBe(true);
    }
  });

  it("retains refusal for unknown overhead, memory and workload before native entry", () => {
    let entered = 0;
    const observed = { ...exports, scpn_alloc(length: number) { entered++; return exports.scpn_alloc(length); } };
    for (const supplied of [policy({ overheadBytes: null }), policy({ memoryBytes: null }), policy({ workUnits: null }), policy({ workUnits: 79n })]) {
      expect(bindKuramoto(observed, supplied)(request()).ok).toBe(false);
    }
    expect(entered).toBe(0);
  });

  it("refuses unsafe metadata and original kernel limit violations before allocation", () => {
    const bounds = readBounds(exports);
    for (const input of [
      { n: 0, steps: 1, mode: "mean-field" as const },
      { n: 2.5, steps: 1, mode: "mean-field" as const },
      { n: Number.MAX_SAFE_INTEGER, steps: 1, mode: "mean-field" as const },
      { n: bounds.maxOscillators + 1, steps: 1, mode: "mean-field" as const },
      { n: 2, steps: bounds.maxSteps + 1, mode: "mean-field" as const },
      { n: 2, steps: 0, mode: "mean-field" as const },
      { n: 2, steps: 1, mode: "unsupported" as "mean-field" },
    ]) expect(() => admitKuramotoResources(input, bounds)).toThrow();
    expect(() => browserResourcePolicy({ maxOscillators: 0, maxSteps: 1 })).toThrow();
    expect(() => browserResourcePolicy({ maxOscillators: 1, maxSteps: NaN })).toThrow();
  });
});


it("offers only a rechecked smaller same-method configuration", () => {
  const bounds = readBounds(exports);
  const original = { n: 16, steps: 100, mode: "networked" as const };
  const limit = policy({ memoryBytes: 1024n });
  const smaller = smallerKuramotoRequest(original, bounds, limit);
  expect(smaller).not.toBeNull();
  if (smaller) {
    expect(smaller.n).toBeLessThan(original.n);
    expect(smaller.steps).toBeLessThan(original.steps);
    expect(smaller.mode).toBe(original.mode);
    expect(admitKuramotoResources(smaller, bounds, limit).allowed).toBe(true);
  }
  expect(smallerKuramotoRequest(original, bounds, policy({ memoryBytes: 0n }))).toBeNull();
  expect(smallerKuramotoRequest({ n: 1, steps: 1, mode: "mean-field" }, bounds, policy({ memoryBytes: 0n }))).toBeNull();
});


it("refuses a wall-clock request before real WASM allocation", () => {
  let entered = 0;
  const observed = { ...exports, scpn_alloc(length: number) { entered++; return exports.scpn_alloc(length); } };
  const result = bindKuramoto(observed, policy(), 1n)(request());
  expect(result.ok).toBe(false);
  expect(entered).toBe(0);
});
