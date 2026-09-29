// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — declared resource admission regressions

// @vitest-environment node
import sharedCases from "../../../../data/studio/resource_plan_contracts.json";
import { describe, expect, it } from "vitest";
import { checkResourcePlan, estimateResourcePlan, hilbertBuffer } from "./admission";
import type { ResourceBuffer, ResourcePlan, ResourcePolicy } from "./admission";

const addressable = (1n << 63n) - 1n;
function buffer(overrides: Partial<ResourceBuffer> = {}): ResourceBuffer {
  return { name: "state", role: "statevector", shape: [8n], dtype: "complex128", count: 1n, ...overrides };
}
function plan(overrides: Partial<ResourcePlan> = {}): ResourcePlan {
  return { backend: "declared-test-backend", method: "declared-state", buffers: [buffer()], concurrency: 1n, workUnits: 100n, ...overrides };
}
function policy(overrides: Partial<ResourcePolicy> = {}): ResourcePolicy {
  return { source: "injected-declared-policy", addressableBytes: addressable, memoryBytes: 128n, workUnits: 100n, overheadBytes: 0n, ...overrides };
}

describe("public resource estimate and admission", () => {
  it("matches declared statevector, density and adjoint formulas with separate transfer/layout", () => {
    const buffers = [
      hilbertBuffer("state", "statevector", 3n, 1n, "complex128", 1n, addressable),
      hilbertBuffer("density", "density", 3n, 2n, "complex128", 1n, addressable),
      hilbertBuffer("tape", "adjoint", 3n, 1n, "complex128", 4n, addressable),
      buffer({ name: "transfer", role: "transfer", shape: [32n], dtype: "uint8", count: 2n }),
      buffer({ name: "layout", role: "graph", shape: [3n, 2n], dtype: "float64" }),
    ];
    const estimate = estimateResourcePlan(plan({ buffers, concurrency: 2n }), addressable);
    expect(estimate.components.map(row => row.bytes)).toEqual([128n, 1024n, 512n, 64n, 48n]);
    expect(estimate.components[1]?.shape).toEqual([8n, 8n]);
    expect(estimate.payloadBytes).toBe(3552n);
    expect(estimate.workUnits).toBe(200n);
  });

  it("allows the exact declared ceiling and refuses the next byte or work unit", () => {
    expect(checkResourcePlan(plan(), policy()).allowed).toBe(true);
    expect(checkResourcePlan(plan(), policy({ memoryBytes: 127n })).blockers).toEqual(["declared_storage_exceeds_budget"]);
    expect(checkResourcePlan(plan(), policy({ workUnits: 99n })).blockers).toEqual(["declared_work_exceeds_budget"]);
    expect(checkResourcePlan(plan(), policy({ memoryBytes: 0n })).allowed).toBe(false);
  });

  it("recalculates precision, copies and concurrency without implicit precision reduction", () => {
    const precision = ["float32", "float64", "complex64", "complex128"] as const;
    expect(precision.map(dtype => estimateResourcePlan(plan({ buffers: [buffer({ dtype, count: 2n })], concurrency: 3n }), addressable).payloadBytes)).toEqual([192n, 384n, 384n, 768n]);
    expect(checkResourcePlan(plan({ buffers: [buffer({ dtype: "complex64" })] }), policy()).estimate.components[0]?.dtype).toBe("complex64");
  });

  it("preserves unknown memory, work and backend overhead as refusals", () => {
    const refusal = checkResourcePlan(plan({ workUnits: null }), policy({ memoryBytes: null, workUnits: null, overheadBytes: null }));
    expect(refusal.allowed).toBe(false);
    expect(refusal.bytesRequired).toBeNull();
    expect(refusal.blockers).toEqual(["memory_limit_unknown", "work_limit_unknown", "backend_overhead_unknown"]);
    expect(checkResourcePlan(plan(), policy({ overheadBytes: 1n })).bytesRequired).toBe(129n);
    expect(checkResourcePlan(plan(), policy({ overheadBytes: addressable })).blockers).toEqual(["total_storage_exceeds_addressability"]);
    expect(refusal.claimBoundary).toContain("original backend admission remains required");
  });

  it("refuses unsafe shifts and products before any dense buffer exists", () => {
    expect(() => hilbertBuffer("large", "statevector", 10n ** 100n, 1n, "complex128", 1n, addressable)).toThrow("exponent");
    expect(() => hilbertBuffer("large", "statevector", 60n, 1n, "complex128", 1n, addressable)).toThrow("addressability");
    expect(() => hilbertBuffer("large", "statevector", 1n, 65n, "complex128", 1n, addressable)).toThrow("rank");
    expect(() => estimateResourcePlan(plan({ buffers: [buffer({ shape: [addressable] })] }), addressable)).toThrow("addressability");
    expect(() => estimateResourcePlan(plan({ buffers: [buffer({ shape: [1n], dtype: "uint8", count: addressable }), buffer({ name: "extra" })] }), addressable)).toThrow("addressability");
    expect(() => estimateResourcePlan(plan({ concurrency: addressable }), addressable)).toThrow("addressability");
    expect(estimateResourcePlan(plan({ workUnits: addressable, concurrency: 2n }), addressable).workUnits).toBe(addressable * 2n);
  });

  it("snapshots caller-owned fields and policy without changing numerical authority", () => {
    const shape = [8n];
    const supplied = policy();
    const request = plan({ buffers: [buffer({ shape })] });
    const result = checkResourcePlan(request, supplied);
    shape[0] = 999n;
    Object.assign(supplied, { memoryBytes: 0n });
    expect(result.estimate.components[0]?.shape).toEqual([8n]);
    expect(result.policy.memoryBytes).toBe(128n);
    expect(Object.isFrozen(result.estimate.components[0]?.shape)).toBe(true);
    expect(Object.isFrozen(result.blockers)).toBe(true);
    expect(result.estimate.backend).toBe(request.backend);
  });

  it("rejects malformed declarations through the public estimate boundary", () => {
    const cases: ResourcePlan[] = [
      plan({ buffers: [] }), plan({ buffers: Array.from({ length: 1001 }, () => buffer()) }),
      plan({ buffers: [buffer(), buffer()] }), plan({ backend: " " }), plan({ method: "" }),
      plan({ concurrency: 0n }), plan({ workUnits: 0n }),
      plan({ buffers: [buffer({ shape: [] })] }),
      plan({ buffers: [buffer({ shape: Array.from({ length: 65 }, () => 1n) })] }),
      plan({ buffers: [buffer({ shape: [0n] })] }), plan({ buffers: [buffer({ count: 0n })] }),
      plan({ buffers: [buffer({ name: "" })] }),
      plan({ buffers: [buffer({ dtype: "unknown" as ResourceBuffer["dtype"] })] }),
      plan({ buffers: [buffer({ role: "unknown" as ResourceBuffer["role"] })] }),
    ];
    for (const request of cases) expect(() => estimateResourcePlan(request, addressable)).toThrow();
    for (const supplied of [policy({ source: "" }), policy({ addressableBytes: 0n }), policy({ memoryBytes: -1n }), policy({ overheadBytes: -1n }), policy({ workUnits: -1n })]) {
      expect(() => checkResourcePlan(plan(), supplied)).toThrow();
    }
  });

  it("refuses data getters without executing them", () => {
    let calls = 0;
    const request = plan();
    Object.defineProperty(request, "concurrency", { enumerable: true, get() { calls++; return 1n; } });
    expect(() => estimateResourcePlan(request, addressable)).toThrow("data member");
    expect(calls).toBe(0);
  });
});


it("shares exact formula cases with the existing Python resource owner", () => {
  expect(sharedCases.schema).toBe("studio.resource-plan.conformance.v1");
  for (const row of sharedCases.cases) {
    const limit = BigInt(sharedCases.addressable_bytes);
    const buffer = hilbertBuffer(row.name, "statevector", BigInt(row.qubits), BigInt(row.rank), row.dtype as ResourceBuffer["dtype"], BigInt(row.count), limit);
    const estimate = estimateResourcePlan(plan({ buffers: [buffer], concurrency: BigInt(row.concurrency) }), limit);
    expect(estimate.payloadBytes.toString()).toBe(row.bytes);
  }
});


it("rejects decorated and accessor metadata before payload allocation", () => {
  const request = plan();
  Object.assign(request, { unexpected: { deeply: "nested" } });
  expect(() => estimateResourcePlan(request, addressable)).toThrow("record fields");
  expect(() => estimateResourcePlan(plan(), 1n << 100n)).toThrow("64 bits");
  const shape = [8n];
  let read = 0;
  Object.defineProperty(shape, "0", { get() { read++; return 8n; }, enumerable: true });
  expect(() => estimateResourcePlan(plan({ buffers: [buffer({ shape })] }), addressable)).toThrow("data member");
  expect(read).toBe(0);
  const custom = [buffer()];
  Object.setPrototypeOf(custom, { map() { throw new Error("custom map invoked"); } });
  expect(() => estimateResourcePlan(plan({ buffers: custom }), addressable)).toThrow("plain resource array");
});


it("refuses a requested wall-clock guarantee instead of fabricating a time estimate", () => {
  const refused = checkResourcePlan(plan(), policy(), 1n);
  expect(refused.allowed).toBe(false);
  expect(refused.requestedWallMs).toBe(1n);
  expect(refused.blockers).toContain("wall_clock_admission_unavailable");
  expect(() => checkResourcePlan(plan(), policy(), 0n)).toThrow("deadline");
});


it("rejects malformed scalar and envelope forms through the public resource boundary", () => {
  expect(() => estimateResourcePlan(plan({ concurrency: 1 as unknown as bigint }), addressable)).toThrow("integer");
  expect(() => estimateResourcePlan(plan({ backend: true as unknown as string }), addressable)).toThrow("identity");
  expect(() => estimateResourcePlan(plan({ backend: "x".repeat(4097) }), addressable)).toThrow("identity");
  expect(() => estimateResourcePlan(null as unknown as ResourcePlan, addressable)).toThrow("record fields");
  const { method, ...remaining } = plan();
  expect(() => estimateResourcePlan({ ...remaining, unrecognised: method } as unknown as ResourcePlan, addressable)).toThrow("unsupported resource record field");
  const inherited = plan();
  Object.setPrototypeOf(inherited, { backend: "ambient" });
  expect(() => estimateResourcePlan(inherited, addressable)).toThrow("unsupported object");
  const decorated = [8n];
  Object.assign(decorated, { extra: 1n });
  expect(() => estimateResourcePlan(plan({ buffers: [buffer({ shape: decorated })] }), addressable)).toThrow("array");
  const sparse = [8n, 8n];
  delete sparse[1];
  Object.assign(sparse, { extra: 1n });
  expect(() => estimateResourcePlan(plan({ buffers: [buffer({ shape: sparse })] }), addressable)).toThrow("data member");
  expect(() => hilbertBuffer("invalid", "statevector", 100n, 1n, "complex128", 1n, 1n << 200n)).toThrow("64 bits");
});
