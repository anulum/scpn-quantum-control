// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Kuramoto Play kernel tests against the real WASM (ST-11)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import committedScenarioJson from "../../../data/studio/kuramoto_scenario_meanfield_20260708.json";
import {
  type KernelSimulate,
  type KuramotoBounds,
  type KuramotoExports,
  type KuramotoRequest,
  bindKuramoto,
  committedScenario,
  encodeKuramotoInput,
  fetchKuramoto,
  instantiateKuramoto,
  maxOrderParameterDeviation,
  parseScenario,
  readBounds,
} from "./kuramoto";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
);

// The WASM kernel is the Rust integrator; the committed expectation is the
// Python reference (numpy). Their float op-order differs, so R(t) agrees within
// a visualisation tolerance, not bit-for-bit.
const KERNEL_VS_REFERENCE_TOL = 1e-6;

let wasmBytes: ArrayBuffer;
let simulate: KernelSimulate;
let bounds: KuramotoBounds;

function scenarioRequest(): KuramotoRequest {
  if (!committedScenario.ok) {
    throw new Error(`committed scenario failed to parse: ${committedScenario.reason}`);
  }
  const s = committedScenario.value;
  return {
    mode: s.mode,
    omega: s.omega,
    theta0: s.theta0,
    steps: s.steps,
    dt: s.dt,
    coupling: s.coupling,
  };
}

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  wasmBytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  const loaded = await instantiateKuramoto(wasmBytes);
  simulate = loaded.simulate;
  bounds = loaded.bounds;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("committed Kuramoto scenario", () => {
  it("parses and exposes the declared fail-closed bounds", () => {
    expect(committedScenario.ok).toBe(true);
    expect(bounds.maxOscillators).toBe(128);
    expect(bounds.maxSteps).toBe(4096);
    if (committedScenario.ok) {
      expect(committedScenario.value.boundaries).toEqual(bounds);
    }
  });

  it("replays through the real kernel within tolerance of the committed expectation", () => {
    const result = simulate(scenarioRequest());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(committedScenario.ok).toBe(true);
    if (!committedScenario.ok) return;
    const deviation = maxOrderParameterDeviation(
      result.run,
      committedScenario.value.expectedOrderParameter,
    );
    expect(deviation).toBeLessThan(KERNEL_VS_REFERENCE_TOL);
    // the scenario visibly synchronises
    const r = result.run.orderParameter;
    expect(r[r.length - 1]!).toBeGreaterThan(r[0]!);
  });

  it("agrees with the mean-field kernel via an explicit uniform network", () => {
    const request = scenarioRequest();
    const n = request.omega.length;
    const kNm: number[] = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        kNm.push(i === j ? 0 : request.coupling / n);
      }
    }
    const networked = simulate({ ...request, mode: "networked", kNm });
    const mean = simulate(request);
    expect(networked.ok && mean.ok).toBe(true);
    if (!networked.ok || !mean.ok) return;
    expect(
      maxOrderParameterDeviation(networked.run, Array.from(mean.run.orderParameter)),
    ).toBeLessThan(1e-9);
  });
});

describe("fail-closed boundaries", () => {
  it("rejects a request past the declared oscillator boundary", () => {
    const n = bounds.maxOscillators + 1;
    const result = simulate({
      mode: "mean-field",
      omega: new Array(n).fill(0.1),
      theta0: new Array(n).fill(0.2),
      steps: 10,
      dt: 0.05,
      coupling: 1,
    });
    expect(result.ok).toBe(false);
  });

  it("rejects a request past the declared step boundary", () => {
    const result = simulate({
      mode: "mean-field",
      omega: [0.1, 0.2],
      theta0: [0.0, 0.5],
      steps: bounds.maxSteps + 1,
      dt: 0.01,
      coupling: 1,
    });
    expect(result.ok).toBe(false);
  });

  it("rejects malformed requests before the kernel runs", () => {
    const bad = simulate({
      mode: "mean-field",
      omega: [0.1, 0.2],
      theta0: [0.0],
      steps: 10,
      dt: 0.05,
      coupling: 1,
    });
    expect(bad.ok).toBe(false);
  });
});

describe("encodeKuramotoInput", () => {
  const base: KuramotoRequest = {
    mode: "mean-field",
    omega: [0.1, 0.2, 0.3],
    theta0: [0.0, 0.5, 1.0],
    steps: 10,
    dt: 0.05,
    coupling: 1,
  };

  it("packs the mean-field and networked layouts to the right length", () => {
    const mean = encodeKuramotoInput(base);
    expect(mean).not.toBeNull();
    expect(mean!.length).toBe(32 + 2 * 3 * 8);
    const kNm = new Array(9).fill(0.1);
    const net = encodeKuramotoInput({ ...base, mode: "networked", kNm });
    expect(net!.length).toBe(32 + (2 * 3 + 9) * 8);
  });

  it("fails closed on shape, matrix, and non-finite errors", () => {
    expect(encodeKuramotoInput({ ...base, theta0: [0.0] })).toBeNull();
    expect(encodeKuramotoInput({ ...base, omega: [] })).toBeNull();
    expect(encodeKuramotoInput({ ...base, steps: 0 })).toBeNull();
    expect(encodeKuramotoInput({ ...base, dt: 0 })).toBeNull();
    expect(encodeKuramotoInput({ ...base, mode: "networked" })).toBeNull();
    expect(
      encodeKuramotoInput({ ...base, mode: "networked", kNm: new Array(4).fill(0) }),
    ).toBeNull();
    expect(encodeKuramotoInput({ ...base, kNm: [1, 2, 3] })).toBeNull();
    expect(encodeKuramotoInput({ ...base, coupling: Number.NaN })).toBeNull();
    expect(encodeKuramotoInput({ ...base, omega: [0.1, Number.NaN, 0.3] })).toBeNull();
  });
});

describe("binding and loading edge cases", () => {
  function stubExports(overrides: Partial<KuramotoExports> = {}): KuramotoExports {
    const memory = new WebAssembly.Memory({ initial: 1 });
    return {
      memory,
      scpn_alloc: () => 16,
      scpn_free: () => undefined,
      scpn_kuramoto_simulate: () => 0,
      scpn_kuramoto_max_oscillators: () => 128,
      scpn_kuramoto_max_steps: () => 4096,
      ...overrides,
    };
  }

  it("surfaces an input allocation failure", () => {
    const run = bindKuramoto(stubExports({ scpn_alloc: () => 0 }))({
      mode: "mean-field",
      omega: [0.1],
      theta0: [0.2],
      steps: 2,
      dt: 0.1,
      coupling: 1,
    });
    expect(run).toEqual({ ok: false, reason: "input allocation failed" });
  });

  it("surfaces an output allocation failure", () => {
    let calls = 0;
    const run = bindKuramoto(
      stubExports({
        scpn_alloc: () => {
          calls += 1;
          return calls === 1 ? 16 : 0;
        },
      }),
    )({ mode: "mean-field", omega: [0.1], theta0: [0.2], steps: 2, dt: 0.1, coupling: 1 });
    expect(run).toEqual({ ok: false, reason: "output allocation failed" });
  });

  it("surfaces a negative kernel status", () => {
    const run = bindKuramoto(stubExports({ scpn_kuramoto_simulate: () => -4 }))({
      mode: "mean-field",
      omega: [0.1],
      theta0: [0.2],
      steps: 2,
      dt: 0.1,
      coupling: 1,
    });
    expect(run.ok).toBe(false);
  });

  it("rejects a malformed request inside the closure", () => {
    const run = bindKuramoto(stubExports())({
      mode: "mean-field",
      omega: [0.1],
      theta0: [],
      steps: 2,
      dt: 0.1,
      coupling: 1,
    });
    expect(run).toEqual({ ok: false, reason: "request is malformed" });
  });

  it("reads the declared bounds", () => {
    expect(readBounds(stubExports())).toEqual({ maxOscillators: 128, maxSteps: 4096 });
  });

  it("fetches and instantiates the deployed kernel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, arrayBuffer: async () => wasmBytes }) as Response),
    );
    const loaded = await fetchKuramoto("wasm/kernel.wasm");
    expect(loaded.bounds.maxOscillators).toBe(128);
  });

  it("throws when the kernel fetch fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503 }) as Response));
    await expect(fetchKuramoto()).rejects.toThrow("kernel fetch failed: 503");
  });
});

describe("scenario parsing and deviation", () => {
  it("fails closed on a malformed scenario artefact", () => {
    expect(parseScenario(null).ok).toBe(false);
    expect(parseScenario({ scenario: {}, expected: {} }).ok).toBe(false);
    const good = committedScenarioJson as Record<string, unknown>;
    expect(parseScenario({ ...good, artifact_id: 123 }).ok).toBe(false);
    // a valid mode but a non-numeric scenario field still fails closed
    const scenario = good["scenario"] as Record<string, unknown>;
    expect(
      parseScenario({ ...good, scenario: { ...scenario, n: "twelve" } }).ok,
    ).toBe(false);
    // malformed boundaries fail closed
    expect(parseScenario({ ...good, boundaries: { max_oscillators: 1 } }).ok).toBe(false);
  });

  it("reports an infinite deviation on a length mismatch", () => {
    const run = { orderParameter: new Float64Array([1, 2, 3]), thetaFinal: new Float64Array() };
    expect(maxOrderParameterDeviation(run, [1, 2])).toBe(Number.POSITIVE_INFINITY);
  });
});
