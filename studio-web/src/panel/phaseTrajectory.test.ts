// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — 3D Lab verified trajectory-capture tests (ST-21)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

import type { KernelSimulate, KuramotoRequest } from "./kuramoto";
import { instantiateKuramoto } from "./kuramoto";
import {
  LAB_MAX_OSCILLATORS,
  LAB_MAX_STEPS,
  ORDER_PARAMETER_TOLERANCE,
  captureTrajectory,
  kernelOrderParameterParity,
  orderParameterSeries,
} from "./phaseTrajectory";

const WASM_PATH = resolve(
  "..",
  "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
);

let simulate: KernelSimulate;

beforeAll(async () => {
  const buffer = readFileSync(WASM_PATH);
  const bytes = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength,
  ) as ArrayBuffer;
  ({ simulate } = await instantiateKuramoto(bytes));
});

function meanFieldRequest(n: number, steps: number): KuramotoRequest {
  return {
    mode: "mean-field",
    omega: Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1)),
    theta0: Array.from({ length: n }, (_, i) => (3 * i) / (n - 1)),
    steps,
    dt: 0.02,
    coupling: 2.5,
  };
}

describe("captureTrajectory with the real kernel", () => {
  it("is bit-identical to the kernel's one-shot run (mean-field)", () => {
    const request = meanFieldRequest(8, 60);
    const captured = captureTrajectory(simulate, request);
    expect(captured.ok).toBe(true);
    if (!captured.ok) return;
    // independent external check, on top of the module's internal one
    const oneShot = simulate(request);
    expect(oneShot.ok).toBe(true);
    if (!oneShot.ok) return;
    expect(Array.from(captured.trajectory.orderParameter)).toEqual(
      Array.from(oneShot.run.orderParameter),
    );
    const lastRow = captured.trajectory.steps * captured.trajectory.n;
    expect(Array.from(captured.trajectory.theta.slice(lastRow))).toEqual(
      Array.from(oneShot.run.thetaFinal),
    );
    // every intermediate snapshot was captured
    expect(captured.trajectory.theta).toHaveLength(61 * 8);
    // row 0 is the initial state verbatim
    expect(Array.from(captured.trajectory.theta.slice(0, 8))).toEqual(request.theta0.slice());
  });

  it("captures the networked kernel identically", () => {
    const n = 4;
    const base = meanFieldRequest(n, 30);
    const kNm: number[] = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        kNm.push(i === j ? 0 : base.coupling / n);
      }
    }
    const request: KuramotoRequest = { ...base, mode: "networked", kNm };
    const captured = captureTrajectory(simulate, request);
    expect(captured.ok).toBe(true);
    if (!captured.ok) return;
    const oneShot = simulate(request);
    expect(oneShot.ok).toBe(true);
    if (!oneShot.ok) return;
    expect(Array.from(captured.trajectory.orderParameter)).toEqual(
      Array.from(oneShot.run.orderParameter),
    );
  });

  it("keeps the TypeScript R(t) recomputation inside the parity tolerance", () => {
    const captured = captureTrajectory(simulate, meanFieldRequest(12, 100));
    expect(captured.ok).toBe(true);
    if (!captured.ok) return;
    const series = orderParameterSeries(captured.trajectory);
    const parity = kernelOrderParameterParity(captured.trajectory, series);
    expect(parity.verified).toBe(true);
    expect(parity.worst).toBeLessThanOrEqual(ORDER_PARAMETER_TOLERANCE);
    // the mean phase is finite everywhere
    for (const psi of series.psi) {
      expect(Number.isFinite(psi)).toBe(true);
    }
  });

  it("propagates a kernel rejection instead of fabricating a run", () => {
    const request = { ...meanFieldRequest(4, 20), dt: -1 };
    const captured = captureTrajectory(simulate, request);
    expect(captured.ok).toBe(false);
    if (captured.ok) return;
    expect(captured.reason).toContain("malformed");
  });
});

describe("captureTrajectory lab boundaries", () => {
  it("fails closed past the Lab oscillator bound", () => {
    const captured = captureTrajectory(simulate, meanFieldRequest(LAB_MAX_OSCILLATORS + 1, 20));
    expect(captured).toEqual({ ok: false, reason: `lab boundary: N ≤ ${LAB_MAX_OSCILLATORS}` });
  });

  it("fails closed past the Lab step bound", () => {
    const captured = captureTrajectory(simulate, meanFieldRequest(4, LAB_MAX_STEPS + 1));
    expect(captured).toEqual({ ok: false, reason: `lab boundary: steps ≤ ${LAB_MAX_STEPS}` });
  });
});

describe("captureTrajectory divergence detection", () => {
  const n = 3;
  const request = meanFieldRequest(n, 5);

  it("fails closed when the one-shot R(t) disagrees with the chain", () => {
    const stub: KernelSimulate = (req) =>
      req.steps === 1
        ? {
            ok: true,
            run: {
              orderParameter: new Float64Array([0.5, 0.5]),
              thetaFinal: new Float64Array(n),
            },
          }
        : {
            ok: true,
            run: {
              orderParameter: new Float64Array(req.steps + 1).fill(0.6),
              thetaFinal: new Float64Array(n),
            },
          };
    const captured = captureTrajectory(stub, request);
    expect(captured.ok).toBe(false);
    if (captured.ok) return;
    expect(captured.reason).toContain("diverged");
    expect(captured.reason).toContain("R(t) sample 0");
  });

  it("fails closed when the one-shot final phases disagree with the chain", () => {
    const stub: KernelSimulate = (req) => ({
      ok: true,
      run: {
        orderParameter: new Float64Array(req.steps + 1).fill(0.5),
        thetaFinal: new Float64Array(n).fill(req.steps === 1 ? 0 : 1),
      },
    });
    const captured = captureTrajectory(stub, request);
    expect(captured.ok).toBe(false);
    if (captured.ok) return;
    expect(captured.reason).toContain("final phase 0");
  });

  it("fails closed when only the verification one-shot is rejected", () => {
    const stub: KernelSimulate = (req) =>
      req.steps === 1
        ? {
            ok: true,
            run: {
              orderParameter: new Float64Array([0.5, 0.5]),
              thetaFinal: new Float64Array(n),
            },
          }
        : { ok: false, reason: "one-shot refused" };
    const captured = captureTrajectory(stub, request);
    expect(captured).toEqual({ ok: false, reason: "one-shot refused" });
  });
});

describe("orderParameterSeries", () => {
  it("reports full coherence for phase-locked snapshots", () => {
    const series = orderParameterSeries({
      n: 3,
      steps: 0,
      theta: new Float64Array([0.7, 0.7, 0.7]),
      orderParameter: new Float64Array([1]),
    });
    expect(series.r[0]).toBeCloseTo(1, 12);
    expect(series.psi[0]).toBeCloseTo(0.7, 12);
  });

  it("reports zero coherence for a splayed snapshot", () => {
    const series = orderParameterSeries({
      n: 4,
      steps: 0,
      theta: new Float64Array([0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]),
      orderParameter: new Float64Array([0]),
    });
    expect(series.r[0]!).toBeLessThan(1e-12);
  });
});
