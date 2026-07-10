// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web 3D Lab verified phase-trajectory capture (ST-21)

/**
 * Full phase-trajectory capture over the shipped WASM Kuramoto kernel.
 *
 * The kernel's one-shot entry point returns only `[R(t) ; θ_final]`. The 3D
 * Lab needs every intermediate phase snapshot θ_i(t), so this module drives
 * the SAME kernel one RK4 step at a time, feeding each call's final phases
 * back in as the next call's initial state. The kernel accumulates raw
 * (unwrapped) phases and the state round-trips exactly through its canonical
 * little-endian encoding, so the chained capture is bit-identical to the
 * one-shot run — and this module PROVES that at runtime: after the capture it
 * replays the request through the kernel's one-shot path and fails closed on
 * any bit-level divergence in the order-parameter series or the final phases.
 * Nothing here re-implements the integrator in JavaScript.
 */

import type { KernelSimulate, KuramotoRequest } from "./kuramoto";

/** The Lab's own fail-closed oscillator bound (stricter than the kernel's). */
export const LAB_MAX_OSCILLATORS = 32;

/** The Lab's own fail-closed step bound (stricter than the kernel's). */
export const LAB_MAX_STEPS = 360;

/**
 * Worst tolerated |R_ts − R_kernel| between the TypeScript order-parameter
 * recomputation and the kernel's own series. The phases are bit-identical;
 * only the `sin`/`cos` implementations differ (Rust libm in the WASM module
 * versus the host's `Math`), each correctly rounded or within 1 ulp, so the
 * mean over ≤32 oscillators stays far inside this bound.
 */
export const ORDER_PARAMETER_TOLERANCE = 1e-12;

/** A captured trajectory: every phase snapshot plus the kernel's R(t). */
export interface PhaseTrajectory {
  readonly n: number;
  readonly steps: number;
  /** Row-major `(steps + 1) × n` phase snapshots, row 0 is the initial state. */
  readonly theta: Float64Array;
  /** The kernel-computed order parameter at every snapshot. */
  readonly orderParameter: Float64Array;
}

export type TrajectoryResult =
  | { readonly ok: true; readonly trajectory: PhaseTrajectory }
  | { readonly ok: false; readonly reason: string };

/** The complex mean of the phases: magnitude R and mean phase ψ per snapshot. */
export interface OrderParameterSeries {
  readonly r: Float64Array;
  readonly psi: Float64Array;
}

/**
 * Capture the full phase trajectory for a request, failing closed.
 *
 * The request is bounded by the Lab's own N/step limits, integrated one step
 * at a time through the given kernel closure, then verified bit-exactly
 * against the kernel's one-shot run of the same request. Any kernel
 * rejection, bound violation, or bit-level divergence surfaces as an explicit
 * reason — never a fabricated or silently truncated trajectory.
 */
export function captureTrajectory(
  simulate: KernelSimulate,
  request: KuramotoRequest,
): TrajectoryResult {
  const n = request.omega.length;
  if (n > LAB_MAX_OSCILLATORS) {
    return { ok: false, reason: `lab boundary: N ≤ ${LAB_MAX_OSCILLATORS}` };
  }
  if (request.steps > LAB_MAX_STEPS) {
    return { ok: false, reason: `lab boundary: steps ≤ ${LAB_MAX_STEPS}` };
  }

  const theta = new Float64Array((request.steps + 1) * n);
  const orderParameter = new Float64Array(request.steps + 1);
  let current: readonly number[] = request.theta0;
  theta.set(current, 0);
  for (let step = 0; step < request.steps; step += 1) {
    const result = simulate({ ...request, theta0: current, steps: 1 });
    if (!result.ok) {
      return { ok: false, reason: result.reason };
    }
    if (step === 0) {
      orderParameter[0] = result.run.orderParameter[0]!;
    }
    orderParameter[step + 1] = result.run.orderParameter[1]!;
    current = Array.from(result.run.thetaFinal);
    theta.set(current, (step + 1) * n);
  }

  const oneShot = simulate(request);
  if (!oneShot.ok) {
    return { ok: false, reason: oneShot.reason };
  }
  for (let index = 0; index <= request.steps; index += 1) {
    if (orderParameter[index] !== oneShot.run.orderParameter[index]) {
      return {
        ok: false,
        reason: `chained capture diverged from the one-shot kernel run at R(t) sample ${index}`,
      };
    }
  }
  const lastRow = request.steps * n;
  for (let index = 0; index < n; index += 1) {
    if (theta[lastRow + index] !== oneShot.run.thetaFinal[index]) {
      return {
        ok: false,
        reason: `chained capture diverged from the one-shot kernel run at final phase ${index}`,
      };
    }
  }

  return { ok: true, trajectory: { n, steps: request.steps, theta, orderParameter } };
}

/** Recompute R(t) and the mean phase ψ(t) in TypeScript from the snapshots. */
export function orderParameterSeries(trajectory: PhaseTrajectory): OrderParameterSeries {
  const { n, steps, theta } = trajectory;
  const r = new Float64Array(steps + 1);
  const psi = new Float64Array(steps + 1);
  for (let row = 0; row <= steps; row += 1) {
    let cosSum = 0;
    let sinSum = 0;
    for (let index = 0; index < n; index += 1) {
      const angle = theta[row * n + index]!;
      cosSum += Math.cos(angle);
      sinSum += Math.sin(angle);
    }
    const re = cosSum / n;
    const im = sinSum / n;
    // Mirror the kernel's own formula so sin/cos are the only divergence source.
    r[row] = Math.sqrt(re * re + im * im);
    psi[row] = Math.atan2(im, re);
  }
  return { r, psi };
}

/** Cross-language parity of the TypeScript R(t) against the kernel's series. */
export function kernelOrderParameterParity(
  trajectory: PhaseTrajectory,
  series: OrderParameterSeries,
): { readonly verified: boolean; readonly worst: number } {
  let worst = 0;
  for (let index = 0; index <= trajectory.steps; index += 1) {
    worst = Math.max(worst, Math.abs(series.r[index]! - trajectory.orderParameter[index]!));
  }
  return { verified: worst <= ORDER_PARAMETER_TOLERANCE, worst };
}
