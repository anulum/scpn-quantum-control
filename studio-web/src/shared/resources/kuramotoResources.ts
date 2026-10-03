// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

import type { KuramotoBounds, KuramotoMode } from "../../panel/kuramoto";
import { checkResourcePlan } from "./admission";
import type { ResourceAdmission, ResourceBuffer, ResourcePolicy } from "./admission";

/** Metadata supplied before the caller builds vectors, transfers or a worker. */
export interface KuramotoResourceRequest {
  /** Oscillators, checked against the actual module limit. */
  readonly n: number;
  /** Positive integration step count. */
  readonly steps: number;
  /** Original coupling method without fallback. */
  readonly mode: KuramotoMode;
}

/** Browser product ceilings for declared numeric payload, not discovered free RAM. */
export function browserResourcePolicy(bounds: KuramotoBounds): ResourcePolicy {
  validateBounds(bounds);
  return Object.freeze({
    source: "browser declared numeric-payload ceiling; allocator/object overhead excluded; kernel limits remain mandatory",
    addressableBytes: 0xffff_ffffn,
    memoryBytes: 4n * 1024n * 1024n,
    workUnits: 4n * BigInt(bounds.maxSteps) * BigInt(bounds.maxOscillators) ** 2n,
    overheadBytes: 0n,
  });
}

function count(value: number, name: string): bigint {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) throw new Error(`${name}: bounded positive count required`);
  return BigInt(value);
}

function validateBounds(bounds: KuramotoBounds): void {
  count(bounds.maxOscillators, "kernel oscillator limit");
  count(bounds.maxSteps, "kernel step limit");
}

/** Project the existing Rust RK4 buffer chain; its equations and ABI remain unchanged. */
export function admitKuramotoResources(
  request: KuramotoResourceRequest, bounds: KuramotoBounds,
  policy: ResourcePolicy = browserResourcePolicy(bounds), requestedWallMs: bigint | null = null,
): ResourceAdmission {
  validateBounds(bounds);
  const n = count(request.n, "oscillators");
  const steps = count(request.steps, "steps");
  if (request.n > bounds.maxOscillators || request.steps > bounds.maxSteps) throw new Error("request exceeds declared kernel bounds");
  if (request.mode !== "mean-field" && request.mode !== "networked") throw new Error("unsupported Kuramoto method");
  const matrix = request.mode === "networked" ? n * n : 0n;
  const payload = 2n * n + matrix;
  const output = steps + 1n + n;
  const buffers: ResourceBuffer[] = [
    { name: "caller_numeric_values", role: "intermediate", shape: [payload], dtype: "float64", count: 1n },
    { name: "encoded_host_and_guest_input", role: "transfer", shape: [32n + 8n * payload], dtype: "uint8", count: 2n },
    { name: "parsed_kernel_input", role: "intermediate", shape: [payload], dtype: "float64", count: 1n },
    { name: "theta_rk4_stages", role: "intermediate", shape: [n], dtype: "float64", count: 6n },
    { name: "kernel_result_guest_output_host_result", role: "transfer", shape: [output], dtype: "float64", count: 3n },
  ];
  return checkResourcePlan({
    backend: "shipped-kuramoto-wasm-float64", method: request.mode, buffers,
    concurrency: 1n, workUnits: 4n * steps * (request.mode === "networked" ? n * n : n),
  }, policy, requestedWallMs);
}

/** Admit original numeric buffers plus explicit worker transport copies before allocation. */
export function admitOwnedKuramotoResources(
  request: KuramotoResourceRequest, bounds: KuramotoBounds, binaryBytes: number,
  policy: ResourcePolicy = browserResourcePolicy(bounds), requestedWallMs: bigint | null = null,
): ResourceAdmission {
  if (!Number.isSafeInteger(binaryBytes) || binaryBytes < 1 || binaryBytes > 2 * 1024 * 1024) throw new Error("bounded worker binary bytes required");
  const original = admitKuramotoResources(request, bounds, policy);
  const n = BigInt(request.n);
  const values = 2n * n + (request.mode === "networked" ? n * n : 0n);
  const buffers: ResourceBuffer[] = original.estimate.components.map(component => ({
    name: component.name, role: component.role, shape: component.shape,
    dtype: component.dtype, count: component.count,
  }));
  buffers.push(
    { name: "owned_worker_transfer_codec_vectors_and_source_request", role: "transfer", shape: [values], dtype: "float64", count: 3n },
    { name: "parent_source_codec_validation", role: "transfer", shape: [32n + 8n * values], dtype: "uint8", count: 1n },
    { name: "retained_and_transferred_kernel_binary", role: "transfer", shape: [BigInt(binaryBytes)], dtype: "uint8", count: 2n },
  );
  return checkResourcePlan({
    backend: "shipped-kuramoto-wasm-float64-owned-worker", method: request.mode,
    buffers, concurrency: 1n, workUnits: original.estimate.workUnits,
  }, policy, requestedWallMs);
}


/** Offer an explicitly applied smaller shape after checking the complete same-method plan. */
export function smallerKuramotoRequest(
  request: KuramotoResourceRequest, bounds: KuramotoBounds, policy: ResourcePolicy,
  binaryBytes?: number,
): KuramotoResourceRequest | null {
  count(request.n, "oscillators");
  count(request.steps, "steps");
  validateBounds(bounds);
  let n = Math.min(request.n, bounds.maxOscillators);
  let steps = Math.min(request.steps, bounds.maxSteps);
  while (n > 1 || steps > 1) {
    n = Math.max(1, Math.floor(n / 2));
    steps = Math.max(1, Math.floor(steps / 2));
    const candidate = Object.freeze({ n, steps, mode: request.mode });
    const admission = binaryBytes === undefined ? admitKuramotoResources(candidate, bounds, policy)
      : admitOwnedKuramotoResources(candidate, bounds, binaryBytes, policy);
    if (admission.allowed) return candidate;
  }
  return null;
}
