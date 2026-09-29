// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio declared resource admission

import { dataEntries, scalarString } from "../contracts/canonical";

/** Numeric storage spellings supported by the declared byte projection. */
export type ResourceDtype = "float32" | "float64" | "complex64" | "complex128" | "uint8";
/** Separate owners of simultaneously retained storage. */
export type ResourceRole = "statevector" | "density" | "intermediate" | "adjoint" | "transfer" | "graph";

/** A fixed-width buffer declaration; it does not allocate numerical storage. */
export interface ResourceBuffer {
  /** Unique buffer identity within this plan. */
  readonly name: string;
  /** Owner of the declared storage component. */
  readonly role: ResourceRole;
  /** Positive dimensions in their declared order. */
  readonly shape: readonly bigint[];
  /** Fixed-width storage precision; never silently converted. */
  readonly dtype: ResourceDtype;
  /** Number of simultaneously live copies. */
  readonly count: bigint;
}

/** Complete declared payload and concurrency with an explicit backend identity. */
export interface ResourcePlan {
  /** Producer or backend identity supplied by the caller. */
  readonly backend: string;
  /** Execution method preserved in the receipt. */
  readonly method: string;
  /** Complete declared live-buffer set. */
  readonly buffers: readonly ResourceBuffer[];
  /** Simultaneous copies of the complete job. */
  readonly concurrency: bigint;
  /** Declared workload units, never inferred seconds or benchmark throughput. */
  readonly workUnits: bigint | null;
}

/** An explicitly supplied ceiling and backend overhead declaration. */
export interface ResourcePolicy {
  /** Origin and limitations of the supplied ceiling. */
  readonly source: string;
  /** Explicit addressability ceiling, at most 64 bits. */
  readonly addressableBytes: bigint;
  /** Declared byte ceiling; null preserves missing capacity. */
  readonly memoryBytes: bigint | null;
  /** Declared work quantity or ceiling; null preserves unknown work. */
  readonly workUnits: bigint | null;
  /** Total extra live bytes per job; null preserves unknown overhead. */
  readonly overheadBytes: bigint | null;
}

/** Readonly component receipt, retaining its declared units and precision. */
export interface ResourceComponent extends ResourceBuffer {
  /** Exact component byte demand before job concurrency. */
  readonly bytes: bigint;
}

/** Projected demand with a snapshot of every caller-owned declaration. */
export interface ResourceEstimate {
  /** Producer or backend identity supplied by the caller. */
  readonly backend: string;
  /** Execution method preserved in the receipt. */
  readonly method: string;
  /** Immutable snapshots of each declared storage component. */
  readonly components: readonly ResourceComponent[];
  /** Simultaneous copies of the complete job. */
  readonly concurrency: bigint;
  /** Exact sum including job concurrency. */
  readonly payloadBytes: bigint;
  /** Declared work quantity or ceiling; null preserves unknown work. */
  readonly workUnits: bigint | null;
}

/** A byte/work policy verdict; it cannot grant numerical backend support. */
export interface ResourceAdmission {
  /** Whether every declared byte and work check passed. */
  readonly allowed: boolean;
  /** Complete declaration underlying this verdict. */
  readonly estimate: ResourceEstimate;
  /** Immutable source policy used for the decision. */
  readonly policy: ResourcePolicy;
  /** Total declaration or unknown on incomplete/overflow data. */
  readonly bytesRequired: bigint | null;
  /** Observed refusals in deterministic order. */
  readonly blockers: readonly string[];
  /** Declaration limits; original runtime admission remains mandatory. */
  readonly claimBoundary: string;
  /** Requested wall-clock admission; null means no deadline was requested. */
  readonly requestedWallMs: bigint | null;
}

const itemBytes: Readonly<Record<ResourceDtype, bigint>> = {
  float32: 4n, float64: 8n, complex64: 8n, complex128: 16n, uint8: 1n,
};
const roles: readonly ResourceRole[] = ["statevector", "density", "intermediate", "adjoint", "transfer", "graph"];
const boundary = "Declared payload and workload policy only; original backend admission remains required. No host capacity, elapsed time, reservation or OOM guarantee.";

function integer(value: bigint, name: string, allowZero = false): bigint {
  if (typeof value !== "bigint" || value < (allowZero ? 0n : 1n)) {
    throw new Error(`${name}: ${allowZero ? "nonnegative" : "positive"} integer required`);
  }
  return value;
}

function label(value: string, name: string): string {
  if (typeof value !== "string" || !value.trim() || value.length > 4096) throw new Error(`${name}: bounded source identity required`);
  return scalarString(value, name);
}

function record(value: object, keys: readonly string[]): void {
  if (typeof value !== "object" || value === null || Reflect.ownKeys(value).length !== keys.length) throw new Error("resource record fields are incomplete or unsupported");
  const entries = dataEntries(value);
  if (entries.some(([key]) => !keys.includes(key))) throw new Error("unsupported resource record field");
}

function array(value: readonly unknown[], minimum: number, maximum: number): void {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || value.length < minimum || value.length > maximum || Reflect.ownKeys(value).length !== value.length + 1) throw new Error("bounded plain resource array required");
  for (let index = 0; index < value.length; index++) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor || !("value" in descriptor)) throw new Error("resource array data member required");
  }
}

function product(left: bigint, right: bigint, limit: bigint): bigint {
  if (right > limit / left) throw new Error("declared storage exceeds addressability");
  return left * right;
}

/** Build a Hilbert buffer, checking exponent and byte addressability before shifting. */
export function hilbertBuffer(
  name: string, role: ResourceRole, qubits: bigint, rank: bigint,
  dtype: ResourceDtype, count: bigint, addressableBytes: bigint,
): ResourceBuffer {
  integer(addressableBytes, "addressableBytes");
  if (addressableBytes > 0xffff_ffff_ffff_ffffn) throw new Error("addressable declaration exceeds 64 bits");
  integer(qubits, "qubits");
  integer(rank, "rank");
  if (rank > 64n) throw new Error("Hilbert rank exceeds metadata depth");
  integer(count, "count");
  const exponent = qubits * rank;
  if (exponent >= BigInt(addressableBytes.toString(2).length)) {
    throw new Error("Hilbert exponent exceeds addressability");
  }
  const buffer = { name, role, shape: Array.from({ length: Number(rank) }, () => 1n << qubits), dtype, count };
  estimateResourcePlan({ backend: "declared", method: "Hilbert", buffers: [buffer], concurrency: 1n, workUnits: null }, addressableBytes);
  return Object.freeze({ ...buffer, shape: Object.freeze(buffer.shape) });
}

/** Sum fixed-width live declarations exactly without constructing their buffers. */
export function estimateResourcePlan(plan: ResourcePlan, addressableBytes: bigint): ResourceEstimate {
  record(plan, ["backend", "method", "buffers", "concurrency", "workUnits"]);
  const limit = integer(addressableBytes, "addressableBytes");
  if (limit > 0xffff_ffff_ffff_ffffn) throw new Error("addressable declaration exceeds 64 bits");
  const concurrency = integer(plan.concurrency, "concurrency");
  label(plan.backend, "backend");
  label(plan.method, "method");
  array(plan.buffers, 1, 1000);
  const names = new Set<string>();
  let total = 0n;
  const components = plan.buffers.map(buffer => {
    record(buffer, ["name", "role", "shape", "dtype", "count"]);
    label(buffer.name, "buffer name");
    if (names.has(buffer.name)) throw new Error("duplicate buffer name");
    names.add(buffer.name);
    if (!roles.includes(buffer.role)) throw new Error("unsupported buffer role");
    if (!Object.hasOwn(itemBytes, buffer.dtype)) throw new Error("unsupported fixed-width dtype");
    array(buffer.shape, 1, 64);
    let bytes = itemBytes[buffer.dtype];
    const shape = buffer.shape.map(dimension => {
      integer(dimension, "dimension");
      bytes = product(bytes, dimension, limit);
      return dimension;
    });
    bytes = product(bytes, integer(buffer.count, "count"), limit);
    if (bytes > limit - total) throw new Error("declared storage exceeds addressability");
    total += bytes;
    return Object.freeze({ name: buffer.name, role: buffer.role, dtype: buffer.dtype, count: buffer.count, shape: Object.freeze(shape), bytes });
  });
  const payloadBytes = product(total, concurrency, limit);
  const workUnits = plan.workUnits === null ? null : integer(plan.workUnits, "workUnits") * concurrency;
  return Object.freeze({ backend: plan.backend, method: plan.method, components: Object.freeze(components), concurrency, payloadBytes, workUnits });
}

/** Reproject the complete plan against supplied policy; unknown limits refuse. */
export function checkResourcePlan(plan: ResourcePlan, supplied: ResourcePolicy, requestedWallMs: bigint | null = null): ResourceAdmission {
  record(supplied, ["source", "addressableBytes", "memoryBytes", "workUnits", "overheadBytes"]);
  label(supplied.source, "policy source");
  const limit = integer(supplied.addressableBytes, "addressableBytes");
  const memoryBytes = supplied.memoryBytes === null ? null : integer(supplied.memoryBytes, "memoryBytes", true);
  const workLimit = supplied.workUnits === null ? null : integer(supplied.workUnits, "work ceiling", true);
  const overhead = supplied.overheadBytes === null ? null : integer(supplied.overheadBytes, "overheadBytes", true);
  const policy = Object.freeze({ source: supplied.source, addressableBytes: limit, memoryBytes, workUnits: workLimit, overheadBytes: overhead });
  const estimate = estimateResourcePlan(plan, limit);
  const blockers: string[] = [];
  if (requestedWallMs !== null) {
    integer(requestedWallMs, "wall-clock deadline");
    blockers.push("wall_clock_admission_unavailable");
  }
  if (memoryBytes === null) blockers.push("memory_limit_unknown");
  if (workLimit === null || estimate.workUnits === null) blockers.push("work_limit_unknown");
  if (overhead === null) blockers.push("backend_overhead_unknown");
  let bytesRequired: bigint | null = null;
  if (overhead !== null) {
    if (overhead > (limit - estimate.payloadBytes) / estimate.concurrency) {
      blockers.push("total_storage_exceeds_addressability");
    } else {
      bytesRequired = estimate.payloadBytes + overhead * estimate.concurrency;
      if (memoryBytes !== null && bytesRequired > memoryBytes) blockers.push("declared_storage_exceeds_budget");
    }
  }
  if (workLimit !== null && estimate.workUnits !== null && estimate.workUnits > workLimit) blockers.push("declared_work_exceeds_budget");
  return Object.freeze({ allowed: blockers.length === 0, estimate, policy, bytesRequired, blockers: Object.freeze(blockers), claimBoundary: boundary, requestedWallMs });
}
