// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original Kuramoto artifact verification

import { encodeKuramotoInput } from "../../panel/kuramoto";
import type { KuramotoBounds, KuramotoRequest } from "../../panel/kuramoto";
import { canonicalDigest, readJson, writeJson } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";
import type { RawCodec } from "../../shared/contracts";
import type { WorkspaceArchiveMember } from "../../shared/storage/workspaceArchive";
import {
  admitOwnedKuramotoResources,
  browserResourcePolicy,
} from "../../shared/resources/kuramotoResources";
import type { ResourcePolicy } from "../../shared/resources/admission";

/** Deliberately authored local refusal; unexpected runtime faults retain a fixed UI message. */
export class ExperimentRefusal extends Error {}

/** Explicit browser producer formats over the unchanged kernel ABI and resource owner. */
export const experimentSchemas = Object.freeze({
  /** Original bounded little-endian kernel input bytes. */ input: "studio.kuramoto-input.v1",
  /** Original WASM v1 program bytes. */ kernel: "studio.kuramoto-kernel.v1",
  /** Declared original numeric-payload and work ceilings. */ policy: "studio.kuramoto-policy.v1",
  /** Recorded source backend, method, precision, units and build. */ environment:
    "studio.kuramoto-environment.v1",
  /** Immutable source-bound effective numerical admission. */ plan: "studio.kuramoto-plan.v1",
  /** Source-bound actual order parameter and final phase bits. */ output:
    "studio.kuramoto-output.v1",
});
/** Supported browser artifact roles; these do not replace the five workspace documents. */
export type ExperimentArtifactKind = keyof typeof experimentSchemas;
const producerKinds = {
  input: "problem",
  kernel: "program",
  policy: "policy",
  environment: "environment",
  plan: "plan",
  output: "output",
} as const;
const byteLimit = 2 * 1024 * 1024;

/** Exact raw source digest; no workspace canonical re-encoding changes its identity. */
export async function artifactBytesDigest(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new Uint8Array(bytes));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Convert a finite backend scalar to the workspace's original big-endian IEEE spelling. */
export function encodeFloat64(value: number): string {
  if (!Number.isFinite(value))
    throw new ExperimentRefusal("finite float64 artifact value required");
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Decode an exact finite scalar without changing its signed zero or source unit. */
export function decodeFloat64(value: unknown): number {
  if (typeof value !== "string" || !/^[0-9a-f]{16}$/.test(value))
    throw new ExperimentRefusal("16 lowercase IEEE float64 digits required");
  const bytes = Uint8Array.from({ length: 8 }, (_, index) =>
    parseInt(value.slice(2 * index, 2 * index + 2), 16),
  );
  const number = new DataView(bytes.buffer).getFloat64(0, false);
  if (!Number.isFinite(number))
    throw new ExperimentRefusal("finite float64 artifact value required");
  return number;
}

/** Decode the original 32-byte little-endian input with bounded vectors, without solving. */
export function decodeKernelInput(content: Uint8Array): KuramotoRequest {
  if (content.byteLength < 32 || content.byteLength > byteLimit)
    throw new ExperimentRefusal("bounded complete kernel input required");
  const bytes = new Uint8Array(content);
  const view = new DataView(bytes.buffer);
  const version = view.getUint32(0, true),
    mode = view.getUint32(4, true),
    n = view.getUint32(8, true),
    steps = view.getUint32(12, true);
  if (version !== 1 || mode > 1 || n < 1 || n > 128 || steps < 1 || steps > 4096)
    throw new ExperimentRefusal("unsupported kernel version, method or browser input shape");
  const matrix = mode === 1 ? n * n : 0;
  if (bytes.byteLength !== 32 + 8 * (2 * n + matrix))
    throw new ExperimentRefusal("kernel input byte shape mismatch");
  const vector = (start: number, length: number) =>
    Object.freeze(
      Array.from({ length }, (_, index) => view.getFloat64(32 + 8 * (start + index), true)),
    );
  const request: KuramotoRequest = Object.freeze({
    mode: mode === 0 ? "mean-field" : "networked",
    omega: vector(0, n),
    theta0: vector(n, n),
    steps,
    dt: view.getFloat64(16, true),
    coupling: view.getFloat64(24, true),
    ...(mode === 1 ? { kNm: vector(2 * n, matrix) } : {}),
  });
  if (encodeKuramotoInput(request) === null)
    throw new ExperimentRefusal("original kernel input admission refused");
  return request;
}

function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new ExperimentRefusal("artifact object required");
  const entries = dataEntries(value);
  if (entries.length !== keys.length || entries.some(([key]) => !keys.includes(key)))
    throw new ExperimentRefusal("unsupported or missing artifact fields");
  return Object.fromEntries(entries);
}
function hash(value: unknown): string {
  if (typeof value !== "string" || value.length !== 64 || !/^[0-9a-f]{64}$/.test(value))
    throw new ExperimentRefusal("exact lowercase source SHA256 required");
  return value;
}
function bounds(value: unknown): KuramotoBounds {
  const fields = object(value, ["maxOscillators", "maxSteps"]);
  const result = {
    maxOscillators: fields["maxOscillators"] as number,
    maxSteps: fields["maxSteps"] as number,
  };
  browserResourcePolicy(result);
  return Object.freeze(result);
}
function policy(value: unknown): ResourcePolicy {
  return object(value, [
    "source",
    "addressableBytes",
    "memoryBytes",
    "workUnits",
    "overheadBytes",
  ]) as unknown as ResourcePolicy;
}
function vector(value: unknown, maximum: number): void {
  if (!Array.isArray(value) || value.length < 1 || value.length > maximum)
    throw new ExperimentRefusal("bounded output vector required");
  value.forEach(decodeFloat64);
}

/** Read one producer's exact metadata after its independent offline verifier succeeds. */
export async function readExperimentArtifact(
  kind: Exclude<ExperimentArtifactKind, "input" | "kernel">,
  bytes: Uint8Array,
): Promise<Readonly<Record<string, unknown>>> {
  if (bytes.byteLength < 1 || bytes.byteLength > byteLimit)
    throw new ExperimentRefusal("bounded artifact metadata required");
  const raw = readJson(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  const fields =
    kind === "policy"
      ? ["source", "addressableBytes", "memoryBytes", "workUnits", "overheadBytes"]
      : kind === "environment"
        ? [
            "backend",
            "integrator",
            "dtype",
            "phase_unit",
            "frequency_unit",
            "time_unit",
            "seed",
            "kernel_sha256",
            "bounds",
            "browser_user_agent",
          ]
        : kind === "plan"
          ? [
              "revision_hash",
              "kernel_sha256",
              "input_sha256",
              "environment_sha256",
              "policy_sha256",
              "shape",
              "bounds",
              "binary_bytes",
              "policy",
              "requested_memory_bytes",
              "deadline_ms",
              "admission",
            ]
          : ["revision_hash", "plan_hash", "kernel_sha256", "order_parameter", "theta_final"];
  const payload = object(raw, fields);
  if (kind === "policy") {
    // The original admission owner validates policy spelling and exact ceilings.
    admitOwnedKuramotoResources(
      { n: 1, steps: 1, mode: "mean-field" },
      { maxOscillators: 128, maxSteps: 4096 },
      8,
      policy(payload),
    );
  } else if (kind === "environment") {
    if (
      payload["backend"] !== "shipped-kuramoto-wasm-float64-owned-worker" ||
      payload["integrator"] !== "Rust fixed-step RK4" ||
      payload["dtype"] !== "float64" ||
      payload["phase_unit"] !== "rad" ||
      payload["frequency_unit"] !== "rad/model-time" ||
      payload["time_unit"] !== "model-time" ||
      payload["seed"] !== null
    )
      throw new ExperimentRefusal("unsupported numerical environment");
    hash(payload["kernel_sha256"]);
    bounds(payload["bounds"]);
    if (
      typeof payload["browser_user_agent"] !== "string" ||
      payload["browser_user_agent"].length > 4096
    )
      throw new ExperimentRefusal("bounded observed browser environment required");
  } else if (kind === "plan") {
    for (const key of [
      "revision_hash",
      "kernel_sha256",
      "input_sha256",
      "environment_sha256",
      "policy_sha256",
    ])
      hash(payload[key]);
    const shape = object(payload["shape"], ["n", "steps", "mode"]);
    const actual = admitOwnedKuramotoResources(
      {
        n: shape["n"] as number,
        steps: shape["steps"] as number,
        mode: shape["mode"] as KuramotoRequest["mode"],
      },
      bounds(payload["bounds"]),
      payload["binary_bytes"] as number,
      policy(payload["policy"]),
    );
    const requested = payload["requested_memory_bytes"];
    if (
      requested !== null &&
      (typeof requested !== "bigint" || requested < 0n || requested !== actual.policy.memoryBytes)
    )
      throw new ExperimentRefusal("requested and effective run byte ceiling differ");
    if (
      (await canonicalDigest("studio.kuramoto-admission.v1", actual)) !==
      (await canonicalDigest("studio.kuramoto-admission.v1", payload["admission"]))
    )
      throw new ExperimentRefusal("resource admission differs from the original owner");
    if (
      !Number.isSafeInteger(payload["deadline_ms"]) ||
      (payload["deadline_ms"] as number) < 1 ||
      (payload["deadline_ms"] as number) > 60_000
    )
      throw new ExperimentRefusal("bounded operational deadline required");
  } else {
    for (const key of ["revision_hash", "plan_hash", "kernel_sha256"]) hash(payload[key]);
    vector(payload["order_parameter"], 4097);
    vector(payload["theta_final"], 128);
  }
  return Object.freeze(payload);
}

async function verify(kind: ExperimentArtifactKind, content: Uint8Array): Promise<void> {
  if (kind === "input") {
    decodeKernelInput(content);
    return;
  }
  if (kind === "kernel") {
    if (
      content.length < 8 ||
      content.length > byteLimit ||
      [0, 97, 115, 109, 1, 0, 0, 0].some((byte, index) => content[index] !== byte)
    )
      throw new ExperimentRefusal("bounded original WASM v1 source required");
    // Format/digest admission grants no execution; the run additionally binds the shipped build.
    return;
  }
  await readExperimentArtifact(kind, content);
}

/** Built-in verifiers supplied in product code; an imported archive cannot install one. */
export const localExperimentCodecs: ReadonlyMap<string, RawCodec> = new Map(
  Object.entries(experimentSchemas).map(([name, schema]) => {
    const kind = name as ExperimentArtifactKind;
    const codec: RawCodec = async (content) => {
      await verify(kind, content);
      return { schema, kind: producerKinds[kind], digest: await artifactBytesDigest(content) };
    };
    return [schema, codec];
  }),
);

/** Create a verified original raw member, preserving supplied bytes and their SHA256. */
export async function makeArtifact(
  kind: ExperimentArtifactKind,
  value: Uint8Array | Readonly<Record<string, unknown>>,
): Promise<WorkspaceArchiveMember> {
  const bytes =
    value instanceof Uint8Array
      ? new Uint8Array(value)
      : new TextEncoder().encode(writeJson(value));
  const verifier = localExperimentCodecs.get(experimentSchemas[kind]);
  if (verifier === undefined) throw new ExperimentRefusal("original artifact producer unavailable");
  const identity = await verifier(bytes);
  return Object.freeze({
    name: `raw/${identity.digest}.bin`,
    kind: "raw",
    schema: identity.schema,
    sha256: identity.digest,
    content: Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(""),
  });
}

/** Restore bounded original bytes from an already graph-admitted raw archive member. */
export function artifactContent(member: WorkspaceArchiveMember): Uint8Array<ArrayBuffer> {
  if (
    member.kind !== "raw" ||
    member.content.length > 2 * byteLimit ||
    member.content.length % 2 !== 0 ||
    /[^0-9a-f]/.test(member.content)
  )
    throw new ExperimentRefusal("bounded admitted original raw member required");
  const nativeDecoder = (
    Uint8Array as Uint8ArrayConstructor & {
      fromHex?: (hex: string) => Uint8Array<ArrayBuffer>;
    }
  ).fromHex;
  if (typeof nativeDecoder === "function") return nativeDecoder(member.content);
  return Uint8Array.from({ length: member.content.length / 2 }, (_, index) =>
    parseInt(member.content.slice(2 * index, 2 * index + 2), 16),
  );
}
