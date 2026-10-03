// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — owned worker over the original Kuramoto WASM ABI

import { bindKuramoto, encodeKuramotoInput, readBounds } from "../panel/kuramoto";
import type { KernelSimulate, KuramotoExports, KuramotoRequest } from "../panel/kuramoto";
import { admitOwnedKuramotoResources } from "../shared/resources/kuramotoResources";
import type { ResourcePolicy } from "../shared/resources/admission";
import { kernelBinaryFingerprint, kernelBinarySize, ownedWorkerBinary, ownedWorkerVector, readWorkerRequest, workerData, workerDigest, MAX_KERNEL_BINARY_BYTES } from "./kernelProtocol";
import type { KernelWorkerEvent, KernelWorkerPayload, KernelWorkerRequest, KernelWireInput } from "./kernelProtocol";

/** Native browser worker operations needed by the entry point. */
export interface KernelWorkerScope {
  /** Receive a structured-cloned transport envelope. */
  addEventListener(kind: "message", listener: (event: { readonly data: unknown }) => void): void;
  /** Send original outputs with their owned transfer list. */
  postMessage(value: KernelWorkerEvent, transfers?: ArrayBuffer[]): void;
}

function numeric(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("finite worker scalar required");
  return value;
}

function vector(value: unknown): Float64Array<ArrayBuffer> {
  if (!ownedWorkerVector(value)) throw new Error("owned float64 worker vector required");
  return value;
}

function payload(value: unknown): KernelWorkerPayload {
  const raw = workerData(value, ["wasm", "build_fingerprint", "input", "bounds", "policy"]);
  const wasm = raw["wasm"];
  if (!ownedWorkerBinary(wasm)) throw new Error("bounded WASM binary required");
  const binaryBytes = kernelBinarySize(wasm);
  if (binaryBytes < 1 || binaryBytes > MAX_KERNEL_BINARY_BYTES) throw new Error("bounded WASM binary required");
  const fields = workerData(raw["input"], ["mode", "omega", "theta0", "kNm", "steps", "dt", "coupling"]);
  if (fields["mode"] !== "mean-field" && fields["mode"] !== "networked") throw new Error("original Kuramoto mode required");
  const input: KernelWireInput = { mode: fields["mode"], omega: vector(fields["omega"]), theta0: vector(fields["theta0"]), kNm: vector(fields["kNm"]), steps: numeric(fields["steps"]), dt: numeric(fields["dt"]), coupling: numeric(fields["coupling"]) };
  const limits = workerData(raw["bounds"], ["maxOscillators", "maxSteps"]);
  const bounds = { maxOscillators: numeric(limits["maxOscillators"]), maxSteps: numeric(limits["maxSteps"]) };
  const policy = raw["policy"] as ResourcePolicy;
  const admission = admitOwnedKuramotoResources({ n: input.omega.length, steps: input.steps, mode: input.mode }, bounds, binaryBytes, policy);
  if (!admission.allowed) throw new Error(`worker resource policy refused: ${admission.blockers.join(", ")}`);
  return { wasm: wasm as Uint8Array<ArrayBuffer>, build_fingerprint: workerDigest(raw["build_fingerprint"]), input, bounds, policy: admission.policy };
}

function sourceRequest(input: KernelWireInput): KuramotoRequest {
  const source = { mode: input.mode, omega: Array.from(input.omega), theta0: Array.from(input.theta0), steps: input.steps, dt: input.dt, coupling: input.coupling, kNm: Array.from(input.kNm) };
  if (encodeKuramotoInput(source) === null) throw new Error("original source request refused");
  return source;
}

/** Install one source-bound lifecycle; cancellation acknowledgement belongs to the disposing host. */
export function installKernelWorker(scope: KernelWorkerScope): void {
  let owned: KernelWorkerRequest | null = null;
  let simulate: KernelSimulate | null = null;
  let source: KuramotoRequest | null = null;
  let fingerprint: string | null = null;
  let sequence = 0;
  let generation = 0;
  const emit = (request: KernelWorkerRequest, kind: KernelWorkerEvent["kind"], detail: Readonly<Record<string, unknown>>, transfers: ArrayBuffer[] = []) => {
    const own = owned?.run_id === request.run_id;
    scope.postMessage({ version: 1, run_id: request.run_id, sequence: own ? ++sequence : 1, kind, payload: { revision_hash: request.revision_hash, plan_hash: request.plan_hash, build_fingerprint: fingerprint, ...detail } }, transfers);
  };
  scope.addEventListener("message", event => {
    // A malformed outer request raises the native event error synchronously;
    // it cannot invent a recipient or leave an unhandled timer behind.
    const request = readWorkerRequest(event.data);
    const execute = async () => {
      try {
        if (request.command === "validate") {
          if (owned !== null) throw new Error("this worker already owns a run");
          owned = request;
          const token = ++generation;
          const declaration = payload(request.payload);
          fingerprint = declaration.build_fingerprint;
          source = sourceRequest(declaration.input);
          if (await kernelBinaryFingerprint(declaration.wasm) !== fingerprint) throw new Error("WASM build fingerprint differs from the frozen run");
          const { instance } = await WebAssembly.instantiate(declaration.wasm, {});
          if (generation !== token) return;
          const exports = instance.exports as unknown as KuramotoExports;
          const actual = readBounds(exports);
          if (actual.maxOscillators !== declaration.bounds.maxOscillators || actual.maxSteps !== declaration.bounds.maxSteps) throw new Error("source kernel limits differ from the frozen run");
          simulate = bindKuramoto(exports, declaration.policy);
          emit(request, "accepted", { bounds: actual });
          return;
        }
        if (owned === null || request.run_id !== owned.run_id || request.revision_hash !== owned.revision_hash || request.plan_hash !== owned.plan_hash) throw new Error("worker command does not own this run revision");
        if (request.payload !== null) throw new Error("validated run/cancel commands carry no replacement payload");
        if (request.command === "cancel") {
          ++generation;
          simulate = null;
          source = null;
          emit(request, "progress", { stage: "cancellation requested; disposing owner must terminate" });
          return;
        }
        if (simulate === null || source === null) throw new Error("run requires accepted source validation");
        emit(request, "progress", { stage: "original WASM computation" });
        const result = simulate(source);
        simulate = null;
        source = null;
        if (!result.ok) throw new Error(result.reason);
        const { orderParameter, thetaFinal } = result.run;
        // The original binder copies finite decoded outputs into new arrays;
        // these buffers never alias the WASM guest or retained caller state.
        emit(request, "result", { orderParameter, thetaFinal }, [orderParameter.buffer as ArrayBuffer, thetaFinal.buffer as ArrayBuffer]);
      } catch (error: unknown) {
        emit(request, "failed", { reason: error instanceof Error ? error.message : "worker execution failed" });
      }
    };
    void execute();
  });
}

installKernelWorker(globalThis as unknown as KernelWorkerScope);
