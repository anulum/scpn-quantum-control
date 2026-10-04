// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound local experiment numerical plan

import { encodeKuramotoInput } from "../../panel/kuramoto";
import type { KuramotoBounds, KuramotoKernel, KuramotoRequest } from "../../panel/kuramoto";
import { readJson } from "../../shared/contracts";
import type { LocalRunRecord, RawCodec } from "../../shared/contracts";
import type { WorkspaceArchiveMember } from "../../shared/storage/workspaceArchive";
import { admitOwnedKuramotoResources, browserResourcePolicy } from "../../shared/resources/kuramotoResources";
import type { ResourceAdmission, ResourcePolicy } from "../../shared/resources/admission";
import type { OwnedKernelOutcome } from "../../workers/kernelProtocol";
import { readLocalExperiment, validateExperimentLifecycle } from "./experimentArchive";
import type { ArchivedExperimentEvent } from "./experimentArchive";
import type { LocalExperimentSource } from "./experimentArchive";
import { artifactBytesDigest, artifactContent, encodeFloat64, ExperimentRefusal, experimentSchemas, makeArtifact, readExperimentArtifact } from "./kuramotoArtifacts";

/** Explicit run settings below the original source/product ceilings. */
export interface ExperimentPlanOptions {
  /** Canonical decimal byte ceiling; omitted retains the recorded source policy. */ readonly memoryBudget?: string;
  /** Operational worker disposal deadline; never a guaranteed compute duration. */ readonly deadlineMs?: number;
}

/** Frozen local plan captured before worker allocation; each identity is source-owned. */
export interface ExperimentPlan {
  /** Exact prior archive to use at the original conditional save boundary. */ readonly sourceJson: string;
  /** Exact selected v1 revision document digest. */ readonly revisionHash: string;
  /** Original raw plan identity binding settings, input, environment and build. */ readonly planHash: string;
  /** Actual currently loaded original WASM SHA256. */ readonly buildFingerprint: string;
  /** Captured source bytes; only the original worker client's owned copy transfers. */ readonly wasmBytes: Uint8Array<ArrayBuffer>;
  /** Actual native kernel limits verified against the recorded environment. */ readonly bounds: KuramotoBounds;
  /** Exact effective source request from the admitted typed revision. */ readonly request: KuramotoRequest;
  /** Complete original declared numeric/worker/binary buffer admission. */ readonly admission: ResourceAdmission;
  /** Original source policy with an optional explicit tighter run ceiling. */ readonly policy: ResourcePolicy;
  /** Bounded operational disposal timeout. */ readonly deadlineMs: number;
  /** Explicit requested run ceiling, or null when the recorded source policy applies. */ readonly requestedMemoryBytes: bigint | null;
  /** Exact effective native input bytes packaged for independent replay. */ readonly inputMember: WorkspaceArchiveMember;
  /** Effective run ceiling and its declared origin. */ readonly policyMember: WorkspaceArchiveMember;
  /** Original raw plan to index in the immutable workspace graph. */ readonly planMember: WorkspaceArchiveMember;
}

/** Original saved result's exact finite float64 spelling, without a fresh-run success claim. */
export interface ReplayExpectation {
  /** Ordered original order-parameter bits. */ readonly orderParameter: readonly string[];
  /** Ordered original final phase bits. */ readonly thetaFinal: readonly string[];
}

/** Reconstructed original admission and saved bits, without starting a worker. */
export interface SavedExperimentReplay {
  /** Original effective numerical plan bound to its admitted source and current kernel. */ readonly plan: ExperimentPlan;
  /** Original saved output spelling required for a subsequent independent comparison. */ readonly expected: ReplayExpectation;
}

/** Reuse original resource admission and bind one source revision to the actually shipped kernel. */
export async function prepareExperimentPlan(source: LocalExperimentSource, kernel: KuramotoKernel, options: ExperimentPlanOptions = {}): Promise<ExperimentPlan> {
  if (!kernel.sourceBytes || kernel.sourceBytes.length < 1 || await artifactBytesDigest(kernel.sourceBytes) !== source.kernelHash) throw new ExperimentRefusal("currently shipped kernel differs from the original source; replay unavailable");
  const recorded = source.environment["bounds"] as KuramotoBounds;
  if (recorded.maxOscillators !== kernel.bounds.maxOscillators || recorded.maxSteps !== kernel.bounds.maxSteps) throw new ExperimentRefusal("native kernel limits differ from the original environment");
  const ceiling = browserResourcePolicy(kernel.bounds), original = source.policy;
  if (original.addressableBytes > ceiling.addressableBytes || (original.memoryBytes !== null && original.memoryBytes > ceiling.memoryBytes!) || (original.workUnits !== null && original.workUnits > ceiling.workUnits!)) throw new ExperimentRefusal("source policy ceiling exceeds the current original product policy");
  let requestedMemory: bigint | null = null;
  let policy = original;
  if (options.memoryBudget !== undefined) {
    if (!/^(?:0|[1-9][0-9]{0,19})$/.test(options.memoryBudget)) throw new ExperimentRefusal("canonical nonnegative bounded byte count required");
    requestedMemory = BigInt(options.memoryBudget);
    if (original.memoryBytes === null || requestedMemory > original.memoryBytes) throw new ExperimentRefusal("requested byte budget exceeds the original policy ceiling");
    policy = { ...original, memoryBytes: requestedMemory, source: "explicit run numeric-payload ceiling below the original source policy" };
  }
  const admission = admitOwnedKuramotoResources({ n: source.request.omega.length, steps: source.request.steps, mode: source.request.mode }, kernel.bounds, kernel.sourceBytes.length, policy);
  const deadlineMs = options.deadlineMs ?? 5000;
  if (!Number.isSafeInteger(deadlineMs) || deadlineMs < 1 || deadlineMs > 60_000) throw new ExperimentRefusal("bounded operational deadline required");
  const input = encodeKuramotoInput(source.request);
  if (input === null) throw new ExperimentRefusal("effective original input refused before worker allocation");
  const inputMember = await makeArtifact("input", input), policyMember = await makeArtifact("policy", { ...admission.policy });
  const planMember = await makeArtifact("plan", { revision_hash: source.revisionHash, kernel_sha256: source.kernelHash,
    input_sha256: inputMember.sha256, environment_sha256: source.environmentMember.sha256, policy_sha256: policyMember.sha256,
    shape: { n: source.request.omega.length, steps: source.request.steps, mode: source.request.mode }, bounds: kernel.bounds,
    binary_bytes: kernel.sourceBytes.length, policy: admission.policy, requested_memory_bytes: requestedMemory, deadline_ms: deadlineMs, admission });
  return Object.freeze({ sourceJson: source.archive.preview.json, revisionHash: source.revisionHash, planHash: planMember.sha256,
    buildFingerprint: source.kernelHash, wasmBytes: new Uint8Array(kernel.sourceBytes), bounds: Object.freeze({ ...kernel.bounds }), request: source.request,
    admission, policy: admission.policy, deadlineMs, requestedMemoryBytes: requestedMemory, inputMember, policyMember, planMember });
}

/** Re-admit an explicitly listed completed record and reprepare its exact original numerical plan. */
export async function prepareSavedReplay(json: string, kernel: KuramotoKernel, rawCodecs: ReadonlyMap<string, RawCodec>): Promise<SavedExperimentReplay> {
  const source = await readLocalExperiment(json, rawCodecs);
  const references = source.archive.manifest.body["artefact_refs"] as readonly { readonly sha256: string }[];
  const members = new Map(source.archive.members.map(member => [member.sha256, member]));
  for (const reference of [...references].reverse()) {
    const member = members.get(reference.sha256)!;
    if (member.schema !== "local_run_record.v1") continue;
    // The original complete graph has already parsed this exact indexed run document.
    const record = (readJson(member.content) as LocalRunRecord).body;
    const events = record["events"] as readonly ArchivedExperimentEvent[];
    if (record["revision_hash"] !== source.revisionHash || events.at(-1)?.kind !== "result" || events.at(-1)?.payload["disposed"] !== true) continue;
    const storedPlan = members.get(record["plan_hash"] as string)!;
    if (storedPlan.schema !== experimentSchemas.plan) throw new ExperimentRefusal("original replay plan source format is unsupported");
    const payload = await readExperimentArtifact("plan", artifactContent(storedPlan));
    const requested = payload["requested_memory_bytes"] as bigint | null;
    const plan = await prepareExperimentPlan(source, kernel, { deadlineMs: payload["deadline_ms"] as number,
      ...(requested === null ? {} : { memoryBudget: requested.toString() }) });
    if (plan.planHash !== record["plan_hash"]) throw new ExperimentRefusal("original replay plan identity differs from the current source");
    validateExperimentLifecycle(events, record["run_id"] as string, plan.revisionHash, plan.planHash, plan.buildFingerprint, "result");
    const outputs = record["output_refs"] as readonly { readonly schema: string; readonly sha256: string }[];
    if (outputs.length !== 1 || outputs[0]!.schema !== experimentSchemas.output) throw new ExperimentRefusal("one original source output artifact required for replay");
    const output = await readExperimentArtifact("output", artifactContent(members.get(outputs[0]!.sha256)!));
    if (output["revision_hash"] !== plan.revisionHash || output["plan_hash"] !== plan.planHash || output["kernel_sha256"] !== plan.buildFingerprint) throw new ExperimentRefusal("original replay output belongs to another source or plan");
    const orderParameter = output["order_parameter"] as readonly string[], thetaFinal = output["theta_final"] as readonly string[];
    if (orderParameter.length !== plan.request.steps + 1 || thetaFinal.length !== plan.request.omega.length) throw new ExperimentRefusal("original replay output shape differs from the plan");
    return Object.freeze({ plan, expected: Object.freeze({ orderParameter: Object.freeze([...orderParameter]), thetaFinal: Object.freeze([...thetaFinal]) }) });
  }
  throw new ExperimentRefusal("no disposed successful saved run for the selected immutable revision");
}

/** Compare a genuine new result to the saved original binary64 output; differences remain failures. */
export function verifyReplayOutcome(outcome: OwnedKernelOutcome, expected: ReplayExpectation): void {
  if (!outcome.ok || !outcome.disposed || outcome.run.orderParameter.length !== expected.orderParameter.length || outcome.run.thetaFinal.length !== expected.thetaFinal.length
    || outcome.run.orderParameter.some((value, index) => encodeFloat64(value) !== expected.orderParameter[index])
    || outcome.run.thetaFinal.some((value, index) => encodeFloat64(value) !== expected.thetaFinal[index])) throw new ExperimentRefusal("replay differs from the original saved float64 result");
}
