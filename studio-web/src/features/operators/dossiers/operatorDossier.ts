// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native review dossier admission

import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import { OPERATOR_DECISION_SCHEMA, parseOperatorDecision } from "../policy/policyDecision";
import type { OperatorDecisionSnapshot } from "../policy/policyDecision";

/** Native source envelope; human review confers no submission authority. */
export const DOSSIER_SCHEMA = "studio.operator-review-dossier.v1";
/** Transport retains native dossier text and native generated code separately. */
export const REVIEW_EXPORT_SCHEMA = "studio.operator-review-export.v1";
/** Expanded UTF-8 export limit; inner dossier remains bounded to one MiB. */
export const MAX_REVIEW_EXPORT_BYTES = 8 * 1024 * 1024;
const maxDossierBytes = 1024 * 1024;
const bodyFields = ["no_submit", "claim_boundary", "producer_identity", "plan", "plan_sha256", "profile", "profile_sha256", "workload_sha256", "payload", "settings", "settings_sha256", "semantic_settings", "semantic_settings_sha256", "policy_decision", "policy_decision_sha256", "calibration", "created_at", "expires_at", "execution_sha256"];
const executionFields = ["plan_sha256", "profile_sha256", "workload_sha256", "payload", "semantic_settings_sha256", "policy_decision_sha256", "calibration", "created_at", "expires_at"];
const semanticFields = new Set(["method", "backend", "device", "precision", "seed", "shots", "parameters", "units", "memory_budget_bytes", "n_qubits", "concurrency", "time_limit_ms", "region", "unattended"]);

/** Readonly native source and standalone verifier, never executed by the browser. */
export interface OperatorDossierSnapshot {
  /** Exact outer input for verbatim export. */ readonly text: string;
  /** Original inner text, retained without reserialization. */ readonly dossierText: string;
  /** Original native inner envelope hash. */ readonly sha256: string;
  /** Complete execution identity, independent of display-only settings. */ readonly executionSha256: string;
  /** Exact source metadata; source hashes do not authenticate a provider. */ readonly body: Readonly<Record<string, unknown>>;
  /** Existing native policy metadata projection, without recomputing policy. */ readonly policy: OperatorDecisionSnapshot;
  /** Native Python verifier source; browser downloads original code only. */ readonly script: string;
}
/** Failed imports preserve the prior admitted source. */
export type OperatorDossierResult = {
  /** Whether exact native dossier metadata was admitted. */ readonly ok: true;
  /** Original immutable source and native script. */ readonly value: OperatorDossierSnapshot;
} | {
  /** Whether original metadata admission refused. */ readonly ok: false;
  /** Observable refusal preserving the prior source. */ readonly message: string;
};
/** A separate local human choice referencing the original immutable source. */
export interface HumanReview {
  /** Original dossier identity, retained across display-only changes. */ readonly dossierSha256: string;
  /** Execution identity at review time. */ readonly executionSha256: string;
  /** Human decision, never run approval. */ readonly choice: "approved" | "denied";
  /** UTC seconds when the human made the choice. */ readonly recordedAt: string;
}
/** Observable review states; approval is never provider authority. */
export type ReviewStatus = "pending" | "approved" | "denied" | "expired" | "invalidated" | "refused" | "not_yet_valid";

function require(condition: boolean): void { if (!condition) throw new Error("Review metadata refused"); }
function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  require(typeof value === "object" && value !== null && !Array.isArray(value));
  const row = value as Record<string, unknown>;
  require(Object.keys(row).length === keys.length && keys.every(key => Object.hasOwn(row, key)));
  return row;
}
function text(value: unknown): string {
  require(typeof value === "string" && value.length > 0 && value.trim() === value && [...value].length <= 256 && !/[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(value));
  return value as string;
}
function hash(value: unknown): string { const result = text(value); require(/^[0-9a-f]{64}$/.test(result)); return result; }
function reference(value: unknown): string {
  const result = text(value); require(!result.startsWith("/") && !result.includes("\\") && !result.includes("://") && !result.includes("?") && !result.split("/").includes("..")); return result;
}
function timestamp(value: unknown): string {
  const result = text(value); require(/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/.test(result) && !result.startsWith("0000"));
  const date = new Date(result); require(Number.isFinite(date.getTime()) && date.toISOString().slice(0, 19) + "Z" === result); return result;
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) { Object.values(value).forEach(freeze); Object.freeze(value); } return value;
}
async function sealed(value: unknown, schema: string): Promise<Record<string, unknown>> {
  const wire = object(value, ["schema", "body", "extensions", "sha256"]); require(wire["schema"] === schema); object(wire["extensions"], []);
  require(hash(wire["sha256"]) === await canonicalDigest(schema, { schema, body: wire["body"], extensions: wire["extensions"] })); return wire;
}
async function equal(left: unknown, right: unknown): Promise<void> { require(await canonicalDigest("review-binding.v1", left) === await canonicalDigest("review-binding.v1", right)); }

/** Validate native identities and structural couplings without compiling or assessing policy. */
export async function parseOperatorDossier(raw: string): Promise<OperatorDossierResult> {
  try {
    require(raw.length <= MAX_REVIEW_EXPORT_BYTES && new TextEncoder().encode(raw).length <= MAX_REVIEW_EXPORT_BYTES);
    const exportWire = await sealed(readJson(raw), REVIEW_EXPORT_SCHEMA);
    const bundle = object(exportWire["body"], ["no_submit", "dossier_text", "dossier_sha256", "script"]); require(bundle["no_submit"] === true);
    const dossierText = bundle["dossier_text"]; require(typeof dossierText === "string" && new TextEncoder().encode(dossierText).length <= maxDossierBytes);
    const wire = await sealed(readJson(dossierText as string), DOSSIER_SCHEMA), sha256 = hash(wire["sha256"]);
    require(bundle["dossier_sha256"] === sha256);
    const body = object(wire["body"], bodyFields);
    require(body["no_submit"] === true && body["claim_boundary"] === "human_review_only" && body["producer_identity"] === "scpn_quantum_control.studio.executive_execute.ExecuteActionHandler");
    for (const [field, identity, domain] of [["plan", "plan_sha256", "execution_plan.v1"], ["profile", "profile_sha256", "backend_profile.v1"], ["settings", "settings_sha256", "resolved_settings.v1"], ["semantic_settings", "semantic_settings_sha256", "operator_review_settings.v1"], ["policy_decision", "policy_decision_sha256", "operator_policy_decision.v1"]]) require(hash(body[identity!]) === await canonicalDigest(domain!, body[field!]));
    const executionSha256 = hash(body["execution_sha256"]);
    require(executionSha256 === await canonicalDigest("studio.operator-review-execution.v1", Object.fromEntries(executionFields.map(key => [key, body[key]]))));
    const policyBody = { no_submit: true, claim_boundary: "dated_plan_admission_only", settings: body["settings"], settings_sha256: body["settings_sha256"], decision: body["policy_decision"] };
    const policyWire = { schema: OPERATOR_DECISION_SCHEMA, body: policyBody, extensions: {} };
    const policy = await parseOperatorDecision(writeJson({ ...policyWire, sha256: await canonicalDigest(OPERATOR_DECISION_SCHEMA, policyWire) }));
    if (!policy.ok) throw new Error(policy.message);
    const request = policy.value.decision.request, settings = policy.value.settings.body;
    const plan = object(body["plan"], ["verb", "action_id", "backend", "contract", "claim_boundary", "steps", "parameters"]);
    require(plan["verb"] === "execute"); text(plan["action_id"]); require(typeof plan["claim_boundary"] === "string" && plan["claim_boundary"].length > 0);
    const contract = object(plan["contract"], ["verb", "side_effect", "safety_tier", "requires_approval", "backends", "produces"]);
    await equal(contract, { verb: "execute", side_effect: "LIVE_HARDWARE", safety_tier: "CERTIFIED", requires_approval: true, backends: ["qiskit-runtime", "provider-hal"], produces: ["studio.hardware-result-pack.v1", "studio.qpu-result-pack.v1"] });
    require((contract["backends"] as unknown[]).includes(plan["backend"]));
    require(Array.isArray(plan["steps"]) && plan["steps"].length === 4 && plan["steps"].every(step => typeof step === "string" && step.length > 0));
    const parameters = object(plan["parameters"], ["provider", "endpoint", "circuit_digest", "circuit_ref", "shots", ...(body["calibration"] === null ? [] : ["calibration_ref"])]);
    const profile = object(body["profile"], ["producer_identity", "backend_id", "provider", "broker", "modality", "sdk_package", "ir_formats", "capabilities", "is_cloud", "submit_requires_approval", "region", "target_family", "notes"]);
    require(profile["producer_identity"] === "scpn_quantum_control.hardware.hal.BackendProfile");
    for (const key of ["backend_id", "provider", "broker", "modality", "sdk_package"]) text(profile[key]);
    require(Array.isArray(profile["ir_formats"]) && profile["ir_formats"].length > 0 && profile["ir_formats"].every(item => typeof item === "string" && item.length > 0));
    require(Array.isArray(profile["notes"]) && profile["notes"].every(item => typeof item === "string"));
    for (const key of ["is_cloud", "submit_requires_approval"]) require(typeof profile[key] === "boolean");
    for (const key of ["region", "target_family"]) if (profile[key] !== null) text(profile[key]);
    const capabilities = object(profile["capabilities"], ["supports_shots", "supports_counts", "supports_statevector", "supports_mid_circuit_measurement", "supports_analog", "supports_pulse", "max_qubits", "supports_cancellation", "supports_cost_estimate", ...(Object.hasOwn(profile["capabilities"] as object, "max_shots") ? ["max_shots"] : [])]);
    for (const [key, value] of Object.entries(capabilities)) require(key === "max_qubits" ? value === null || typeof value === "bigint" && value > 0n : key === "max_shots" ? typeof value === "bigint" && value > 0n : typeof value === "boolean");
    require(profile["backend_id"] === request["backend_id"] && profile["provider"] === parameters["provider"] && profile["region"] === request["region"]);
    require(parameters["endpoint"] === request["target"] && parameters["shots"] === request["shots"] && hash(body["workload_sha256"]) === request["workload_sha256"]);
    const payload = object(body["payload"], ["reference", "sha256", "size_bytes"]);
    require(reference(payload["reference"]) === parameters["circuit_ref"] && "sha256:" + hash(payload["sha256"]) === parameters["circuit_digest"]);
    require(typeof payload["size_bytes"] === "bigint" && payload["size_bytes"] > 0n && payload["size_bytes"] <= BigInt(maxDossierBytes));
    const semantic = object(body["semantic_settings"], ["policy_ref", "environment_ref", "settings_plan_sha256", "requested", "effective", "origins"]);
    for (const key of ["policy_ref", "environment_ref"]) await equal(semantic[key], settings[key]);
    for (const key of ["requested", "effective", "origins"]) await equal(semantic[key], Object.fromEntries(Object.entries(settings[key] as Record<string, unknown>).filter(([field]) => semanticFields.has(field))));
    const expectedSettings = { effective: semantic["effective"], policy_ref: settings["policy_ref"], environment_ref: settings["environment_ref"] };
    require(hash(semantic["settings_plan_sha256"]) === await canonicalDigest("quantum_workspace_semantic_settings.v1", expectedSettings));
    const createdAt = timestamp(body["created_at"]), expiresAt = timestamp(body["expires_at"]);
    require(createdAt >= policy.value.decision.assessed_at && createdAt < expiresAt && expiresAt <= String(policy.value.decision.policy["expires_at"]));
    if (policy.value.decision.estimate !== null) require(expiresAt <= String(policy.value.decision.estimate["expires_at"]));
    if (body["calibration"] === null) require(!Object.hasOwn(parameters, "calibration_ref"));
    else {
      const calibration = object(body["calibration"], ["reference", "sha256", "target", "observed_at", "expires_at"]);
      require(reference(calibration["reference"]) === parameters["calibration_ref"] && text(calibration["target"]) === request["target"]); hash(calibration["sha256"]);
      require(timestamp(calibration["observed_at"]) <= createdAt && createdAt < timestamp(calibration["expires_at"]) && expiresAt <= String(calibration["expires_at"]));
    }
    const script = object(bundle["script"], ["language", "filename", "entrypoint", "source", "digest"]);
    require(script["language"] === "python" && script["filename"] === "verify_operator_review.py" && script["entrypoint"] === "python verify_operator_review.py" && typeof script["source"] === "string" && script["source"].length > 0);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(script["source"])));
    require(script["digest"] === "sha256:" + Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join(""));
    return { ok: true, value: freeze({ text: raw, dossierText: dossierText as string, sha256, executionSha256, body, policy: policy.value, script: script["source"] as string }) };
  } catch { return { ok: false, message: "Operator review metadata refused; the prior admitted dossier remains unchanged." }; }
}

/** Resolve current review eligibility against exact source identity and UTC time. */
export function reviewStatus(snapshot: OperatorDossierSnapshot, review: HumanReview | null, now = new Date()): ReviewStatus {
  if (review !== null && review.executionSha256 !== snapshot.executionSha256) return "invalidated";
  if (now.getTime() < Date.parse(String(snapshot.body["created_at"])) || review !== null && Date.parse(review.recordedAt) > now.getTime()) return "not_yet_valid";
  if (review?.choice === "denied") return "denied";
  if (now.getTime() >= Date.parse(String(snapshot.body["expires_at"]))) return "expired";
  if (!snapshot.policy.decision.allowed) return "refused";
  return review === null ? "pending" : "approved";
}

/** Seal a separate human record; the original source is never mutated. */
export async function reviewDocument(review: HumanReview): Promise<string> {
  const wire = { schema: "studio.operator-review.v1", body: { no_submit: true, claim_boundary: "human_review_only", dossier_sha256: review.dossierSha256, execution_sha256: review.executionSha256, choice: review.choice, reviewer_ref: "studio/local-human", recorded_at: review.recordedAt }, extensions: {} };
  return writeJson({ ...wire, sha256: await canonicalDigest(wire.schema, wire) }) + "\n";
}
