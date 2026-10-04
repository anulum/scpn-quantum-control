// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned policy verdict admission

import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson } from "../../../shared/contracts/jsonTransport";
import { parseResolvedSettings } from "../../../shared/contracts/workspace";
import type { ResolvedSettings } from "../../../shared/contracts/workspace";

/** Exact native verdict envelope; this consumer never computes provider policy. */
export const OPERATOR_DECISION_SCHEMA = "studio.operator-policy-decision.v1";
/** Product UTF-8 bound matching the native producer. */
export const MAX_OPERATOR_DECISION_BYTES = 1024 * 1024;

/** Native immutable inputs and dated source verdict, without execution authority. */
export interface NativeOperatorDecision {
  /** Original plan admission, distinct from run approval and execution. */
  readonly allowed: boolean;
  /** Source clock at which policy and estimate were assessed. */
  readonly assessed_at: string;
  /** Ordered original refusal identifiers. */
  readonly reasons: readonly string[];
  /** Source binding differences that were refused, never applied. */
  readonly rejected_substitutions: readonly string[];
  /** Exact request fields; integers retain bigint precision. */
  readonly request: Readonly<Record<string, unknown>>;
  /** Source-owned ceilings and dated validity inputs. */
  readonly policy: Readonly<Record<string, unknown>>;
  /** Dated complete-request price inputs or unknown estimate. */
  readonly estimate: Readonly<Record<string, unknown>> | null;
}

/** Admitted readonly projection retaining the original input bytes. */
export interface OperatorDecisionSnapshot {
  /** Original admitted UTF-8 text for verbatim export. */
  readonly text: string;
  /** Complete native envelope identity. */
  readonly sha256: string;
  /** Original settings document identity. */
  readonly settingsSha256: string;
  /** Original full requested/effective/origin and authority references. */
  readonly settings: ResolvedSettings;
  /** Source verdict; importing does not rerun or approve policy. */
  readonly decision: NativeOperatorDecision;
}

/** Refusal never carries a replacement admitted snapshot. */
export type OperatorDecisionResult =
  { readonly ok: true; readonly value: OperatorDecisionSnapshot } |
  { readonly ok: false; readonly message: string };

function require(condition: boolean): void { if (!condition) throw new Error("Policy metadata refused"); }
function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  require(typeof value === "object" && value !== null && !Array.isArray(value));
  const row = value as Record<string, unknown>;
  require(Object.keys(row).length === keys.length && keys.every(key => Object.hasOwn(row, key)));
  return row;
}
function text(value: unknown): string {
  require(typeof value === "string" && value.trim().length > 0 && [...value].length <= 256 && !/[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(value));
  return value as string;
}
function hash(value: unknown): string { const result = text(value); require(/^[0-9a-f]{64}$/.test(result)); return result; }
function timestamp(value: unknown): string {
  const result = text(value);
  require(/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/.test(result) && !result.startsWith("0000"));
  const date = new Date(result);
  require(Number.isFinite(date.getTime()) && date.toISOString().slice(0, 19) + "Z" === result);
  return result;
}
function nullableText(value: unknown): void { if (value !== null) text(value); }
function positive(value: unknown): void { require(typeof value === "bigint" && value >= 1n && value <= 2n ** 63n - 1n); }
function money(value: unknown): void { require(typeof value === "string" && /^(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,9})?$/.test(value)); }
function currency(value: unknown): void { require(typeof value === "string" && /^[A-Z]{3}$/.test(value)); }
function strings(value: unknown, minimum = 0): readonly string[] {
  require(Array.isArray(value) && value.length >= minimum && value.length <= 256);
  const result = (value as unknown[]).map(text);
  require(new Set(result).size === result.length);
  return Object.freeze(result);
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}

/** Admit exact native structure, provenance and hashes without executing a policy. */
export async function parseOperatorDecision(raw: string): Promise<OperatorDecisionResult> {
  try {
    require(raw.length <= MAX_OPERATOR_DECISION_BYTES && new TextEncoder().encode(raw).length <= MAX_OPERATOR_DECISION_BYTES);
    const wire = object(readJson(raw), ["schema", "body", "extensions", "sha256"]);
    require(wire["schema"] === OPERATOR_DECISION_SCHEMA);
    object(wire["extensions"], []);
    const body = object(wire["body"], ["no_submit", "claim_boundary", "settings", "settings_sha256", "decision"]);
    require(body["no_submit"] === true && body["claim_boundary"] === "dated_plan_admission_only");
    const settings = parseResolvedSettings(body["settings"]);
    if (!settings.ok) throw new Error(settings.message);
    const settingsSha256 = hash(body["settings_sha256"]);
    require(await canonicalDigest("resolved_settings.v1", body["settings"]) === settingsSha256);
    const decision = object(body["decision"], ["request", "policy", "estimate", "assessed_at", "allowed", "reasons", "rejected_substitutions"]);
    timestamp(decision["assessed_at"]); require(typeof decision["allowed"] === "boolean");
    const reasons = strings(decision["reasons"]), substitutions = strings(decision["rejected_substitutions"]);
    require(decision["allowed"] === (reasons.length === 0) && substitutions.every(reason => reasons.includes(reason)));
    const request = object(decision["request"], ["workload_sha256", "backend_id", "target", "region", "shots", "concurrency", "time_limit_ms", "unattended"]);
    hash(request["workload_sha256"]); text(request["backend_id"]);
    nullableText(request["target"]); nullableText(request["region"]); require(typeof request["unattended"] === "boolean");
    for (const key of ["shots", "concurrency", "time_limit_ms"]) positive(request[key]);
    const policy = object(decision["policy"], ["reference", "backend_id", "targets", "regions", "max_shots", "max_concurrency", "max_time_limit_ms", "max_cost", "currency", "valid_from", "expires_at"]);
    text(policy["reference"]); text(policy["backend_id"]); strings(policy["targets"], 1); strings(policy["regions"], 1);
    for (const key of ["max_shots", "max_concurrency", "max_time_limit_ms"]) positive(policy[key]);
    money(policy["max_cost"]); currency(policy["currency"]);
    require(timestamp(policy["valid_from"]) < timestamp(policy["expires_at"]));
    const reference = settings.value.body["policy_ref"] as Readonly<Record<string, unknown>>;
    require(reference["schema"] === "operator_policy.v1" && reference["sha256"] === await canonicalDigest("operator_policy.v1", policy));
    const effective = settings.value.body["effective"] as Readonly<Record<string, unknown>>;
    const requested = settings.value.body["requested"];
    require(await canonicalDigest("operator-settings-fields.v1", requested) === await canonicalDigest("operator-settings-fields.v1", effective));
    for (const [source, target] of [["backend", "backend_id"], ["device", "target"], ["region", "region"], ["shots", "shots"], ["concurrency", "concurrency"], ["time_limit_ms", "time_limit_ms"], ["unattended", "unattended"]] as const) require(effective[source] === request[target]);
    if (decision["estimate"] !== null) {
      const price = object(decision["estimate"], ["request_sha256", "amount", "currency", "source_ref", "observed_at", "expires_at"]);
      hash(price["request_sha256"]); if (price["amount"] !== null) money(price["amount"]);
      currency(price["currency"]); text(price["source_ref"]);
      require(timestamp(price["observed_at"]) < timestamp(price["expires_at"]));
      // A mismatch is an observable native refusal input, not a reason to
      // recompute admission or silently replace the estimate in this view.
    }
    const sha256 = hash(wire["sha256"]);
    require(await canonicalDigest(OPERATOR_DECISION_SCHEMA, { schema: wire["schema"], body, extensions: wire["extensions"] }) === sha256);
    return { ok: true, value: freeze({ text: raw, sha256, settingsSha256, settings: settings.value, decision: decision as unknown as NativeOperatorDecision }) };
  } catch {
    return { ok: false, message: "Operator policy metadata refused; the prior admitted decision remains unchanged." };
  }
}
