// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — dated offline backend metadata admission

import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson } from "../../../shared/contracts/jsonTransport";

/** Maximum imported UTF-8 bytes, matching the original native producer. */
export const MAX_PROFILE_BYTES = 1024 * 1024;
/** Exact source-owned metadata domain. */
export const BACKEND_PROFILES_SCHEMA = "studio.backend-profiles.v1";
/** Operation order inherited from the existing route catalogue. */
export const routeVerbs = ["metadata", "compile", "submit", "retrieve", "cancel", "result_formats"] as const;
/** Names of HAL boolean declarations, independent of runtime observations. */
export const declaredCapabilities = ["supports_shots", "supports_counts", "supports_statevector", "supports_mid_circuit_measurement", "supports_analog", "supports_pulse", "supports_cancellation", "supports_cost_estimate"] as const;
/** Separate nullable declaration and observation from the existing route inventory. */
export interface ProfileVerb {
  /** Source-owned operation name. */ readonly verb: typeof routeVerbs[number];
  /** A declaration is not execution evidence. */ readonly declared: boolean | null;
  /** Declaring source, or absent provenance. */ readonly declared_source: string | null;
  /** Original declaration date. */ readonly declared_on: string | null;
  /** Supplied runtime observation, or unknown. */ readonly observed: boolean | null;
  /** Original observation date. */ readonly observed_on: string | null;
  /** Original native conformance owner, not reverified in the browser. */ readonly conformance_owner: string | null;
}
/** Digest references have no execution or approval authority. */
export interface ProfileBinding {
  /** Exact dated profile identity. */ readonly profile_sha256: string;
  /** Immutable plan metadata, or absent. */ readonly plan_ref: string | null;
  /** Calibration metadata, or absent. */ readonly calibration_ref: string | null;
  /** Review metadata, never permission to run. */ readonly approval_ref: string | null;
}
/** A bounded source-owned declared option and its explicit explanation. */
export interface ProfileOption {
  /** HAL declaration, separate from observed runtime readiness. */ readonly supported: boolean;
  /** Native producer's explanation, never browser provider policy. */ readonly reason: string;
}
/** Exact provider, broker, physical device and dated metadata projection. */
export interface BackendProfile {
  /** Digest of this complete original row body. */ readonly sha256: string;
  /** Unmerged route identifier, including aliases. */ readonly route_id: string;
  /** Physical provider declared by the route. */ readonly provider: string;
  /** Broker identity, or null for direct routing. */ readonly broker: string | null;
  /** Observed physical device or the declared backend route. */ readonly device: string;
  /** Owning HAL backend identifier. */ readonly backend_id: string;
  /** Source-declared physical modality. */ readonly modality: string;
  /** Capture date retained at day precision. */ readonly observed_at: string;
  /** Region declaration, or unknown. */ readonly region: string | null;
  /** Required SDK package, not an availability claim. */ readonly sdk_package: string;
  /** Original supported interchange declarations. */ readonly ir_formats: readonly string[];
  /** Opaque configuration locators, or unknown for a custom adapter. */ readonly credential_refs: readonly string[] | null;
  /** Source HAL capability declarations. */ readonly declared: Readonly<Record<typeof declaredCapabilities[number], boolean>> & {
    /** Exact declared capacity, or unknown. */ readonly max_qubits: bigint | null;
  };
  /** Whitelisted producer observations without arbitrary SDK metadata. */ readonly observed: {
    /** Supplied online state, or unknown. */ readonly online: boolean | null;
    /** Supplied supported IR formats, or unknown. */ readonly ir_formats: readonly string[] | null;
    /** Supplied gate declarations, or unknown. */ readonly basis_gates: readonly string[] | null;
    /** Supplied native feature declarations, or unknown. */ readonly native_features: readonly string[] | null;
    /** Observed qubit count. */ readonly n_qubits: bigint | null;
    /** Observed shot ceiling. */ readonly max_shots: bigint | null;
    /** Observed circuit ceiling. */ readonly max_circuits: bigint | null;
    /** Observed queue depth; zero is preserved. */ readonly queue_depth: bigint | null;
    /** Original UTC calibration timestamp. */ readonly calibration_timestamp: string | null;
    /** Source-owned immutable calibration reference. */ readonly calibration_ref: string | null;
  };
  /** Native declarations and disabled-option explanations. */ readonly options: {
    /** Native pulse declaration and its reason. */ readonly pulse: ProfileOption;
    /** Native analog declaration and its reason. */ readonly analog: ProfileOption;
  };
  /** Six original ordered operation records. */ readonly verbs: readonly ProfileVerb[];
}
/** Admitted immutable-source snapshot; imports confer no execution authority. */
export interface BackendProfileSnapshot {
  /** Exact input text retained for read-only export. */ readonly text: string;
  /** Envelope identity. */ readonly sha256: string;
  /** Original capture day. */ readonly observedAt: string;
  /** Distinct ordered route profiles. */ readonly profiles: readonly BackendProfile[];
  /** Optional metadata binding from the original producer. */ readonly binding: ProfileBinding | null;
}
/** A failed admission carries no replacement snapshot. */
export type BackendProfilesResult = {
  /** Whether exact native metadata was admitted. */ readonly ok: true;
  /** Immutable admitted values and original input text. */ readonly value: BackendProfileSnapshot;
} | {
  /** Whether exact native metadata was admitted. */ readonly ok: false;
  /** Observable refusal preserving prior selection. */ readonly message: string;
};

function require(value: boolean): void { if (!value) throw new Error("Profile metadata refused"); }
function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  require(typeof value === "object" && value !== null && !Array.isArray(value));
  const row = value as Record<string, unknown>;
  require(Object.keys(row).length === keys.length && keys.every(key => Object.hasOwn(row, key)));
  return row;
}
function text(value: unknown): string {
  require(typeof value === "string" && value.trim().length > 0 && [...value].length <= 512 && !/[\u0000-\u001f\u007f-\u009f\ud800-\udfff]/u.test(value));
  return value as string;
}
function hash(value: unknown): string { const result = text(value); require(/^[0-9a-f]{64}$/.test(result)); return result; }
function day(value: unknown): string {
  const result = text(value); require(/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(result));
  const parsed = new Date(result + "T00:00:00Z");
  require(Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === result && !result.startsWith("0000"));
  return result;
}
function nullable<T>(value: unknown, parse: (item: unknown) => T): T | null { return value === null ? null : parse(value); }
function bool(value: unknown): boolean { require(typeof value === "boolean"); return value as boolean; }
function count(value: unknown, positive = false): bigint {
  require(typeof value === "bigint" && value >= (positive ? 1n : 0n) && value <= 2n ** 64n - 1n);
  return value as bigint;
}
function strings(value: unknown): readonly string[] {
  require(Array.isArray(value) && value.length <= 256);
  const result = (value as unknown[]).map(text); require(new Set(result).size === result.length); return Object.freeze(result);
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}

/** Admit exact native envelopes and row digests; never run provider discovery. */
export async function parseBackendProfiles(raw: string): Promise<BackendProfilesResult> {
  try {
    require(raw.length <= MAX_PROFILE_BYTES && new TextEncoder().encode(raw).length <= MAX_PROFILE_BYTES);
    const wire = object(readJson(raw), ["schema", "body", "extensions", "sha256"]);
    require(wire["schema"] === BACKEND_PROFILES_SCHEMA);
    object(wire["extensions"], []);
    const body = object(wire["body"], ["observed_at", "no_submit", "profiles", "binding"]);
    const observedAt = day(body["observed_at"]); require(body["no_submit"] === true);
    const rows = body["profiles"]; require(Array.isArray(rows) && rows.length > 0 && rows.length <= 256);
    const profiles: BackendProfile[] = [];
    for (const input of rows as unknown[]) {
      const row = object(input, ["body", "sha256"]), identity = hash(row["sha256"]);
      const data = object(row["body"], ["route_id", "provider", "broker", "device", "backend_id", "modality", "observed_at", "region", "sdk_package", "ir_formats", "credential_refs", "declared", "observed", "options", "verbs"]);
      for (const key of ["route_id", "provider", "device", "backend_id", "modality", "sdk_package"]) text(data[key]);
      nullable(data["broker"], text); require(data["broker"] !== "direct"); nullable(data["region"], text);
      require(day(data["observed_at"]) === observedAt); strings(data["ir_formats"]);
      const credentials = nullable(data["credential_refs"], strings);
      require(credentials === null || credentials.every(ref => /^credential-ref:[0-9a-f]{64}$/.test(ref)));
      const declared = object(data["declared"], [...declaredCapabilities, "max_qubits"]);
      declaredCapabilities.forEach(key => bool(declared[key])); nullable(declared["max_qubits"], item => count(item, true));
      const observed = object(data["observed"], ["online", "n_qubits", "max_shots", "max_circuits", "queue_depth", "calibration_timestamp", "calibration_ref", "ir_formats", "basis_gates", "native_features"]);
      for (const key of ["ir_formats", "basis_gates", "native_features"]) nullable(observed[key], strings);
      nullable(observed["online"], bool); nullable(observed["n_qubits"], item => count(item, true));
      for (const key of ["max_shots", "max_circuits"]) nullable(observed[key], item => count(item, true));
      nullable(observed["queue_depth"], count);
      const timestamp = nullable(observed["calibration_timestamp"], text);
      const calibration = nullable(observed["calibration_ref"], hash);
      require((timestamp === null) === (calibration === null));
      if (timestamp !== null) {
        require(/^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$/.test(timestamp));
        day(timestamp.slice(0, 10)); require(Number.isFinite(Date.parse(timestamp)) && timestamp.slice(11, 19) === new Date(timestamp).toISOString().slice(11, 19));
      }
      const options = object(data["options"], ["pulse", "analog"]);
      for (const name of ["pulse", "analog"]) {
        const option = object(options[name], ["supported", "reason"]);
        require(bool(option["supported"]) === declared["supports_" + name]); text(option["reason"]);
      }
      const verbs = data["verbs"]; require(Array.isArray(verbs) && verbs.length === routeVerbs.length);
      (verbs as unknown[]).forEach((input, index) => {
        const verb = object(input, ["verb", "declared", "declared_source", "declared_on", "observed", "observed_on", "conformance_owner"]);
        require(verb["verb"] === routeVerbs[index]); nullable(verb["declared"], bool); nullable(verb["observed"], bool);
        nullable(verb["declared_source"], text); nullable(verb["conformance_owner"], text);
        nullable(verb["declared_on"], day); nullable(verb["observed_on"], day);
        require(verb["declared"] !== true || (verb["declared_source"] !== null && verb["declared_on"] !== null && verb["conformance_owner"] !== null));
        require(verb["observed"] !== true || (verb["declared"] !== false && verb["observed_on"] !== null && verb["conformance_owner"] !== null));
      });
      require(await canonicalDigest("studio.backend-profile.v1", data) === identity);
      profiles.push(freeze({ ...data, sha256: identity } as unknown as BackendProfile));
    }
    require(new Set(profiles.map(row => row.route_id)).size === profiles.length);
    let binding: ProfileBinding | null = null;
    if (body["binding"] !== null) {
      const refs = object(body["binding"], ["profile_sha256", "plan_ref", "calibration_ref", "approval_ref"]);
      hash(refs["profile_sha256"]); for (const key of ["plan_ref", "calibration_ref", "approval_ref"]) nullable(refs[key], hash);
      require(profiles.some(row => row.sha256 === refs["profile_sha256"])); binding = freeze(refs as unknown as ProfileBinding);
    }
    const identity = hash(wire["sha256"]);
    require(await canonicalDigest(BACKEND_PROFILES_SCHEMA, { schema: wire["schema"], body, extensions: wire["extensions"] }) === identity);
    return { ok: true, value: freeze({ text: raw, sha256: identity, observedAt, profiles, binding }) };
  } catch { return { ok: false, message: "Backend profile metadata refused; prior selection and references remain unchanged." }; }
}

/** Source date age in whole days; a future date stays explicitly observable. */
export function profileAge(observedAt: string, now = new Date()): string {
  const age = Math.floor((now.getTime() - Date.parse(day(observedAt) + "T00:00:00Z")) / 86400000);
  return age < 0 ? "future snapshot date" : `${age} day(s), date precision`;
}
/** Switching or refreshing a row clears all dependent metadata references. */
export function selectBackendProfile(current: ProfileBinding | null, profile: BackendProfile): ProfileBinding {
  return current?.profile_sha256 === profile.sha256 ? current : Object.freeze({ profile_sha256: profile.sha256, plan_ref: null, calibration_ref: null, approval_ref: null });
}
