// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native profile contract conformance
import { expect, it } from "vitest";
import native from "../../../../../data/studio/backend_profiles.json?raw";
import type CaseShape from "../../../../../data/studio/backend_profiles_cases.json";
import casesRaw from "../../../../../data/studio/backend_profiles_cases.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import { BACKEND_PROFILES_SCHEMA, MAX_PROFILE_BYTES, parseBackendProfiles, profileAge, selectBackendProfile } from "./backendProfiles";

const cases = readJson(casesRaw) as typeof CaseShape;
interface Envelope { schema: string; body: { observed_at: string; no_submit: boolean; profiles: { body: Record<string, unknown>; sha256: string }[]; binding: Record<string, unknown> | null }; extensions: Record<string, unknown>; sha256: string }
async function changed(change: (value: Envelope) => void, original = native): Promise<string> {
 const envelope = readJson(original) as Envelope; change(envelope);
 for (const row of envelope.body.profiles) row.sha256 = await canonicalDigest("studio.backend-profile.v1", row.body);
 envelope.sha256 = await canonicalDigest(envelope.schema, {schema:envelope.schema,body:envelope.body,extensions:envelope.extensions});
 return writeJson(envelope);
}

it("binds original native digests, exact bigint limits, detached immutable metadata and raw text", async () => {
 for (const [name, value] of Object.entries(cases)) {
  if (typeof value !== "object") continue;
  const raw = writeJson(value), result = await parseBackendProfiles(raw);
  expect(result.ok, name).toBe(true);
  if (!result.ok) throw new Error("Original native profile refused");
  expect(result.value.text).toBe(raw); expect(result.value.sha256).toBe(value.sha256);
  expect(Object.isFrozen(result.value.profiles[0]!.observed)).toBe(true);
 }
 const result = await parseBackendProfiles(writeJson(cases.offline));
 if (!result.ok) throw new Error("Offline original refused");
 expect(result.value.profiles[0]!.observed.max_shots).toBe(18446744073709551615n);
 expect(result.value.profiles[0]!.observed.queue_depth).toBe(0n);
 expect(result.value.profiles[0]!.observed.online).toBe(false);
 expect(profileAge("2026-10-03", new Date("2026-10-05T23:59:59Z"))).toBe("2 day(s), date precision");
 expect(profileAge("2026-10-03", new Date("2026-10-02T00:00:00Z"))).toBe("future snapshot date");
});

it("test_operator_backend_profiles_03: selecting a different device clears all references", async () => {
 const result = await parseBackendProfiles(writeJson(cases.offline));
 if (!result.ok) throw new Error("Offline original refused");
 const binding = result.value.binding!;
 const current = result.value.profiles.find(row => row.sha256 === binding.profile_sha256)!;
 expect(selectBackendProfile(binding, current)).toBe(binding);
 const other = result.value.profiles.find(row => row.sha256 !== current.sha256)!;
 expect(selectBackendProfile(binding,other)).toEqual({profile_sha256:other.sha256,plan_ref:null,calibration_ref:null,approval_ref:null});
 expect(selectBackendProfile(null,current).plan_ref).toBeNull();
});

it.each(["", "null", "[]", "{}", '{"schema":"unsupported"}', '{"x":1,"x":2}', "x".repeat(MAX_PROFILE_BYTES + 1), "α".repeat(MAX_PROFILE_BYTES)])("refuses malformed transport and resource excess without a replacement", async raw => {
 expect((await parseBackendProfiles(raw)).ok).toBe(false);
});

const invalid: [string, (wire: Envelope) => void][] = [
 ["future major", wire => {wire.schema = "studio.backend-profiles.v2";}],
 ["unknown extensions", wire => {wire.extensions["credential_value"] = "unsafe";}],
 ["execution flag", wire => {wire.body.no_submit = false;}],
 ["bad date", wire => {wire.body.observed_at = "2026-02-29";}],
 ["year zero", wire => {wire.body.observed_at = "0000-01-01";}],
 ["ambiguous date", wire => {wire.body.observed_at = "2026-10-3";}],
 ["non-date", wire => {wire.body.observed_at = "2026-10-03T00:00:00Z";}],
 ["empty", wire => {wire.body.profiles = [];}],
 ["too many", wire => {wire.body.profiles = Array.from({length:257}, () => wire.body.profiles[0]!);} ],
 ["duplicate routes", wire => {wire.body.profiles[1]!.body["route_id"] = wire.body.profiles[0]!.body["route_id"];}],
 ["missing field", wire => {delete wire.body.profiles[0]!.body["provider"];}],
 ["extra secret", wire => {wire.body.profiles[0]!.body["api_key"] = "must-refuse";}],
 ["wrong date binding", wire => {wire.body.profiles[0]!.body["observed_at"] = "2026-10-02";}],
 ["direct broker token", wire => {wire.body.profiles[0]!.body["broker"] = "direct";}],
 ["opaque credential value", wire => {wire.body.profiles[0]!.body["credential_refs"] = ["token-secret"];}],
 ["duplicate credential references", wire => {wire.body.profiles[0]!.body["credential_refs"] = ["credential-ref:"+"a".repeat(64),"credential-ref:"+"a".repeat(64)];}],
 ["missing binding identity", wire => {wire.body.binding = {profile_sha256:"a".repeat(64), plan_ref:null, calibration_ref:null, approval_ref:null};}],
 ["wrong binding shape", wire => {wire.body.binding = {profile_sha256:"a".repeat(64)};}],
];
for (const value of [null, true, 4n, "", " ", "α".repeat(513), "bad\ntext"]) invalid.push(["invalid route text", wire => {wire.body.profiles[0]!.body["route_id"] = value;}]);
for (const value of [null, "text", {}, ["x", "x"], Array.from({length:257},()=>"ir")]) invalid.push(["invalid IR array", wire => {wire.body.profiles[0]!.body["ir_formats"] = value;}]);
for (const value of [1n, null, "false"]) invalid.push(["nonboolean declaration", wire => {(wire.body.profiles[0]!.body["declared"] as Record<string,unknown>)["supports_pulse"] = value;}]);
for (const value of [-1n, 0n, 18446744073709551616n, 1.5, true]) invalid.push(["bad observed count", wire => {(wire.body.profiles[0]!.body["observed"] as Record<string,unknown>)["n_qubits"] = value;}]);
invalid.push(["option contradicts source", wire => {((wire.body.profiles[0]!.body["options"] as Record<string,Record<string,unknown>>)["pulse"]!)["supported"] = true;}]);
invalid.push(["missing verb", wire => {(wire.body.profiles[0]!.body["verbs"] as unknown[]).pop();}]);
invalid.push(["reordered verbs", wire => {(wire.body.profiles[0]!.body["verbs"] as unknown[]).reverse();}]);
for (const [key,value] of [["declared",true],["observed",true],["observed",1n],["declared_on","2026-02-29"]] as const) invalid.push(["missing verb provenance", wire => {((wire.body.profiles[0]!.body["verbs"] as Record<string,unknown>[])[0]!)[key]=value;}]);
it.each(invalid)("refuses rehashed malformed source metadata: %s", async (_label,change) => {
 expect((await parseBackendProfiles(await changed(change))).ok).toBe(false);
});

it("rejects a stale envelope or original row binding even when outer metadata is valid", async () => {
 const wire = JSON.parse(native) as Envelope; wire.sha256 = "0".repeat(64);
 expect((await parseBackendProfiles(JSON.stringify(wire))).ok).toBe(false);
 const row = JSON.parse(native) as Envelope; row.body.profiles[0]!.sha256 = "0".repeat(64);
 row.sha256 = await canonicalDigest(BACKEND_PROFILES_SCHEMA,{schema:row.schema,body:row.body,extensions:row.extensions});
 expect((await parseBackendProfiles(JSON.stringify(row))).ok).toBe(false);
});

it.each(["2026-02-29T00:00:00Z","2026-10-02T24:00:00Z","2026-10-02T01:00:00+00:00","", "2026-10-02T01:00:00.1234567Z"])("refuses malformed calibration time %s", async timestamp => {
 const raw = await changed(wire => {(wire.body.profiles[0]!.body["observed"] as Record<string,unknown>)["calibration_timestamp"] = timestamp;}, writeJson(cases.offline));
 expect((await parseBackendProfiles(raw)).ok).toBe(false);
});
it("requires calibration timestamp and reference together", async () => {
 const raw = await changed(wire => {(wire.body.profiles[0]!.body["observed"] as Record<string,unknown>)["calibration_ref"] = null;}, writeJson(cases.offline));
 expect((await parseBackendProfiles(raw)).ok).toBe(false);
});

it("refuses an invalid escaped Unicode scalar before digest admission", async () => {
 const wire=JSON.parse(native) as Envelope; wire.body.profiles[0]!.body["route_id"]="\ud800";
 expect((await parseBackendProfiles(JSON.stringify(wire))).ok).toBe(false);
});
