// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native dossier identities and refusal cases

import { expect, it } from "vitest";
import native from "../../../../../data/studio/operator_review_dossier.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import { MAX_REVIEW_EXPORT_BYTES, parseOperatorDossier, reviewDocument, reviewStatus } from "./operatorDossier";
import type { HumanReview, OperatorDossierSnapshot } from "./operatorDossier";

type Wire = { schema: string; body: Record<string, unknown>; extensions: Record<string, unknown>; sha256: string };
const row = (value: unknown) => value as Record<string, unknown>;
const bundle = () => readJson(native) as Wire;
const dossier = (wire: Wire) => readJson(String(wire.body["dossier_text"])) as Wire;

/** Test-only source envelope sealing; native CLI parity is exercised in Chromium. */
export async function changedExample(change: (body: Record<string, unknown>) => void): Promise<string> {
  const wire = bundle(), source = dossier(wire); change(source.body);
  for (const [field, identity, domain] of [["plan", "plan_sha256", "execution_plan.v1"], ["profile", "profile_sha256", "backend_profile.v1"], ["settings", "settings_sha256", "resolved_settings.v1"], ["semantic_settings", "semantic_settings_sha256", "operator_review_settings.v1"], ["policy_decision", "policy_decision_sha256", "operator_policy_decision.v1"]]) source.body[identity!] = await canonicalDigest(domain!, source.body[field!]);
  const keys = ["plan_sha256", "profile_sha256", "workload_sha256", "payload", "semantic_settings_sha256", "policy_decision_sha256", "calibration", "created_at", "expires_at"];
  source.body["execution_sha256"] = await canonicalDigest("studio.operator-review-execution.v1", Object.fromEntries(keys.map(key => [key, source.body[key]])));
  source.sha256 = await canonicalDigest(source.schema, { schema: source.schema, body: source.body, extensions: source.extensions });
  wire.body["dossier_text"] = writeJson(source); wire.body["dossier_sha256"] = source.sha256;
  wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions }); return writeJson(wire);
}

/** Require an actual parser result rather than replacing admission with a mock. */
export async function admitted(raw = native): Promise<OperatorDossierSnapshot> {
  const result = await parseOperatorDossier(raw); expect(result.ok).toBe(true); if (!result.ok) throw new Error(result.message); return result.value;
}

it("preserves native integer precision, script bytes, dates and immutable original source", async () => {
  const value = await admitted(), source = dossier(bundle());
  expect(value.text).toBe(native); expect(value.dossierText).toBe(bundle().body["dossier_text"]);
  expect(value.sha256).toBe(source.sha256);
  expect(row(value.policy.settings.body["effective"])["seed"]).toBe(9007199254740993n);
  expect(value.script).toBe(row(bundle().body["script"])["source"]);
  expect(Object.isFrozen(value.body)).toBe(true);
});

it("test_operator_review_dossiers_02: no denied, expired, future or source-refused review is approved", async () => {
  const value = await admitted(), instant = new Date("2026-10-04T12:00:00Z");
  const record: HumanReview = { dossierSha256: value.sha256, executionSha256: value.executionSha256, choice: "approved", recordedAt: instant.toISOString().slice(0, 19) + "Z" };
  expect(reviewStatus(value, null, instant)).toBe("pending"); expect(reviewStatus(value, record, instant)).toBe("approved");
  expect(reviewStatus(value, { ...record, choice: "denied" }, instant)).toBe("denied");
  expect(reviewStatus(value, record, new Date("2026-10-05T00:00:00Z"))).toBe("expired");
  expect(reviewStatus(value, null, new Date("2026-10-03T23:59:59Z"))).toBe("not_yet_valid");
  expect(reviewStatus(value, { ...record, choice: "denied", recordedAt: "2026-10-04T13:00:00Z" }, instant)).toBe("not_yet_valid");
  expect(reviewStatus(value, { ...record, executionSha256: "f".repeat(64) }, instant)).toBe("invalidated");
  const refused = await admitted(await changedExample(body => { const decision = row(body["policy_decision"]); decision["allowed"] = false; decision["reasons"] = ["price_unknown"]; row(decision["estimate"])["amount"] = null; }));
  expect(reviewStatus(refused, null, instant)).toBe("refused");
  const document = row(readJson(await reviewDocument(record))); expect(row(document["body"])["dossier_sha256"]).toBe(value.sha256);
  expect(row(document["body"])["no_submit"]).toBe(true);
});

it("admits explicit absent calibration and price without inventing source metadata", async () => {
  const value = await admitted(await changedExample(body => { body["calibration"] = null; delete row(row(body["plan"])["parameters"])["calibration_ref"]; const decision = row(body["policy_decision"]); decision["estimate"] = null; decision["allowed"] = false; decision["reasons"] = ["price_unknown"]; }));
  expect(value.body["calibration"]).toBeNull(); expect(value.policy.decision.estimate).toBeNull();
});

it("admits the native optional positive shot capacity without losing integer precision", async () => {
  const value = await admitted(await changedExample(body => { row(row(body["profile"])["capabilities"])["max_shots"] = 9007199254740993n; }));
  expect(row(row(value.body["profile"])["capabilities"])["max_shots"]).toBe(9007199254740993n);
});

const damages: [string, (body: Record<string, unknown>) => void][] = [
  ["authority claim", body => { body["no_submit"] = false; }],
  ["extra field", body => { body["credential"] = "refuse"; }],
  ["producer", body => { body["producer_identity"] = "browser"; }],
  ["policy schema", body => { row(body["settings"])["schema"] = "resolved_settings.v2"; }],
  ["plan verb", body => { row(body["plan"])["verb"] = "run"; }],
  ["changed contract", body => { row(row(body["plan"])["contract"])["requires_approval"] = false; }],
  ["undeclared backend", body => { row(body["plan"])["backend"] = "cloud-other"; }],
  ["missing steps", body => { row(body["plan"])["steps"] = []; }],
  ["float shots", body => { row(row(body["plan"])["parameters"])["shots"] = 1024.0; }],
  ["provider substitution", body => { row(row(body["plan"])["parameters"])["provider"] = "other"; }],
  ["target substitution", body => { row(row(body["plan"])["parameters"])["endpoint"] = "other"; }],
  ["logical identity", body => { body["workload_sha256"] = "a".repeat(64); }],
  ["parent path", body => { row(body["payload"])["reference"] = "../payload"; }],
  ["absolute path", body => { row(body["payload"])["reference"] = "/payload"; }],
  ["URL path", body => { row(body["payload"])["reference"] = "https://example.invalid/payload"; }],
  ["query path", body => { row(body["payload"])["reference"] = "payload?key"; }],
  ["backslash path", body => { row(body["payload"])["reference"] = "a\\payload"; }],
  ["payload hash", body => { row(body["payload"])["sha256"] = "A".repeat(64); }],
  ["empty payload", body => { row(body["payload"])["size_bytes"] = 0n; }],
  ["payload bool size", body => { row(body["payload"])["size_bytes"] = true; }],
  ["oversized payload", body => { row(body["payload"])["size_bytes"] = 1048577n; }],
  ["source expired at creation", body => { body["expires_at"] = body["created_at"]; }],
  ["impossible UTC", body => { body["created_at"] = "2026-02-30T00:00:00Z"; }],
  ["local time", body => { body["created_at"] = "2026-10-04"; }],
  ["extended expiry", body => { body["expires_at"] = "2026-10-06T00:00:00Z"; }],
  ["wrong calibration", body => { row(body["calibration"])["target"] = "other"; }],
  ["future calibration", body => { row(body["calibration"])["observed_at"] = "2026-10-04T00:00:01Z"; }],
  ["invalid capability", body => { row(row(body["profile"])["capabilities"])["supports_shots"] = 1n; }],
  ["invalid capacity", body => { row(row(body["profile"])["capabilities"])["max_qubits"] = -1n; }],
  ["zero shot capacity", body => { row(row(body["profile"])["capabilities"])["max_shots"] = 0n; }],
  ["boolean shot capacity", body => { row(row(body["profile"])["capabilities"])["max_shots"] = true; }],
  ["null shot capacity", body => { row(row(body["profile"])["capabilities"])["max_shots"] = null; }],
  ["profile region", body => { row(body["profile"])["region"] = "other"; }],
  ["native profile note", body => { row(body["profile"])["notes"] = [false]; }],
  ["semantic origin", body => { row(row(body["semantic_settings"])["origins"])["shots"] = "default"; }],
];
for (const [name, change] of damages) it("refuses " + name + " despite enclosing hashes", async () => { expect((await parseOperatorDossier(await changedExample(change))).ok).toBe(false); });

it("refuses malformed syntax, unsupported versions, stale hashes, script substitutions and UTF-8 limits", async () => {
  for (const raw of ["null", "[]", "{", '{"schema":"a","schema":"b"}', "x".repeat(MAX_REVIEW_EXPORT_BYTES + 1), "α".repeat(MAX_REVIEW_EXPORT_BYTES)]) expect((await parseOperatorDossier(raw)).ok).toBe(false);
  for (const field of ["schema", "sha256", "extensions", "body"] as const) {
    const wire = bundle(); (wire as unknown as Record<string, unknown>)[field] = "unknown";
    expect((await parseOperatorDossier(writeJson(wire))).ok).toBe(false);
  }
  const wire = bundle(); row(wire.body["script"])["source"] = "raise SystemExit(0)";
  wire.sha256 = await canonicalDigest(wire.schema, { schema: wire.schema, body: wire.body, extensions: wire.extensions });
  expect((await parseOperatorDossier(writeJson(wire))).ok).toBe(false);
});
