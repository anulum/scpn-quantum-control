// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact native policy wire admission

import { describe, expect, it } from "vitest";
import native from "../../../../../data/studio/operator_policy_decisions.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import { MAX_OPERATOR_DECISION_BYTES, parseOperatorDecision } from "./policyDecision";

type Wire = { schema: string; body: Record<string, unknown>; extensions: Record<string, unknown>; sha256: string };
const record = (value: unknown) => value as Record<string, unknown>;
const copy = () => readJson(native) as Wire;
const decision = (value: Wire) => record(value.body["decision"]);
const settings = (value: Wire) => record(record(value.body["settings"])["body"]);
const policy = (value: Wire) => record(decision(value)["policy"]);
const request = (value: Wire) => record(decision(value)["request"]);
const estimate = (value: Wire) => record(decision(value)["estimate"]);

async function seal(value: Wire): Promise<string> {
  value.body["settings_sha256"] = await canonicalDigest("resolved_settings.v1", value.body["settings"]);
  value.sha256 = await canonicalDigest(value.schema, { schema: value.schema, body: value.body, extensions: value.extensions });
  return writeJson(value);
}

describe("source-owned operator decision admission", () => {
  it("preserves exact native bigint, source dates, immutable values and original bytes", async () => {
    const result = await parseOperatorDecision(native);
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(result.message);
    expect(result.value.text).toBe(native);
    expect(result.value.decision.allowed).toBe(false);
    expect(result.value.decision.reasons).toEqual(["price_unknown"]);
    expect(record(result.value.settings.body["effective"])["seed"]).toBe(9007199254740993n);
    expect(Object.isFrozen(result.value.decision.request)).toBe(true);
    expect(result.value.decision.estimate!["amount"]).toBeNull();
  });

  it("retains declared refusal inputs with mismatched or absent price without recomputing policy", async () => {
    const value = copy();
    estimate(value)["request_sha256"] = "a".repeat(64);
    decision(value)["reasons"] = ["price_unknown", "price_request_mismatch"];
    expect((await parseOperatorDecision(await seal(value))).ok).toBe(true);
    decision(value)["estimate"] = null;
    decision(value)["reasons"] = ["price_unknown"];
    expect((await parseOperatorDecision(await seal(value))).ok).toBe(true);
    const nullable = copy();
    request(nullable)["target"] = null; request(nullable)["region"] = null;
    for (const field of ["requested", "effective"]) {
      record(settings(nullable)[field])["device"] = null;
      record(settings(nullable)[field])["region"] = null;
    }
    expect((await parseOperatorDecision(await seal(nullable))).ok).toBe(true);
  });

  it("object key order never changes settings equivalence", async () => {
    const value = copy();
    settings(value)["requested"] = Object.fromEntries(Object.entries(record(settings(value)["requested"])).reverse());
    expect((await parseOperatorDecision(await seal(value))).ok).toBe(true);
  });

  const damages: [string, (value: Wire) => void][] = [
    ["future schema", value => { value.schema = "studio.operator-policy-decision.v2"; }],
    ["extra envelope field", value => { (value as unknown as Record<string, unknown>)["token"] = "refuse"; }],
    ["unknown extension", value => { value.extensions["extra"] = 1n; }],
    ["not metadata", value => { value.body["no_submit"] = false; }],
    ["wrong claim", value => { value.body["claim_boundary"] = "approved"; }],
    ["future settings", value => { record(value.body["settings"])["schema"] = "resolved_settings.v2"; }],
    ["malformed settings", value => { value.body["settings"] = null; }],
    ["invalid source date", value => { decision(value)["assessed_at"] = "2026-02-30T00:00:00Z"; }],
    ["local time", value => { decision(value)["assessed_at"] = "2026-10-04"; }],
    ["zero year", value => { decision(value)["assessed_at"] = "0000-01-01T00:00:00Z"; }],
    ["invalid verdict type", value => { decision(value)["allowed"] = 1n; }],
    ["contradictory verdict", value => { decision(value)["allowed"] = true; }],
    ["reason not a list", value => { decision(value)["reasons"] = "price_unknown"; }],
    ["repeated reasons", value => { decision(value)["reasons"] = ["price_unknown", "price_unknown"]; }],
    ["empty reason", value => { decision(value)["reasons"] = [""]; }],
    ["unowned substitution", value => { decision(value)["rejected_substitutions"] = ["missing_reason"]; }],
    ["too many reasons", value => { decision(value)["reasons"] = Array.from({length:257}, (_, i) => "r" + i); }],
    ["extra request field", value => { request(value)["secret"] = "refuse"; }],
    ["bad workload hash", value => { request(value)["workload_sha256"] = "A".repeat(64); }],
    ["non-text route", value => { request(value)["backend_id"] = 1n; }],
    ["control text", value => { request(value)["target"] = "\n"; }],
    ["long text", value => { request(value)["target"] = "x".repeat(257); }],
    ["unattended numeric", value => { request(value)["unattended"] = 1n; }],
    ["float shots", value => { request(value)["shots"] = 1.5; }],
    ["zero shots", value => { request(value)["shots"] = 0n; }],
    ["oversize shots", value => { request(value)["shots"] = 2n ** 63n; }],
    ["empty target list", value => { policy(value)["targets"] = []; }],
    ["bad ceiling", value => { policy(value)["max_concurrency"] = false; }],
    ["float cost", value => { policy(value)["max_cost"] = 12.5; }],
    ["currency", value => { policy(value)["currency"] = "usd"; }],
    ["zero-duration policy", value => { policy(value)["expires_at"] = policy(value)["valid_from"]; }],
    ["wrong policy source", value => { record(settings(value)["policy_ref"])["sha256"] = "f".repeat(64); }],
    ["substituted settings", value => { record(settings(value)["effective"])["shots"] = 3n; }],
    ["request differs", value => { request(value)["shots"] = 3n; }],
    ["invalid estimate", value => { decision(value)["estimate"] = []; }],
    ["fractional precision", value => { estimate(value)["amount"] = "0.0000000001"; }],
    ["zero price interval", value => { estimate(value)["expires_at"] = estimate(value)["observed_at"]; }],
  ];
  for (const [name, damage] of damages) {
    it("refuses " + name + " even with an enclosing digest", async () => {
      const value = copy(); damage(value);
      const raw = await seal(value);
      expect((await parseOperatorDecision(raw)).ok).toBe(false);
    });
  }
  it("refuses raw syntax, duplicate keys, byte bounds and altered original hashes", async () => {
    for (const raw of ["[]", "null", "{", '{"schema":"a","schema":"a"}', "x".repeat(MAX_OPERATOR_DECISION_BYTES + 1), "α".repeat(MAX_OPERATOR_DECISION_BYTES)]) expect((await parseOperatorDecision(raw)).ok).toBe(false);
    const value = copy(); value.sha256 = "f".repeat(64);
    expect((await parseOperatorDecision(writeJson(value))).ok).toBe(false);
    value.sha256 = copy().sha256; value.body["settings_sha256"] = "b".repeat(64);
    expect((await parseOperatorDecision(writeJson(value))).ok).toBe(false);
  });
});
