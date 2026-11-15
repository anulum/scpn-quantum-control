// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — committed attestation data and its fail-closed parser

import { expect, it } from "vitest";
import committedText from "../../../../data/studio/attestation_instrument_20261115.json?raw";
import { attestationInstrument, parseAttestationInstrument } from "./attestationInstrument";

type Raw = Record<string, unknown> & {
  cases: Record<string, unknown>[];
  canonical_oracle: Record<string, unknown>[];
};
const edited = (change: (raw: Raw) => void) => {
  const raw = JSON.parse(committedText) as Raw;
  change(raw);
  return parseAttestationInstrument(JSON.stringify(raw));
};

it("parses the committed cases, oracle and whitespace with exact number tokens", () => {
  expect(attestationInstrument.ok).toBe(true);
  if (!attestationInstrument.ok) return;
  const data = attestationInstrument.value;
  expect(data.id).toBe("studio-attestation-instrument-20261115");
  expect(data.source).toMatchObject({
    backend: "ibm_fez",
    path: "results/ibm_hardware_v2_2026-03-29/full_results.json",
  });
  expect(data.cases.map((item) => item.id)).toEqual([
    "intact",
    "changed-value",
    "rebound-axes",
    "stripped-axes",
    "missing-digest",
    "unverified-signature",
    "signature-required",
  ]);
  expect(data.cases[6]?.requireSignature).toBe(true);
  expect(data.cases[4]?.python.expectedDigest).toBe("");
  expect(data.whitespace.size).toBe(29);
  expect(data.oracle).toHaveLength(6);
});

it("refuses data whose schema, verdicts, oracle or whitespace are malformed", () => {
  expect(edited((raw) => (raw["schema"] = "other"))).toEqual({
    ok: false,
    reason: "Unsupported attestation instrument schema",
  });
  expect(
    edited(
      (raw) => ((raw.cases[0]?.["python_verdict"] as Record<string, unknown>)["status"] = "GREEN"),
    ),
  ).toEqual({ ok: false, reason: "Unknown attestation status GREEN" });
  expect(edited((raw) => (raw.cases[0] = { ...raw.cases[0], require_signature: "no" }))).toEqual({
    ok: false,
    reason: "require_signature must be true or false",
  });
  expect(
    edited((raw) => (raw.canonical_oracle[0] = { ...raw.canonical_oracle[0], digest: "x" })),
  ).toEqual({
    ok: false,
    reason: "Oracle digest must be lowercase SHA-256",
  });
  expect(edited((raw) => (raw["python_whitespace"] = []))).toEqual({
    ok: false,
    reason: "python_whitespace is empty",
  });
  for (const bad of [1.5, -1, 0x110000, "9"])
    expect(edited((raw) => (raw["python_whitespace"] = [bad]))).toEqual({
      ok: false,
      reason: "python_whitespace holds code points",
    });
  expect(parseAttestationInstrument("[")).toMatchObject({ ok: false });
});
