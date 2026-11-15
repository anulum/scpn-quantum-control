// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser attestation verdicts against the Python owner's verdicts

import { expect, it } from "vitest";
import { attestationInstrument } from "./attestationInstrument";
import {
  checkEnvelopeText,
  missingAxes,
  readEnvelope,
  verifyEnvelope,
} from "./attestationVerifier";
import type { AttestedEnvelope } from "./attestationVerifier";
import { pythonCanonicalText } from "./pythonCanonical";

const committed = () => {
  if (!attestationInstrument.ok) throw new Error(attestationInstrument.reason);
  return attestationInstrument.value;
};
const sealed = (): AttestedEnvelope => committed().cases[0]?.envelope as AttestedEnvelope;
const wire = (envelope: AttestedEnvelope) => ({
  schema: envelope.schema,
  claim_id: envelope.claimId,
  claim_axes: envelope.claimAxes,
  content: envelope.content,
  content_digest: envelope.contentDigest,
  signature: envelope.signature,
  claim_boundary: envelope.claimBoundary,
});

it("test_attestation_browser_01: every committed case gets the Python owner's verdict", async () => {
  const data = committed();
  expect(data.cases.map((item) => item.python.status)).toEqual([
    "VERIFIED",
    "FORGED",
    "FORGED",
    "STRIPPED",
    "UNGRADED",
    "UNGRADED",
    "UNGRADED",
  ]);
  for (const item of data.cases) {
    const here = await verifyEnvelope(item.envelope, item.requireSignature, data.whitespace);
    expect(here).toEqual(item.python);
  }
});

it("test_attestation_browser_02: a changed value or rebound axes no longer match", async () => {
  const data = committed();
  const changed = structuredClone(sealed());
  (changed.content["B_M+4"] as Record<string, unknown>)["mean"] = 0.9937;
  const forged = await verifyEnvelope(changed, false, data.whitespace);
  expect(forged.status).toBe("FORGED");
  expect(forged.observedDigest).toBe(sealed().contentDigest);
  const relabelled = { ...sealed(), claimAxes: { ...sealed().claimAxes, claim_unit: "other" } };
  expect((await verifyEnvelope(relabelled, false, data.whitespace)).status).toBe("FORGED");
  const undigestable = { ...sealed(), content: { value: "\ud800" } };
  expect(await verifyEnvelope(undigestable, false, data.whitespace)).toMatchObject({
    status: "UNGRADED",
    reason: "content not digestable: $.content.value: invalid Unicode scalar",
    expectedDigest: "",
  });
});

it("reads absent, null, blank and empty axes as missing, and an empty mapping as present", () => {
  const whitespace = committed().whitespace;
  expect(missingAxes({}, whitespace)).toEqual(["claim_unit", "honesty_axes", "backend_class"]);
  expect(
    missingAxes({ claim_unit: null, honesty_axes: [], backend_class: "\u001c　" }, whitespace),
  ).toEqual(["claim_unit", "honesty_axes", "backend_class"]);
  expect(
    missingAxes({ claim_unit: 0n, honesty_axes: {}, backend_class: false }, whitespace),
  ).toEqual([]);
});

it("refuses an envelope the owner would refuse to construct", () => {
  const whitespace = committed().whitespace;
  const base = wire(sealed());
  expect(() => readEnvelope([], whitespace)).toThrow("envelope must be a mapping");
  expect(() => readEnvelope(null, whitespace)).toThrow("envelope must be a mapping");
  expect(() => readEnvelope({ ...base, claim_axes: [] }, whitespace)).toThrow(
    "claim_axes and content must be mappings when present",
  );
  expect(() => readEnvelope({ ...base, content: "x" }, whitespace)).toThrow("mappings");
  expect(() => readEnvelope({ ...base, signature: 5n }, whitespace)).toThrow(
    "signature must be text",
  );
  expect(() => readEnvelope({ ...base, claim_id: " 　" }, whitespace)).toThrow(
    "claim_id must be non-empty",
  );
  expect(() => readEnvelope({ ...base, content_digest: "abc" }, whitespace)).toThrow(
    "content_digest must be empty or 64-char lowercase hex",
  );
  const minimal = readEnvelope(
    { claim_id: " c ", content_digest: ` ${"A".repeat(64)}\u0085` },
    whitespace,
  );
  expect(minimal).toEqual({
    schema: "",
    claimId: "c",
    claimAxes: {},
    content: {},
    contentDigest: "a".repeat(64),
    signature: "",
    claimBoundary: "",
  });
});

it("checks an envelope file's text, or says why it reached no verdict", async () => {
  const whitespace = committed().whitespace;
  // The owner's spelling keeps 0.0 a float; JSON.stringify would write 0 and change the digest.
  const text = pythonCanonicalText(wire(sealed()));
  expect(await checkEnvelopeText(JSON.stringify(wire(sealed())), false, whitespace)).toMatchObject({
    ok: true,
    verdict: { status: "FORGED" },
  });
  expect(await checkEnvelopeText(text, false, whitespace)).toMatchObject({
    ok: true,
    verdict: { status: "VERIFIED" },
  });
  expect(await checkEnvelopeText(text, true, whitespace)).toMatchObject({
    ok: true,
    verdict: { status: "UNGRADED" },
  });
  expect(await checkEnvelopeText("{", false, whitespace)).toMatchObject({ ok: false });
  expect(await checkEnvelopeText('{"claim_id":""}', false, whitespace)).toEqual({
    ok: false,
    reason: "claim_id must be non-empty",
  });
});
