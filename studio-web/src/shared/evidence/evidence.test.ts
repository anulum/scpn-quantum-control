// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence projection tests
// @vitest-environment node

import { describe, expect, it } from "vitest";
import { projectEvidenceBundle, projectProgramAdEvidence } from "./index";
import { programAdUnit } from "../../panel/programAd";

function bundle() {
  return { schema: "studio.evidence-replay.v1", prov: {
    entity: { id: "synthetic-negative-metadata", digest: "sha256:" + "a".repeat(64) },
    activity: { regenerated_by: "synthetic metadata fixture", started: "2026-09-29T00:00:00Z" },
    agent: { operator: "synthetic-conformance" },
  }, evidence_kind: "falsified", claim_boundary: { status: "refuted", admission: "rejected", validity_domain: { note: "Synthetic projection fixture; no scientific result" } },
  freshness: "traceable-unchecked", substrate: "numerical-model",
  numeric_provenance: { parity: [{ exactness: "bit-exact" }] }, attestation: { signature: "unverified-test-metadata" } };
}
describe("source-owned evidence projection", () => {
  it("preserves native schema-B axes without upgrading an attested negative claim", () => {
    const source = bundle();
    const view = projectEvidenceBundle(source);
    expect(view).toMatchObject({ source: source.prov.entity.id, digest: source.prov.entity.digest,
      kind: "falsified", claim: "refuted", admission: "rejected", freshness: "traceable-unchecked",
      seal: "present-unverified", exactness: '[{"exactness":"bit-exact"}]', issues: [] });
    const identity = view.snapshotIdentity;
    source.attestation.signature = "changed";
    expect(view.snapshotIdentity).toBe(identity);
    expect(projectEvidenceBundle(source).snapshotIdentity).not.toBe(identity);
    expect(Object.isFrozen(view)).toBe(true);
    expect(Object.isFrozen(view.issues)).toBe(true);
  });
  it.each([null, [], 1, {}, { schema: " " }, { schema: "studio.unknown.v2", attestation: [] },
    { schema: "studio.hardware-result-pack.v1", numeric_provenance: { parity: [{}] } },
    { numeric_provenance: { parity: [] }, attestation: {} }])("makes partial and unsupported fields visible: %j", source => {
    const view = projectEvidenceBundle(source);
    expect(view.issues.length).toBeGreaterThan(0);
    expect(view.claim).toBeNull();
    expect(view.source).toBeNull();
  });
  it("does not invoke imported accessors or stringify unsupported values", () => {
    let invoked = false;
    const source = Object.defineProperty({}, "schema", { enumerable: true, get: () => { invoked = true; return "studio.evidence-replay.v1"; } });
    expect(projectEvidenceBundle(source).identifiable).toBe(false);
    expect(invoked).toBe(false);
    expect(projectEvidenceBundle({ extension: Infinity }).identifiable).toBe(false);
  });
  it("retains the original replay boundary and reports absent classifications", () => {
    if (!programAdUnit.ok) throw new Error(programAdUnit.reason);
    const view = projectProgramAdEvidence(programAdUnit.value);
    expect(view.boundary).toBe(programAdUnit.value.claimBoundary);
    expect(view.digest).toBe(programAdUnit.value.inputSha256);
    expect(view.kind).toBeNull();
    expect(view.freshness).toBeNull();
    expect(view.seal).toBe("missing");
  });
});
