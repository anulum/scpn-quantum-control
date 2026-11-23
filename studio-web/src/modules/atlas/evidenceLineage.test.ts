// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence lineage admission tests

import { describe, expect, it } from "vitest";
import lineageText from "../../../../data/studio/evidence_lineage_20261123.json?raw";
import type { AtlasNode } from "../../shared/contracts/atlas";
import {
  type EvidenceLineage,
  admitCommittedEvidenceLineage,
  admitEvidenceLineage,
  evidenceLineageSchema,
  lineageRecordOf,
  verifyLineageRecord,
} from "./evidenceLineage";

type Json = Record<string, unknown>;

function committed(): Json {
  return JSON.parse(lineageText) as Json;
}
function admitted(): EvidenceLineage {
  const admission = admitCommittedEvidenceLineage();
  if (!admission.ok) throw new Error(admission.reason);
  return admission.lineage;
}
function reason(document: unknown): string {
  const admission = admitEvidenceLineage(
    typeof document === "string" ? document : JSON.stringify(document),
  );
  if (admission.ok) throw new Error("expected a refusal");
  return admission.reason;
}
function firstClaim(document: Json): Json {
  return (document["claims"] as Json[])[0] as Json;
}
function node(id: string, sourceRef: string, sourceDigest: string | null): AtlasNode {
  return {
    id,
    kind: "artefact",
    label: id,
    sourceRef,
    sourceDigest,
    validityDomain: null,
    asOf: null,
    access: "public",
    status: "current",
    evidenceRefs: [],
    nullReasons: [],
  };
}

describe("Evidence lineage admission", () => {
  it("admits the committed lineage with its claims, packs and failed replication", () => {
    const lineage = admitted();
    expect(lineage.claims.size).toBe(16);
    expect(lineage.packs.size).toBe(5);
    const [replication] = [...lineage.replications.values()];
    expect(replication?.decisionField).toBe("negative_sign_replicates");
    expect(replication?.decision).toBe(false);
    const negative = lineage.packs.get("scpn_fim_negative_ibm_kingston_2026_05_05");
    expect(negative?.manifestStatus).toBe("promoted_negative_result");
    for (const claim of lineage.claims.values()) {
      expect(claim.certificate).toBeNull();
      expect(claim.absentReason).toContain("no reference-validation certification");
    }
  });

  it("refuses text, schemas and records that do not hold", () => {
    expect(reason("{not json")).toContain("the lineage is not JSON");
    expect(reason({ schema: "other" })).toContain("unknown schema");
    expect(reason({ schema: evidenceLineageSchema, claims: {} })).toContain(
      "$.claims must be an array",
    );
    const laundered = committed();
    ((firstClaim(laundered)["bundle"] as Json)["claim_boundary"] as Json)["status"] =
      "reference-validated";
    expect(reason(laundered)).toContain("reference-validated without a certificate");
    const unknownState = committed();
    (firstClaim(unknownState)["certification"] as Json)["state"] = "pending";
    expect(reason(unknownState)).toContain("certification has an unknown state");
    const unknownEvidence = committed();
    ((firstClaim(unknownEvidence)["evidence"] as Json[])[0] as Json)["state"] = "guessed";
    expect(reason(unknownEvidence)).toContain("evidence[0].state is unknown");
    const blankText = committed();
    firstClaim(blankText)["claim_id"] = "";
    expect(reason(blankText)).toContain("$.claims[0].claim_id must be text");
    const badList = committed();
    ((badList["packs"] as Json[])[0] as Json)["non_claims"] = "none";
    expect(reason(badList)).toContain("non_claims must be an array");
    const badDecision = committed();
    (((badDecision["replications"] as Json[])[0] as Json)["decision"] as Json)["value"] = "false";
    expect(reason(badDecision)).toContain("decision.value must be a boolean");
    expect(reason({ ...committed(), claims: [null] })).toContain("$.claims[0] must be an object");
    expect(reason({ ...committed(), claims: [[]] })).toContain("$.claims[0] must be an object");
    const missingField = committed();
    delete ((missingField["packs"] as Json[])[0] as Json)["id"];
    expect(reason(missingField)).toContain("$.packs[0].id must be text");
  });

  it("admits a certified claim whose bundle is reference-validated", () => {
    const document = committed();
    const claim = firstClaim(document);
    ((claim["bundle"] as Json)["claim_boundary"] as Json)["status"] = "reference-validated";
    claim["certification"] = { state: "certified", record: { certificate_ref: "fixture" } };
    const admission = admitEvidenceLineage(JSON.stringify(document));
    if (!admission.ok) throw new Error(admission.reason);
    const admittedClaim = [...admission.lineage.claims.values()][0];
    expect(admittedClaim?.certificate).toEqual({ certificate_ref: "fixture" });
    expect(admittedClaim?.absentReason).toBeNull();
  });
});

describe("Evidence lineage records", () => {
  it("maps claim, pack and replication nodes to their records and other nodes to none", () => {
    const lineage = admitted();
    const claimId = [...lineage.claims.keys()][0] as string;
    const [replication] = [...lineage.replications.values()];
    expect(
      lineageRecordOf(lineage, node(`claim:differentiable.claim_ledger/${claimId}`, "x", null))
        ?.kind,
    ).toBe("claim");
    expect(
      lineageRecordOf(
        lineage,
        node(
          "artefact:sha256:p",
          "data/hardware_result_packs/manifest.json#scpn_fim_negative_ibm_kingston_2026_05_05",
          "p",
        ),
      )?.kind,
    ).toBe("pack");
    expect(
      lineageRecordOf(lineage, node("artefact:x", "x", replication?.sha256 ?? null))?.kind,
    ).toBe("replication");
    expect(
      lineageRecordOf(lineage, node("claim:differentiable.claim_ledger/none", "x", null)),
    ).toBeNull();
    expect(
      lineageRecordOf(
        lineage,
        node("artefact:x", "data/hardware_result_packs/manifest.json#none", "p"),
      ),
    ).toBeNull();
    expect(lineageRecordOf(lineage, node("artefact:x", "x", null))).toBeNull();
    expect(lineageRecordOf(lineage, node("artefact:x", "x", "unknown"))).toBeNull();
  });

  it("recomputes each record's canonical digest and reports a changed record", async () => {
    const lineage = admitted();
    for (const record of [
      ...lineage.claims.values(),
      ...lineage.packs.values(),
      ...lineage.replications.values(),
    ])
      expect((await verifyLineageRecord(record)).display).toBe("match");
    const [claim] = [...lineage.claims.values()];
    if (claim === undefined) throw new Error("expected a claim");
    const changed = { ...claim, wire: { ...claim.wire, promotion_status: "promoted" } };
    const verdict = await verifyLineageRecord(changed);
    expect(verdict.display).toBe("mismatch");
    expect(verdict.detail).toContain(`differs from ${claim.recordDigest}`);
  });
});
