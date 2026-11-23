// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence lineage view tests

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { EvidenceVerification } from "../../shared/evidence/EvidenceInspector";
import AtlasView from "./AtlasView";
import { EvidenceLineage } from "./EvidenceLineage";
import {
  type LineageClaim,
  type LineagePack,
  type LineageRecord,
  admitCommittedEvidenceLineage,
} from "./evidenceLineage";

afterEach(cleanup);

function lineage() {
  const admission = admitCommittedEvidenceLineage();
  if (!admission.ok) throw new Error(admission.reason);
  return admission.lineage;
}
const match = async (): Promise<EvidenceVerification> => ({
  display: "match",
  detail: "fixture match",
});
function axis(name: string): HTMLElement {
  return screen.getByText(name).nextElementSibling as HTMLElement;
}
function inspector(): HTMLElement {
  return screen.getByRole("region", { name: "Atlas inspector" });
}

describe("Evidence lineage", () => {
  it("test_atlas_evidence_lineage_01: a sealed falsified record stays falsified after verification", async () => {
    const negative = lineage().packs.get(
      "scpn_fim_negative_ibm_kingston_2026_05_05",
    ) as LineagePack;
    const sealed: LineagePack = {
      ...negative,
      bundle: {
        schema: "studio.hardware-result-pack.v1",
        prov: { entity: { id: "fixture:falsified", digest: `sha256:${"e".repeat(64)}` } },
        evidence_kind: "falsified",
        claim_boundary: { status: "refuted", admission: "admitted" },
        attestation: { signature: "fixture" },
      },
    };
    render(<EvidenceLineage record={sealed} verify={match} contextSuffix="" />);
    expect(axis("Manifest status").textContent).toBe("promoted_negative_result");
    expect(axis("Scientific kind").textContent).toBe("falsified");
    expect(axis("Scientific kind").getAttribute("data-negative")).toBe("true");
    expect(axis("Claim status").textContent).toBe("refuted");
    expect(screen.getByText("Seal present — not verified")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Verify lineage record" }));
    expect((await screen.findByRole("status")).textContent).toBe("fixture match");
    expect(axis("Scientific kind").textContent).toBe("falsified");
    expect(axis("Claim status").textContent).toBe("refuted");
  });

  it("test_atlas_evidence_lineage_02: a missing certificate cannot appear reference-validated", () => {
    const [claim] = [...lineage().claims.values()] as LineageClaim[];
    render(<EvidenceLineage record={claim as LineageClaim} verify={match} contextSuffix="" />);
    const panel = screen.getByRole("region", { name: "Evidence lineage" });
    expect(axis("Reference-validation certificate").textContent).toBe(
      `Absent: ${claim?.absentReason}`,
    );
    expect(axis("Claim status").textContent).toBe("bounded-model");
    expect(panel.textContent).not.toContain("reference-validated");
    expect(
      within(panel).getByRole("link", { name: "Verify an attestation" }).getAttribute("href"),
    ).toBe("#/evidence/attestation");
  });

  it("shows a committed certificate as it is recorded", () => {
    const [claim] = [...lineage().claims.values()];
    if (claim === undefined) throw new Error("expected a claim");
    const certified = { ...claim, certificate: { certificate_ref: "fixture" }, absentReason: null };
    render(<EvidenceLineage record={certified} verify={match} contextSuffix="" />);
    expect(axis("Reference-validation certificate").textContent).toBe(
      'Certified: {"certificate_ref":"fixture"}',
    );
  });

  it("verifies a replication record and shows its decision verbatim", async () => {
    const [replication] = [...lineage().replications.values()] as LineageRecord[];
    render(
      <EvidenceLineage record={replication as LineageRecord} verify={match} contextSuffix="?p=1" />,
    );
    expect(axis("negative_sign_replicates").textContent).toBe("false");
    expect(axis("negative_sign_replicates").getAttribute("data-negative")).toBe("true");
    fireEvent.click(screen.getByRole("button", { name: "Verify lineage record" }));
    expect((await screen.findByRole("status")).getAttribute("data-verdict")).toBe("match");
    expect(
      screen.getByRole("link", { name: "Open the evidence catalogue" }).getAttribute("href"),
    ).toBe("#/evidence?p=1");
  });

  it("shows a replicated decision without marking it negative", () => {
    const [replication] = [...lineage().replications.values()];
    if (replication === undefined) throw new Error("expected a replication");
    render(
      <EvidenceLineage
        record={{ ...replication, decision: true }}
        verify={match}
        contextSuffix=""
      />,
    );
    expect(axis("negative_sign_replicates").getAttribute("data-negative")).toBe("false");
  });
});

describe("Evidence lineage in the Atlas", () => {
  async function openLens(lens: "claims" | "evidence", verifyRecord = match): Promise<void> {
    render(<AtlasView instrument={lens} contextSuffix="" verifyRecord={verifyRecord} />);
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
  }
  function selectRow(label: string): void {
    fireEvent.click(
      within(screen.getByRole("table", { name: "Atlas nodes" })).getByRole("button", {
        name: label,
      }),
    );
  }

  it("test_atlas_evidence_lineage_03: a failed replication remains searchable", async () => {
    await openLens("evidence");
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "replicates = false" } });
    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Atlas summary" }).textContent).toBe(
        "Showing 1 of 1 nodes.",
      ),
    );
    selectRow("Replication iqm_dla_d10_retest_prereg_2026-07-22: negative_sign_replicates = false");
    expect(
      within(inspector()).getByText("negative_sign_replicates").nextElementSibling?.textContent,
    ).toBe("false");
  }, 30_000);

  it("test_atlas_evidence_lineage_04: selecting a claim opens its exact source hashes and missing custody", async () => {
    const claims = [...lineage().claims.values()];
    const withFile = claims.find((item) =>
      item.evidence.some((entry) => entry.state === "resolved"),
    ) as LineageClaim;
    const withGap = claims.find((item) =>
      item.evidence.some((entry) => entry.state === "missing"),
    ) as LineageClaim;
    const resolved = withFile.evidence.find((entry) => entry.state === "resolved");
    await openLens("claims");
    selectRow(String(withFile.wire["claim_text"]));
    const evidence = within(inspector()).getByRole("table", { name: "Declared evidence" });
    expect(evidence.textContent).toContain(resolved?.candidates[0]?.sha256);
    expect(evidence.textContent).toContain(resolved?.candidates[0]?.path);
    selectRow(String(withGap.wire["claim_text"]));
    expect(
      within(inspector()).getByRole("table", { name: "Declared evidence" }).textContent,
    ).toContain("Missing custody: no committed file carries this identifier");
  }, 30_000);

  it("test_atlas_evidence_lineage_05: a verdict for an earlier selection is ignored", async () => {
    const pending: ((verdict: EvidenceVerification) => void)[] = [];
    const deferred = (): Promise<EvidenceVerification> =>
      new Promise((resolve) => pending.push(resolve));
    const [first, second] = [...lineage().claims.values()] as LineageClaim[];
    await openLens("claims", deferred);
    selectRow(String(first?.wire["claim_text"]));
    fireEvent.click(within(inspector()).getByRole("button", { name: "Verify lineage record" }));
    selectRow(String(second?.wire["claim_text"]));
    await act(async () => pending[0]?.({ display: "match", detail: "stale verdict" }));
    expect(within(inspector()).queryByText("stale verdict")).toBeNull();
    fireEvent.click(within(inspector()).getByRole("button", { name: "Verify lineage record" }));
    await act(async () => pending[1]?.({ display: "mismatch", detail: "current verdict" }));
    expect(within(inspector()).getByText("current verdict").getAttribute("data-verdict")).toBe(
      "mismatch",
    );
  }, 30_000);

  it("reports a refused lineage and then shows no lineage for any node", async () => {
    render(
      <AtlasView
        instrument="claims"
        contextSuffix=""
        lineage={() => ({ ok: false, reason: "The evidence lineage was refused: fixture." })}
      />,
    );
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The evidence lineage was refused: fixture.",
    );
    await screen.findByRole("table", { name: "Atlas nodes" }, { timeout: 15_000 });
    selectRow(String([...lineage().claims.values()][0]?.wire["claim_text"]));
    expect(within(inspector()).queryByRole("region", { name: "Evidence lineage" })).toBeNull();
  }, 30_000);
});
