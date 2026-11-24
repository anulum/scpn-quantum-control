// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — claim comparison record admission and panel tests

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import recordText from "../../../../data/studio/atlas_claim_comparisons_20261123.json?raw";
import { canonicalDigest } from "../../shared/contracts/canonical";
import { readJson } from "../../shared/contracts/jsonTransport";
import { ClaimComparisons } from "./ClaimComparisons";
import {
  type ClaimComparisonsAdmission,
  admitClaimComparisons,
  admitCommittedClaimComparisons,
  claimComparisonsRecordSchema,
} from "./claimComparisons";

afterEach(cleanup);

type Json = Record<string, unknown>;

async function resealed(change: (body: Json) => void): Promise<string> {
  const body = JSON.parse(recordText) as Json;
  delete body["record_digest"];
  change(body);
  const digest = await canonicalDigest(
    claimComparisonsRecordSchema,
    readJson(JSON.stringify(body)) as Json,
  );
  return JSON.stringify({ ...body, record_digest: digest });
}
async function reason(source: string): Promise<string> {
  const admission = await admitClaimComparisons(source);
  if (admission.ok) throw new Error("expected a refusal");
  return admission.reason;
}
function first(body: Json): Json {
  return (body["comparisons"] as Json[])[0] as Json;
}

describe("Claim comparison record admission", () => {
  it("admits the record the Python tool sealed, with the same digest over real floats", async () => {
    const admission = await admitCommittedClaimComparisons();
    if (!admission.ok) throw new Error(admission.reason);
    const record = admission.record;
    expect(record.recordDigest).toBe((JSON.parse(recordText) as Json)["record_digest"]);
    expect(record.home).toBe("scpn-quantum-control 1.3.0");
    expect(record.accepted).toEqual([]);
    expect(record.comparisons.map((item) => [item.left, item.status, item.negative])).toEqual([
      ["d10-even-arm-leakage", "non_comparable", false],
      ["d10-asymmetry", "insufficient_evidence", true],
      ["powered-delta-8", "insufficient_evidence", false],
    ]);
    expect(record.sources.get("d10-asymmetry")).toBe(
      "data/iqm_paper_replication/iqm_dla_d10_retest_primary_analysis_2026-07-22.json",
    );
  });

  it("refuses text, schemas, every malformed field and a changed record", async () => {
    expect(await reason("{not json")).toBe(
      "The claim comparison record was refused: it is not JSON.",
    );
    expect(await reason("[]")).toContain("$ must be an object");
    expect(await reason(await resealed((body) => (body["schema"] = "other")))).toContain(
      "unknown schema",
    );
    expect(await reason(await resealed((body) => (body["values"] = {})))).toContain(
      "$.values must be an array",
    );
    expect(
      await reason(
        await resealed((body) => (((body["values"] as Json[])[0] as Json)["source"] = "x")),
      ),
    ).toContain("$.values[0].source must be an object");
    expect(await reason(await resealed((body) => (first(body)["status"] = "proven")))).toContain(
      "$.comparisons[0].status is not one of",
    );
    expect(await reason(await resealed((body) => (first(body)["reasons"] = [""])))).toContain(
      "$.comparisons[0].reasons[0] must be text",
    );
    expect(await reason(await resealed((body) => (first(body)["federated"] = "no")))).toContain(
      "$.comparisons[0].federated must be a boolean",
    );
    expect(await reason(await resealed((body) => (body["home"] = "here")))).toContain(
      "$.home must be an object",
    );
    expect(
      await reason(await resealed((body) => (body["accepted"] = [{ producer: "x" }]))),
    ).toContain("$.accepted[0].version must be text");
    const changed = JSON.parse(recordText) as Json;
    first(changed)["status"] = "agreement";
    expect(await reason(JSON.stringify(changed))).toContain("differs from the recorded one");
  });
});

describe("Claim comparisons panel", () => {
  it("lists every verdict with its sources and reasons", async () => {
    render(<ClaimComparisons />);
    expect(screen.getByRole("status").textContent).toBe("Verifying the claim comparison record…");
    const table = await screen.findByRole("table", { name: "Selected comparisons" });
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    expect(table.textContent).toContain("estimand differs");
    expect(table.textContent).toContain("counter-evidence");
    expect(
      screen.getByText(
        /Accepted producer equivalence: none, so comparisons across producers are disabled/,
      ),
    ).toBeTruthy();
    expect(screen.queryByRole("table", { name: "Counter-evidence kept visible" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Scroll Selected comparisons" }));
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "Selected comparisons table scrolling" }),
    );
  });

  it("never lets the filter hide counter-evidence", async () => {
    render(<ClaimComparisons />);
    await screen.findByRole("table", { name: "Selected comparisons" });
    fireEvent.click(screen.getByLabelText("insufficient_evidence"));
    const selected = screen.getByRole("table", { name: "Selected comparisons" });
    expect(within(selected).getAllByRole("row")).toHaveLength(2);
    const kept = screen.getByRole("table", { name: "Counter-evidence kept visible" });
    expect(
      within(kept)
        .getAllByRole("row")
        .slice(1)
        .map((row) => row.getAttribute("data-status")),
    ).toEqual(["insufficient_evidence"]);
    expect(kept.textContent).toContain("d10-asymmetry");
    fireEvent.click(screen.getByLabelText("insufficient_evidence"));
    expect(screen.queryByRole("table", { name: "Counter-evidence kept visible" })).toBeNull();
  });

  it("keeps a filtered-out disagreement visible and names federated pairs and accepted producers", async () => {
    const source = await resealed((body) => {
      body["accepted"] = [{ producer: "other-studio", version: "3.1" }];
      const item = first(body);
      item["status"] = "disagreement";
      item["federated"] = true;
      item["right"] = "unknown-value";
      item["left"] = "unknown-left";
    });
    render(<ClaimComparisons admit={() => admitClaimComparisons(source)} />);
    await screen.findByRole("table", { name: "Selected comparisons" });
    expect(screen.getByText(/Accepted producer equivalence: other-studio 3.1\./)).toBeTruthy();
    fireEvent.click(screen.getByLabelText("disagreement"));
    const kept = screen.getByRole("table", { name: "Counter-evidence kept visible" });
    expect(kept.textContent).toContain(
      "unknown-left (unknown source) · unknown-value (unknown source) · federated",
    );
  });

  it("reports a refused record", async () => {
    const refused = (): Promise<ClaimComparisonsAdmission> =>
      Promise.resolve({ ok: false, reason: "The claim comparison record was refused: fixture." });
    render(<ClaimComparisons admit={refused} />);
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The claim comparison record was refused: fixture.",
    );
  });
});

describe("Claim comparisons panel lifecycle", () => {
  it("discards an admission that resolves after the panel closed", async () => {
    let release: (admission: ClaimComparisonsAdmission) => void = () => {};
    const pending = (): Promise<ClaimComparisonsAdmission> =>
      new Promise((resolve) => {
        release = resolve;
      });
    const { unmount } = render(<ClaimComparisons admit={pending} />);
    unmount();
    release({ ok: false, reason: "late" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("late")).toBeNull();
  });
});
