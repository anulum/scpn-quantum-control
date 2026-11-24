// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — calibration inspection admission and inspector tests

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import recordText from "../../../../../data/studio/calibration_inspection_20261123.json?raw";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson } from "../../../shared/contracts/jsonTransport";
import { CalibrationInspector } from "./CalibrationInspector";
import {
  type CalibrationInspectionAdmission,
  admitCalibrationInspection,
  calibrationInspectionRecordSchema,
  referenceStatus,
} from "./calibrationInspection";

afterEach(cleanup);

type Json = Record<string, unknown>;

async function resealed(change: (body: Json) => void): Promise<string> {
  const body = JSON.parse(recordText) as Json;
  delete body["record_digest"];
  change(body);
  const digest = await canonicalDigest(
    calibrationInspectionRecordSchema,
    readJson(JSON.stringify(body)) as Json,
  );
  return JSON.stringify({ ...body, record_digest: digest });
}
async function reason(source: string): Promise<string> {
  const admission = await admitCalibrationInspection(source);
  if (admission.ok) throw new Error("expected a refusal");
  return admission.reason;
}
function list(body: Json, key: string): Json[] {
  return body[key] as Json[];
}
async function open(source?: string): Promise<void> {
  render(
    <CalibrationInspector
      {...(source === undefined ? {} : { admit: () => admitCalibrationInspection(source) })}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Open calibration inspection" }));
  await screen.findByRole("table", { name: "Plans and their calibration" }, { timeout: 15_000 });
}

describe("Calibration inspection admission", () => {
  it("admits the record the Python tool sealed, with the same digest over real readings", async () => {
    const admission = await admitCalibrationInspection(recordText);
    if (!admission.ok) throw new Error(admission.reason);
    const record = admission.record;
    expect(record.recordDigest).toBe((JSON.parse(recordText) as Json)["record_digest"]);
    expect(record.snapshots).toHaveLength(12);
    expect(record.plans.map((plan) => plan.status)).toEqual(Array(12).fill("admitted"));
    expect(record.drifts.every((drift) => drift.calibrationSetChanged)).toBe(true);
    expect(record.policies[0]?.rules).toEqual([
      "snapshot_date_equals_block_date",
      "distinct_calibration_set",
      "required_couplers",
      "request_names_calibration_set",
    ]);
    const first = record.snapshots[0];
    expect(first?.qubits).toHaveLength(20);
    expect(first?.couplers).toHaveLength(30);
    expect(referenceStatus(record, first?.sha256 ?? "").kind).toBe("superseded");
    expect(referenceStatus(record, record.snapshots[11]?.sha256 ?? "").kind).toBe("current");
    expect(referenceStatus(record, "0".repeat(64))).toEqual({ kind: "unknown" });
  });

  it("refuses text, schemas, every malformed field and a changed record", async () => {
    expect(await reason("{not json")).toBe(
      "The calibration inspection record was refused: it is not JSON.",
    );
    expect(await reason("[]")).toContain("$ must be an object");
    const cases: [(body: Json) => void, string][] = [
      [(body) => (body["schema"] = "other"), "unknown schema"],
      [(body) => (body["plans"] = {}), "$.plans must be an array"],
      [
        (body) => ((list(body, "plans")[0] as Json)["status"] = "maybe"),
        "must be admitted or blocked",
      ],
      [(body) => ((list(body, "plans")[0] as Json)["reasons"] = [""]), "reasons[0] must be text"],
      [
        (body) => ((list(body, "snapshots")[0] as Json)["superseded_by"] = 3),
        "superseded_by must be text",
      ],
      [
        (body) =>
          (((list(body, "snapshots")[0] as Json)["qubits"] as Json[])[0] = {
            qubit: 0,
            readout_error: "x",
          }),
        "readout_error must be a number or null",
      ],
      [
        (body) => ((list(body, "drifts")[0] as Json)["calibration_set_changed"] = "yes"),
        "must be a boolean",
      ],
      [
        (body) => ((list(body, "drifts")[0] as Json)["unchanged"] = "0"),
        "unchanged must be a non-negative integer",
      ],
      [
        (body) => ((list(body, "policies")[0] as Json)["citation"] = "x"),
        "citation must be an object",
      ],
    ];
    for (const [change, expected] of cases)
      expect(await reason(await resealed(change))).toContain(expected);
    const changed = JSON.parse(recordText) as Json;
    (list(changed, "plans")[0] as Json)["status"] = "blocked";
    expect(await reason(JSON.stringify(changed))).toContain("differs from the recorded one");
  });
});

describe("Calibration inspector", () => {
  it("test_calibration_snapshot_diff_01: a missing reading shows as unknown, never as a perfect value", async () => {
    const source = await resealed((body) => {
      const last = list(body, "snapshots")[11] as Json;
      (last["qubits"] as Json[])[3] = { qubit: 3, readout_error: null };
      (last["couplers"] as Json[])[0] = { coupler: [0, 1], cz_fidelity: null };
    });
    await open(source);
    const readings = screen.getByRole("table", { name: "Element readings" });
    expect(within(readings).getByRole("rowheader", { name: "q3" }).parentElement?.textContent).toBe(
      "q3readout errorunknown",
    );
    expect(
      within(readings).getByRole("rowheader", { name: "c0-1" }).parentElement?.textContent,
    ).toBe("c0-1CZ fidelityunknown");
    fireEvent.click(screen.getByRole("button", { name: "Scroll Element readings" }));
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "Element readings table scrolling" }),
    );
  });

  it("test_calibration_snapshot_diff_02: a superseded reference blocks its dossiers", async () => {
    await open();
    const record = JSON.parse(recordText) as Json;
    const first = list(record, "snapshots")[0] as Json;
    fireEvent.change(screen.getByLabelText("Calibration file sha256"), {
      target: { value: String(first["sha256"]) },
    });
    expect(screen.getByLabelText("Calibration reference verdict").textContent).toBe(
      `Superseded by ${String(first["superseded_by"])}: dossiers bound to ${String(first["calibration_set_id"])} are blocked, and no other snapshot is substituted.`,
    );
    fireEvent.change(screen.getByLabelText("Calibration file sha256"), {
      target: { value: "f".repeat(64) },
    });
    expect(screen.getByLabelText("Calibration reference verdict").textContent).toBe(
      "Not an admitted snapshot: dossiers bound to it are blocked.",
    );
    const last = list(record, "snapshots")[11] as Json;
    fireEvent.change(screen.getByLabelText("Calibration file sha256"), {
      target: { value: ` ${String(last["sha256"])} ` },
    });
    expect(screen.getByLabelText("Calibration reference verdict").textContent).toBe(
      "Current snapshot of iqm-garnet, 2026-10-06.",
    );
    fireEvent.change(screen.getByLabelText("Calibration file sha256"), { target: { value: "" } });
    expect(screen.getByLabelText("Calibration reference verdict").textContent).toContain(
      "Enter the calibration",
    );
  });

  it("test_calibration_snapshot_diff_03: the source's age verdict at the boundary is shown as written", async () => {
    const source = await resealed((body) => {
      const plan = list(body, "plans")[0] as Json;
      plan["status"] = "blocked";
      plan["reasons"] = ["max_age_days: 3 days old, bound 2"];
    });
    await open(source);
    const plans = screen.getByRole("table", { name: "Plans and their calibration" });
    const row = within(plans).getAllByRole("row")[1];
    expect(row?.getAttribute("data-status")).toBe("blocked");
    expect(row?.textContent).toContain("max_age_days: 3 days old, bound 2");
    fireEvent.click(screen.getByRole("button", { name: "Scroll Plans and their calibration" }));
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "Plans and their calibration table scrolling" }),
    );
  });

  it("test_calibration_snapshot_diff_04: every historical snapshot stays viewable with its drift", async () => {
    await open();
    expect(screen.getByLabelText("Snapshot standing").textContent).toBe(
      "Current calibration of this target.",
    );
    fireEvent.change(screen.getByLabelText("Calibration snapshot"), { target: { value: "0" } });
    expect(screen.getByLabelText("Snapshot standing").textContent).toMatch(
      /^Superseded by 7c39177a/,
    );
    expect(screen.getByText("No earlier snapshot of this target is recorded.")).toBeTruthy();
    expect(
      screen.getByText(
        /Referenced by: iqm_dla_depth_profile_powered_epoch_01_submission_2026-09-23\./,
      ),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Calibration snapshot"), { target: { value: "1" } });
    const drift = screen.getByRole("table", { name: "Changes from the previous snapshot" });
    expect(within(drift).getAllByRole("row")).toHaveLength(51);
    expect(
      screen.getByText(/50 element\(s\) differ, 0 read the same; the calibration set changed\./),
    ).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: "Scroll Changes from the previous snapshot" }),
    );
    expect(document.activeElement).toBe(
      screen.getByRole("region", { name: "Changes from the previous snapshot table scrolling" }),
    );
  });

  it("test_calibration_snapshot_diff_05: a blocked plan keeps its named set and offers no replacement", async () => {
    const source = await resealed((body) => {
      const plan = list(body, "plans")[1] as Json;
      plan["status"] = "blocked";
      plan["calibration_set_id"] = null;
      plan["reasons"] = ["the plan names no calibration set"];
      const drift = list(body, "drifts")[0] as Json;
      drift["calibration_set_changed"] = false;
      (drift["changes"] as Json[])[0] = {
        element: "q0",
        quantity: "readout_error",
        before: null,
        after: 0.1,
        change: "became_known",
      };
      (list(body, "snapshots")[11] as Json)["referenced_by"] = [];
    });
    await open(source);
    const row = within(
      screen.getByRole("table", { name: "Plans and their calibration" }),
    ).getAllByRole("row")[2];
    expect(row?.textContent).toContain("none named");
    expect(screen.queryByRole("button", { name: /replace|substitute|submit|compile/i })).toBeNull();
    expect(screen.getByText("Referenced by: no plan.")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Calibration snapshot"), { target: { value: "1" } });
    expect(screen.getByText(/50 element\(s\) differ, 0 read the same\./)).toBeTruthy();
    expect(screen.getAllByText("unknown").length).toBeGreaterThan(0);
  });

  it("reports a refused record and an empty one", async () => {
    const refused = (): Promise<CalibrationInspectionAdmission> =>
      Promise.resolve({
        ok: false,
        reason: "The calibration inspection record was refused: fixture.",
      });
    render(<CalibrationInspector admit={refused} />);
    fireEvent.click(screen.getByRole("button", { name: "Open calibration inspection" }));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The calibration inspection record was refused: fixture.",
    );
    cleanup();
    await open(
      await resealed((body) => {
        body["snapshots"] = [];
        body["drifts"] = [];
      }),
    );
    expect(screen.getByText("No snapshot is recorded.")).toBeTruthy();
  });
});
