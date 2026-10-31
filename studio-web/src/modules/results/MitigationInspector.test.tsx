// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the mitigation instrument keeps raw values beside every correction

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import committedInstrumentJson from "../../../../data/studio/mitigation_instrument_20261031.json";
import { MitigationInspector } from "./MitigationInspector";
import {
  MITIGATION_INSTRUMENT_SCHEMA,
  describeMap,
  mitigationInstrument,
  parseMitigationInstrument,
  readoutCaseCsv,
} from "./mitigationInstrument";
import type { MitigationInstrument, ReadoutCase } from "./mitigationInstrument";

afterEach(cleanup);

type Json = Record<string, unknown>;

function instrument(): MitigationInstrument {
  if (!mitigationInstrument.ok) throw new Error(mitigationInstrument.reason);
  return mitigationInstrument.value;
}

/** A deep copy of the committed data that a test may edit. */
function copy(): Json {
  return JSON.parse(JSON.stringify(committedInstrumentJson)) as Json;
}

function section(payload: Json, name: string): Json {
  return payload[name] as Json;
}

function cases(payload: Json): Json[] {
  return section(payload, "readout")["cases"] as Json[];
}

/** The readout case with an identifier, failing the test when it is missing. */
function caseOf(payload: Json, id: string): Json {
  const found = cases(payload).find((item) => item["case_id"] === id);
  if (found === undefined) throw new Error(`fixture case ${id} is missing`);
  return found;
}

function parsedCase(id: string): ReadoutCase {
  const found = instrument().readout.cases.find((item) => item.caseId === id);
  if (found === undefined) throw new Error(`case ${id} is missing`);
  return found;
}

function choose(id: string) {
  const index = instrument().readout.cases.findIndex((item) => item.caseId === id);
  fireEvent.change(screen.getByLabelText("Readout case"), { target: { value: String(index) } });
}

function openTab(name: string) {
  fireEvent.click(screen.getByRole("tab", { name }));
}

function csvOf(link: HTMLElement): string {
  const href = link.getAttribute("href") ?? "";
  const prefix = "data:text/csv;charset=utf-8,";
  expect(href.startsWith(prefix)).toBe(true);
  return decodeURIComponent(href.slice(prefix.length));
}

describe("mitigation and readout instrument", () => {
  it("test_mitigation_inspector_01: a permuted, partial map survives plots and CSV", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    choose("permuted-partial-map");
    const view = screen.getByRole("region", { name: "Readout correction" });
    expect(within(view).getByLabelText("Measurement map").textContent).toBe("c0←q2, c1←q0");
    expect(within(view).getByText("Not measured: q1")).toBeTruthy();
    const item = parsedCase("permuted-partial-map");
    const plot = within(view).getByRole("img", { name: "Raw and corrected values by label" });
    expect(
      [...plot.querySelectorAll("g[data-label]")].map((g) => g.getAttribute("data-label")),
    ).toEqual(item.labels);
    const link = within(view).getByRole("link", { name: "Download CSV" });
    expect(link.getAttribute("download")).toBe("readout-permuted-partial-map.csv");
    const csv = csvOf(link).split("\n");
    expect(csv[1]).toBe('# measurement map,"c0←q2, c1←q0"');
    expect(csv.slice(4, 8).map((line) => line.split(",")[0])).toEqual(item.labels);
    expect(csv[4]?.split(",")[1]).toBe(String(item.rawCounts[0]));
  });

  it("test_mitigation_inspector_02: identity calibration leaves values", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    choose("identity-calibration");
    const item = parsedCase("identity-calibration");
    expect(item.status).toBe("corrected");
    if (item.status !== "corrected") return;
    expect(item.corrected.quasiprobabilities).toEqual(item.rawFrequencies);
    const table = screen.getByRole("table", { name: "Raw and corrected values" });
    for (const row of within(table).getAllByRole("row").slice(1)) {
      const cells = within(row).getAllByRole("cell");
      expect(cells[1]?.textContent).toBe(cells[2]?.textContent);
    }
    expect(screen.getByRole("status", { name: "Signed values" }).textContent).not.toContain(
      "Negative",
    );
    expect(screen.queryByText(/Not measured/)).toBeNull();
  });

  it("test_mitigation_inspector_03: negative quasi-probabilities are visibly signed", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    choose("negative-quasi-probabilities");
    const status = screen.getByRole("status", { name: "Signed values" }).textContent ?? "";
    expect(status).toContain("Negative at 01, 10: these are not probabilities");
    expect(status).toContain("not clipped or renormalised");
    const table = screen.getByRole("table", { name: "Raw and corrected values" });
    expect(within(table).getAllByText(/\(negative\)/)).toHaveLength(2);
    expect(within(table).getAllByText(/^-0\.1094/)).toHaveLength(2);
    const plot = screen.getByRole("img", { name: "Raw and corrected values by label" });
    const below = [...plot.querySelectorAll("title")].filter((title) =>
      /corrected -/.test(title.textContent ?? ""),
    );
    expect(below).toHaveLength(2);
    const csv = csvOf(screen.getByRole("link", { name: "Download CSV" }));
    expect(csv).toContain("01,20,0.02,-0.109375");
  });

  it("test_mitigation_inspector_04: an ill-conditioned correction keeps its diagnostic", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    choose("ill-conditioned");
    const accounting = screen.getByLabelText("Shot accounting").textContent ?? "";
    expect(accounting).toContain("Condition number100.0");
    expect(accounting).toContain("40000 over 4 prepared states");
    expect(screen.getByRole("status", { name: "Signed values" }).textContent).toContain(
      "Negative at 01, 10",
    );
    choose("singular-calibration");
    expect(screen.getByRole("alert", { name: "Correction refused" }).textContent).toContain(
      "calibration condition exceeds the admitted ceiling",
    );
    const table = screen.getByRole("table", { name: "Raw and corrected values" });
    expect(within(table).getAllByText("refused")).toHaveLength(4);
    expect(within(table).getByRole("cell", { name: "600" })).toBeTruthy();
    const plot = screen.getByRole("img", { name: "Raw and corrected values by label" });
    expect(plot.querySelectorAll("rect")).toHaveLength(4);
    const csv = csvOf(screen.getByRole("link", { name: "Download CSV" })).split("\n");
    expect(csv[2]).toBe("# status,refused: calibration condition exceeds the admitted ceiling");
    expect(csv[4]).toBe("00,600,0.6,,,,");
  });

  it("shows the extrapolation with raw values beside corrected ones", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    openTab("Zero-noise extrapolation");
    const data = instrument().zne;
    const table = screen.getByRole("table", { name: "Observable at each noise scale" });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + data.points.length);
    expect(within(table).getByRole("cell", { name: "0.7000" })).toBeTruthy();
    const fit = screen.getByLabelText("Extrapolation").textContent ?? "";
    expect(fit).toContain("gls-delta, polynomial order 1");
    expect(fit).toContain("95 % interval");
    const plot = screen.getByRole("img", {
      name: "Raw and corrected observable against noise scale",
    });
    expect(plot.querySelectorAll("circle")).toHaveLength(1 + 2 * data.points.length);
    expect(screen.getByLabelText("Measurement map").textContent).toBe(
      describeMap(data.measurementMap),
    );
  });

  it("shows the measured hardware pairs with their caveat", () => {
    render(<MitigationInspector instrument={mitigationInstrument} />);
    openTab("Hardware pairs");
    const data = instrument().hardware;
    expect(screen.getByRole("note", { name: "Correction caveat" }).textContent).toBe(
      `${data.method}: ${data.caveat}`,
    );
    const table = screen.getByRole("table", { name: "Measured parity pairs" });
    expect(within(table).getAllByRole("row")).toHaveLength(1 + data.pairs.length);
    expect(screen.getByText(data.bundleDigest)).toBeTruthy();
    openTab("Readout correction");
    expect(screen.getByLabelText("Readout case")).toBeTruthy();
  });

  it("draws a flat extrapolation without dividing by zero", () => {
    const payload = copy();
    const zne = section(payload, "zne");
    for (const point of zne["points"] as Json[]) {
      point["raw_expectation"] = 0.5;
      point["corrected_expectation"] = 0.5;
      point["corrected_standard_error"] = 0;
    }
    Object.assign(section(zne, "extrapolation"), {
      zero_noise_estimate: 0.5,
      low: 0.5,
      high: 0.5,
    });
    render(<MitigationInspector instrument={parseMitigationInstrument(payload)} />);
    openTab("Zero-noise extrapolation");
    const plot = screen.getByRole("img", {
      name: "Raw and corrected observable against noise scale",
    });
    expect(
      [...plot.querySelectorAll("circle")].every((circle) => circle.getAttribute("cy") === "180"),
    ).toBe(true);
  });

  it("exports CSV fields that need quoting", () => {
    const item = parsedCase("singular-calibration");
    const quoted = readoutCaseCsv({ ...item, caseId: 'a "b"', status: "refused", reason: "x" });
    expect(quoted.split("\n")[0]).toBe('# case,"a ""b"""');
    expect(quoted.endsWith("\n")).toBe(true);
  });

  it("refuses malformed data loudly instead of drawing a partial instrument", () => {
    const refused = parseMitigationInstrument({ schema: "other" });
    expect(refused).toEqual({
      ok: false,
      reason: `schema must be ${MITIGATION_INSTRUMENT_SCHEMA}`,
    });
    render(<MitigationInspector instrument={refused} />);
    expect(screen.getByRole("alert").textContent).toContain("mitigation_instrument_20261031.json");
  });

  it("refuses every malformed field with its own reason", () => {
    const first = (p: Json) => caseOf(p, "permuted-partial-map");
    const corrected = (p: Json) => first(p)["corrected"] as Json;
    const rows: [string, (payload: Json) => void, string][] = [
      ["not an object", () => undefined, "artefact must be an object"],
      ["label order", (p) => (p["label_order"] = ""), "label_order must be a non-empty string"],
      ["cases", (p) => (section(p, "readout")["cases"] = {}), "readout.cases must be a list"],
      [
        "map order",
        (p) => {
          (first(p)["measurement_map"] as Json[]).reverse();
        },
        "must list classical bits in order",
      ],
      [
        "map qubits",
        (p) =>
          (first(p)["measurement_map"] = [
            { classical_bit: 0, qubit: 1 },
            { classical_bit: 1, qubit: 1 },
          ]),
        "must read distinct qubits",
      ],
      ["empty map", (p) => (first(p)["measurement_map"] = []), "must read distinct qubits"],
      [
        "missing label",
        (p) => (first(p)["labels"] = ["00", "01", "10", "10"]),
        "must list every one of the 4 labels once",
      ],
      [
        "label width",
        (p) => (first(p)["labels"] = ["000", "01", "10", "11"]),
        "must be binary strings of width 2",
      ],
      [
        "label alphabet",
        (p) => (first(p)["labels"] = ["0x", "01", "10", "11"]),
        "must be binary strings of width 2",
      ],
      ["count total", (p) => (first(p)["shots"] = 1), "raw_counts must add up to the shots"],
      ["count length", (p) => (first(p)["raw_counts"] = [1, 2]), "raw_counts must have 4 entries"],
      [
        "fractional count",
        (p) => ((first(p)["raw_counts"] as number[])[0] = 0.5),
        "raw_counts[0] must be a non-negative integer",
      ],
      [
        "matrix",
        (p) => ((first(p)["calibration"] as Json)["matrix"] = [[1, 0, 0, 0]]),
        "calibration.matrix must be 4 square",
      ],
      ["status", (p) => (first(p)["status"] = "clipped"), "status must be corrected or refused"],
      [
        "hidden negative",
        (p) => (corrected(p)["negative_labels"] = []),
        "negative_labels must name exactly the negative values",
      ],
      [
        "reason",
        (p) => (caseOf(p, "singular-calibration")["reason"] = ""),
        "reason must be a non-empty string",
      ],
      [
        "interval",
        (p) => (section(section(p, "zne"), "extrapolation")["low"] = 5),
        "interval must contain its estimate",
      ],
      [
        "interval above",
        (p) => (section(section(p, "zne"), "extrapolation")["high"] = -5),
        "interval must contain its estimate",
      ],
      [
        "full matrix claim",
        (p) => (section(p, "hardware")["full_confusion_matrix_available"] = true),
        "no full confusion matrix is available",
      ],
      [
        "not admitted",
        (p) => (section(p, "hardware")["admitted"] = false),
        "hardware bundle must be admitted",
      ],
      [
        "pair depth",
        (p) => (((section(p, "hardware")["pairs"] as Json[])[0] as Json)["depth"] = -1),
        "depth must be a non-negative integer",
      ],
      [
        "flip",
        (p) => (((first(p)["readout"] as Json[])[0] as Json)["zero_to_one"] = null),
        "zero_to_one must be a finite number",
      ],
      ["unmeasured", (p) => (first(p)["unmeasured_qubits"] = [true]), "must be a finite number"],
    ];
    for (const [name, edit, reason] of rows) {
      const payload = name === "not an object" ? (null as unknown as Json) : copy();
      if (payload !== null) edit(payload);
      const result = parseMitigationInstrument(payload);
      expect(result.ok, name).toBe(false);
      if (!result.ok) expect(result.reason, name).toContain(reason);
    }
  });
});
