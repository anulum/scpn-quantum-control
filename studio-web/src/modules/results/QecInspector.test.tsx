// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the QEC instrument shows source values in separate views

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import committedInstrumentJson from "../../../../data/studio/qec_resource_instrument_20261031.json";
import { QecInspector } from "./QecInspector";
import {
  QEC_INSTRUMENT_SCHEMA,
  decoderSampleAt,
  distinct,
  intervalNote,
  parseQecInstrument,
  qecInstrument,
} from "./qecInstrument";
import type { QecInstrument } from "./qecInstrument";

afterEach(cleanup);

type Json = Record<string, unknown>;

function instrument(): QecInstrument {
  if (!qecInstrument.ok) throw new Error(qecInstrument.reason);
  return qecInstrument.value;
}

/** A deep copy of the committed data that a test may edit. */
function copy(): Json {
  return JSON.parse(JSON.stringify(committedInstrumentJson)) as Json;
}

function section(payload: Json, name: string): Json {
  return payload[name] as Json;
}

function rows(payload: Json, name: string, key: string): Json[] {
  return section(payload, name)[key] as Json[];
}

/** The first item of a list, failing the test when it is missing. */
function at(items: readonly Json[]): Json {
  return present(items[0]);
}

/** A value that must exist, failing the test when it does not. */
function present(value: Json | undefined): Json {
  if (value === undefined) throw new Error("fixture row is missing");
  return value;
}

function openTab(name: string) {
  fireEvent.click(screen.getByRole("tab", { name }));
}

describe("QEC and future-resource instrument", () => {
  it("test_qec_resource_instrument_01: switching noise and distance shows the source sample", () => {
    render(<QecInspector instrument={qecInstrument} />);
    const data = instrument();
    const view = screen.getByRole("region", { name: "Decoder simulation" });
    const facts = () => within(view).getByLabelText("Selected decoder sample");
    for (const distance of distinct(data.decoder.experiments, (row) => row.distance)) {
      fireEvent.change(within(view).getByLabelText("Code distance"), {
        target: { value: String(distance) },
      });
      for (const rate of distinct(data.decoder.experiments, (row) => row.physicalErrorRate)) {
        fireEvent.change(within(view).getByLabelText("Physical error rate"), {
          target: { value: String(rate) },
        });
        const sample = decoderSampleAt(data, distance, rate);
        expect(sample).not.toBeNull();
        expect(facts().textContent).toContain(`${sample?.failures} of ${sample?.shots}`);
        expect(facts().textContent).toContain(String(sample?.seed));
      }
    }
    const syndromes = within(view).getByRole("table", { name: "Single-error syndromes" });
    const distance = Number(
      (within(view).getByLabelText("Code distance") as HTMLSelectElement).value,
    );
    expect(within(syndromes).getAllByRole("row")).toHaveLength(
      1 + data.decoder.syndromes.filter((row) => row.distance === distance).length,
    );
  });

  it("test_qec_resource_instrument_02: the hardware logical-error reduction stays unmeasured", () => {
    render(<QecInspector instrument={qecInstrument} />);
    openTab("Offline readiness");
    expect(
      screen.getByRole("status", { name: "Hardware logical-error reduction" }).textContent,
    ).toBe("Hardware logical-error reduction: not measured");
    const blocked = screen.getByRole("list", { name: "Blocked claims" });
    expect(
      within(blocked)
        .getAllByRole("listitem")
        .map((item) => item.textContent),
    ).toEqual(instrument().readiness.blocked);
    expect(screen.queryByText(/reduction factor|×/)).toBeNull();
    const malformed = copy();
    section(malformed, "readiness")["hardware_logical_error_reduction"] = "0.5x";
    expect(parseQecInstrument(malformed)).toEqual({
      ok: false,
      reason: "readiness must state that hardware logical-error reduction is not measured",
    });
    const claiming = copy();
    (rows(claiming, "decoder", "experiments")[0] as Json)[
      "measured_logical_error_reduction_claim_allowed"
    ] = true;
    expect(parseQecInstrument(claiming).ok).toBe(false);
  });

  it("test_qec_resource_instrument_03: excluded factory and routing time are shown", () => {
    render(<QecInspector instrument={qecInstrument} />);
    openTab("Future resources");
    const excluded = screen.getByRole("list", { name: "Excluded overheads" });
    const items = within(excluded)
      .getAllByRole("listitem")
      .map((item) => item.textContent);
    expect(items).toEqual(instrument().resources.excluded);
    expect(items.some((item) => item?.includes("magic-state factory"))).toBe(true);
    expect(items.some((item) => item?.includes("routing"))).toBe(true);
    const scenarios = screen.getByRole("table", { name: "Overhead scenarios" });
    expect(within(scenarios).getAllByRole("row")).toHaveLength(4);
    expect(screen.getByRole("img").getAttribute("aria-label")).toContain("scenario range");
    const select = screen.getByLabelText("Assumed physical error rate");
    const refused = instrument().resources.rows.findIndex((row) => row.status === "refused");
    fireEvent.change(select, { target: { value: String(refused) } });
    expect(screen.getByRole("alert", { name: "Resource estimate refused" }).textContent).toContain(
      "threshold",
    );
    fireEvent.change(select, { target: { value: "1" } });
    const estimate = screen.getByLabelText("Resource estimate");
    const row = instrument().resources.rows[1];
    expect(row?.status).toBe("estimated");
    if (row?.status === "estimated") expect(estimate.textContent).toContain(row.payloadSha256);
  });

  it("test_qec_resource_instrument_04: the interval method states the actual trial count", () => {
    render(<QecInspector instrument={qecInstrument} />);
    const data = instrument();
    const zero = data.decoder.experiments.find((row) => row.failures === 0);
    expect(zero).toBeDefined();
    if (zero === undefined) return;
    const view = screen.getByRole("region", { name: "Decoder simulation" });
    fireEvent.change(within(view).getByLabelText("Code distance"), {
      target: { value: String(zero.distance) },
    });
    fireEvent.change(within(view).getByLabelText("Physical error rate"), {
      target: { value: String(zero.physicalErrorRate) },
    });
    const facts = within(view).getByLabelText("Selected decoder sample").textContent ?? "";
    expect(facts).toContain(`exact Clopper-Pearson interval from 0 of ${zero.shots} trials`);
    expect(facts).toContain("not zero");
    expect(zero.upper).toBeCloseTo(1 - 0.025 ** (1 / zero.shots), 12);
    expect(intervalNote({ failures: 3, trials: 10, upper: 0.6, method: "wilson" })).toBe(
      "wilson interval from 3 of 10 trials.",
    );
  });

  it("refuses malformed data loudly instead of drawing a partial instrument", () => {
    const refused = parseQecInstrument({ schema: "other" });
    expect(refused).toEqual({ ok: false, reason: `schema must be ${QEC_INSTRUMENT_SCHEMA}` });
    render(<QecInspector instrument={refused} />);
    expect(screen.getByRole("alert").textContent).toContain(
      "qec_resource_instrument_20261031.json",
    );
  });

  it("reports a grid point that was not run", () => {
    const payload = copy();
    const experiments = rows(payload, "decoder", "experiments");
    section(payload, "decoder")["experiments"] = experiments.filter(
      (row) => !(row["distance"] === 5 && row["physical_error_rate"] === 0.1),
    );
    render(<QecInspector instrument={parseQecInstrument(payload)} />);
    const view = screen.getByRole("region", { name: "Decoder simulation" });
    fireEvent.change(within(view).getByLabelText("Code distance"), { target: { value: "5" } });
    fireEvent.change(within(view).getByLabelText("Physical error rate"), {
      target: { value: "0.1" },
    });
    expect(within(view).getByRole("status").textContent).toContain("No sample was run");
  });

  it("shows refused bundles, uncorrected syndromes and a one-point plot as they are", () => {
    const payload = copy();
    section(payload, "readiness")["admitted"] = false;
    at(rows(payload, "decoder", "single_error_syndromes"))["corrected"] = false;
    const resources = section(payload, "resources");
    resources["rows"] = (resources["rows"] as Json[]).slice(0, 1);
    render(<QecInspector instrument={parseQecInstrument(payload)} />);
    const syndromes = screen.getByRole("table", { name: "Single-error syndromes" });
    expect(within(syndromes).getAllByRole("cell", { name: "no" }).length).toBe(1);
    openTab("Offline readiness");
    expect(screen.getByText("refused")).toBeTruthy();
    openTab("Future resources");
    expect(screen.getByRole("img").querySelectorAll("circle")).toHaveLength(1);
  });

  it("refuses every malformed field with its own reason", () => {
    const cases: [string, (payload: Json) => void, string][] = [
      ["not an object", () => undefined, "artefact must be an object"],
      ["artifact id", (p) => (p["artifact_id"] = ""), "artifact_id must be a non-empty string"],
      [
        "shots",
        (p) => (section(p, "decoder")["shots"] = 1.5),
        "decoder.shots must be a non-negative integer",
      ],
      [
        "experiments",
        (p) => (section(p, "decoder")["experiments"] = {}),
        "decoder.experiments must be a list",
      ],
      [
        "rate",
        (p) => (at(rows(p, "decoder", "experiments"))["rate"] = "x"),
        "rate must be a finite number",
      ],
      [
        "counts",
        (p) => (at(rows(p, "decoder", "experiments"))["failures"] = 5000),
        "counts are inconsistent",
      ],
      [
        "order",
        (p) => (at(rows(p, "decoder", "experiments"))["lower"] = 0.9),
        "interval must satisfy",
      ],
      [
        "zero",
        (p) => {
          const row = present(
            rows(p, "decoder", "experiments").find((item) => item["failures"] === 0),
          );
          row["upper"] = 0;
        },
        "with zero failures needs lower 0",
      ],
      [
        "syndrome flag",
        (p) => (at(rows(p, "decoder", "single_error_syndromes"))["corrected"] = 1),
        "must be true or false",
      ],
      ["readiness object", (p) => (p["readiness"] = []), "readiness must be an object"],
      [
        "supported",
        (p) => (section(p, "readiness")["supported"] = [""]),
        "readiness.supported[0] must be a non-empty string",
      ],
      [
        "status",
        (p) => (at(rows(p, "resources", "rows"))["status"] = "guessed"),
        "status must be estimated or refused",
      ],
      [
        "request",
        (p) => (section(section(p, "resources"), "request")["trotter_steps"] = null),
        "request.trotter_steps must be a finite number",
      ],
      [
        "rate with no failures",
        (p) => {
          const row = present(
            rows(p, "decoder", "experiments").find((item) => item["failures"] === 0),
          );
          row["rate"] = 0.5;
        },
        "rate is 0.5 but its other fields give 0",
      ],
      [
        "rate outside interval",
        (p) => {
          const row = at(rows(p, "readiness", "aggregates"));
          row["upper"] = row["rate"];
          row["lower"] = row["rate"];
          row["rate"] = (row["rate"] as number) + 1e-3;
        },
        "rate is",
      ],
      [
        "rate below lower",
        (p) => {
          const row = at(rows(p, "readiness", "aggregates"));
          row["lower"] = row["upper"];
        },
        "rate must lie inside its interval",
      ],
      [
        "confidence level",
        (p) => (at(rows(p, "decoder", "experiments"))["confidence_level"] = 7),
        "confidence_level must lie strictly between 0 and 1",
      ],
      [
        "trials named shots",
        (p) => {
          const row = at(rows(p, "readiness", "aggregates"));
          row["shots"] = row["trials"];
          delete row["trials"];
        },
        "readiness.aggregates[0].trials must be a finite number",
      ],
      [
        "union bound",
        (p) => (at(rows(p, "resources", "rows"))["logical_failure_union_bound"] = -0.1),
        "union_bound must lie in [0, 1]",
      ],
      [
        "no excluded overheads",
        (p) => (section(p, "resources")["excluded_from_base_estimate"] = []),
        "must list the overheads the base estimate excludes",
      ],
      [
        "scenario",
        (p) => {
          const row = at(rows(p, "resources", "rows"));
          at(row["scenarios"] as Json[])["label"] = "";
        },
        "scenarios[0].label must be a non-empty string",
      ],
    ];
    for (const [name, edit, reason] of cases) {
      const payload = name === "not an object" ? (null as unknown as Json) : copy();
      if (payload !== null) edit(payload);
      const result = parseQecInstrument(payload);
      expect(result.ok, name).toBe(false);
      if (!result.ok) expect(result.reason, name).toContain(reason);
    }
  });
});
