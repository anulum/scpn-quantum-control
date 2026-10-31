// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the pulse instrument marks violations and exports only no-submit dossiers

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import committedInstrumentJson from "../../../../data/studio/pulse_instrument_20261031.json";
import { PulseInspector } from "./PulseInspector";
import { PULSE_INSTRUMENT_SCHEMA, parsePulseInstrument, pulseInstrument } from "./pulseInstrument";
import type { PulseInstrument } from "./pulseInstrument";

afterEach(cleanup);

type Json = Record<string, unknown>;

function instrument(): PulseInstrument {
  if (!pulseInstrument.ok) throw new Error(pulseInstrument.reason);
  return pulseInstrument.value;
}

function copy(): Json {
  return JSON.parse(JSON.stringify(committedInstrumentJson)) as Json;
}

function caseOf(payload: Json, id: string): Json {
  const found = (payload["cases"] as Json[]).find((item) => item["case_id"] === id);
  if (found === undefined) throw new Error(`fixture case ${id} is missing`);
  return found;
}

function choose(id: string) {
  const index = instrument().cases.findIndex((item) => item.caseId === id);
  fireEvent.change(screen.getByLabelText("Design"), { target: { value: String(index) } });
}

describe("pulse feasibility instrument", () => {
  it("test_pulse_analog_instrument_01: half-grid and over-amplitude pulses are marked at their source", () => {
    render(<PulseInspector instrument={pulseInstrument} />);
    choose("half-grid-and-over-amplitude");
    const timeline = screen.getByRole("img", { name: "Design timeline by drive and coupler" });
    const marked = [...timeline.querySelectorAll("rect[data-violated='true']")].map((rect) =>
      rect.getAttribute("data-pulse"),
    );
    expect(marked).toEqual(["0", "1"]);
    const violations = within(screen.getByRole("list", { name: "Violations" }))
      .getAllByRole("listitem")
      .map((item) => item.textContent ?? "");
    expect(violations.some((item) => item.startsWith("off_grid (pulse 0)"))).toBe(true);
    expect(violations.some((item) => item.startsWith("amplitude_over_limit (pulse 1)"))).toBe(true);
    expect(screen.queryByRole("table", { name: "Native pulses" })).toBeNull();
    expect(screen.getByRole("alert", { name: "Dossier refused" }).textContent).toContain(
      "violations",
    );
    choose("disconnected-coupler");
    expect(screen.getByRole("table", { name: "Placement" }).textContent).toContain("q12");
    expect(screen.getByRole("list", { name: "Violations" }).textContent).toContain(
      "disconnected_placement",
    );
  });

  it("test_pulse_analog_instrument_02: an expired capability snapshot invalidates the dossier", () => {
    render(<PulseInspector instrument={pulseInstrument} />);
    choose("current-snapshot");
    expect(screen.getByRole("note", { name: "Capability snapshot" }).textContent).toContain(
      "valid at the assessment",
    );
    expect(screen.getByRole("link", { name: "Download no-submit dossier" })).toBeTruthy();
    choose("expired-snapshot");
    expect(screen.getByRole("note", { name: "Capability snapshot" }).textContent).toContain(
      "expired before the assessment",
    );
    expect(screen.queryByRole("link", { name: "Download no-submit dossier" })).toBeNull();
    expect(screen.getByRole("alert", { name: "Dossier refused" }).textContent).toContain(
      "expired at 2026-10-30T21:00:00Z",
    );
    const expired = copy();
    const rogue = caseOf(expired, "expired-snapshot");
    rogue["dossier"] = caseOf(copy(), "current-snapshot")["dossier"];
    rogue["dossier_refusal"] = null;
    expect(parsePulseInstrument(expired)).toMatchObject({ ok: false });
  });

  it("test_pulse_analog_instrument_03: the dossier is a local no-submit file", () => {
    render(<PulseInspector instrument={pulseInstrument} />);
    const link = screen.getByRole("link", { name: "Download no-submit dossier" });
    const href = link.getAttribute("href") ?? "";
    expect(href.startsWith("data:application/json;charset=utf-8,")).toBe(true);
    const dossier = JSON.parse(decodeURIComponent(href.split(",").slice(1).join(","))) as Json;
    expect(dossier).toMatchObject({
      hardware_submission: false,
      device_authority: false,
      submissions: 0,
    });
    expect(link.getAttribute("download")).toBe("pulse-dossier-feasible-design.json");
    for (const field of ["hardware_submission", "device_authority"] as const) {
      const tampered = copy();
      ((caseOf(tampered, "feasible-design")["dossier"] as Json)[field] as boolean) = true;
      expect(parsePulseInstrument(tampered).ok).toBe(false);
    }
    const counted = copy();
    (caseOf(counted, "feasible-design")["dossier"] as Json)["submissions"] = 1;
    expect(parsePulseInstrument(counted)).toEqual({
      ok: false,
      reason: "cases[0].dossier must record zero submissions",
    });
  });

  it("test_pulse_analog_instrument_04: design units never offer native execution", () => {
    render(<PulseInspector instrument={pulseInstrument} />);
    expect(screen.getByRole("note", { name: "Design units" }).textContent).toContain(
      "not a device-executable",
    );
    for (const id of instrument().cases.map((item) => item.caseId)) {
      choose(id);
      expect(screen.queryByRole("button", { name: /submit|execute|run/i })).toBeNull();
    }
    const native = screen.queryByRole("table", { name: "Native pulses" });
    expect(native?.textContent).toContain("2.000 ns each");
    const authority = copy();
    caseOf(authority, "feasible-design")["device_authority"] = true;
    expect(parsePulseInstrument(authority).ok).toBe(false);
  });

  it("shows declared limits and an undeclared coupler conversion", () => {
    const payload = copy();
    const limits = caseOf(payload, "disconnected-coupler")["limits"] as Json;
    limits["coupler_rate_per_amplitude"] = null;
    limits["max_slew"] = null;
    const edge = (caseOf(payload, "disconnected-coupler")["violations"] as Json[])[0] as Json;
    edge["pulse"] = null;
    edge["channel"] = "u0_2";
    render(<PulseInspector instrument={parsePulseInstrument(payload)} />);
    choose("disconnected-coupler");
    expect(screen.getByRole("list", { name: "Violations" }).textContent).toContain(
      "disconnected_placement (channel u0_2)",
    );
    const facts = screen.getByLabelText("Limits").textContent ?? "";
    expect(facts).toContain("not declared");
    expect(facts).toContain("not checked");
    expect(facts).toContain("the designer");
    choose("current-snapshot");
    expect(screen.getByLabelText("Limits").textContent).toContain("provider capability snapshot");
  });

  it("refuses malformed data loudly", () => {
    const refused = parsePulseInstrument({ schema: "other" });
    expect(refused).toEqual({ ok: false, reason: `schema must be ${PULSE_INSTRUMENT_SCHEMA}` });
    render(<PulseInspector instrument={refused} />);
    expect(screen.getByRole("alert").textContent).toContain("pulse_instrument_20261031.json");
    const first = (p: Json) => caseOf(p, "half-grid-and-over-amplitude");
    const expired = (p: Json) => caseOf(p, "expired-snapshot");
    const feasible = (p: Json) => caseOf(p, "feasible-design");
    const rows: [string, (payload: Json) => void, string][] = [
      ["source", (p) => ((first(p)["limits"] as Json)["source"] = "guess"), "design or provider"],
      ["order", (p) => (((first(p)["design"] as Json[])[1] as Json)["index"] = 0), "index order"],
      [
        "foreign pulse",
        (p) => (((first(p)["violations"] as Json[])[0] as Json)["pulse"] = 9),
        "names a pulse the design does not have",
      ],
      ["feasible", (p) => (first(p)["feasible"] = true), "exactly when there is no violation"],
      [
        "native",
        (p) => (first(p)["native"] = caseOf(copy(), "feasible-design")["native"]),
        "native pulses although it is not feasible",
      ],
      [
        "both",
        (p) => (first(p)["dossier"] = caseOf(copy(), "feasible-design")["dossier"]),
        "either a dossier or the reason",
      ],
      [
        "infeasible dossier",
        (p) => {
          first(p)["dossier"] = caseOf(copy(), "feasible-design")["dossier"];
          first(p)["dossier_refusal"] = null;
        },
        "infeasible design or an expired snapshot",
      ],
      [
        "submission",
        (p) => (first(p)["hardware_submission"] = true),
        "must not submit or claim device authority",
      ],
      ["dt", (p) => ((first(p)["limits"] as Json)["dt_seconds"] = 0), "dt_seconds must lie in"],
      [
        "valid flag flipped",
        (p) => ((expired(p)["snapshot"] as Json)["valid_at_assessment"] = true),
        "valid_at_assessment disagrees with its dates",
      ],
      [
        "flipped with a dossier",
        (p) => {
          (expired(p)["snapshot"] as Json)["valid_at_assessment"] = true;
          expired(p)["dossier"] = caseOf(copy(), "current-snapshot")["dossier"];
          expired(p)["dossier_refusal"] = null;
        },
        "valid_at_assessment disagrees with its dates",
      ],
      [
        "expiry not a UTC second",
        (p) => ((expired(p)["snapshot"] as Json)["expires_at"] = "2026-10-30"),
        "expires_at must be an exact UTC second",
      ],
      [
        "observed after expiry",
        (p) => ((expired(p)["snapshot"] as Json)["observed_at"] = "2026-10-31T00:00:00Z"),
        "must be observed before it expires",
      ],
      ["assessment time", (p) => (p["assessed_at"] = "yesterday"), "assessed_at must be an exact"],
      [
        "sample above the limit",
        (p) =>
          (((feasible(p)["native"] as Json[])[0] as Json)["samples"] as number[]).splice(0, 1, 2),
        "sample above the amplitude limit",
      ],
      [
        "dossier period",
        (p) => ((feasible(p)["dossier"] as Json)["dt"] = 1e-9),
        "must carry the case's own limits",
      ],
      [
        "dossier pulses",
        (p) => ((feasible(p)["dossier"] as Json)["pulses"] = []),
        "pulses must be the case's native pulses",
      ],
      [
        "dossier schema",
        (p) => ((feasible(p)["dossier"] as Json)["schema"] = "other"),
        "schema must be native_pulse_design_v1",
      ],
      [
        "negative amplitude limit",
        (p) => ((first(p)["limits"] as Json)["max_amplitude"] = -1),
        "max_amplitude must lie in",
      ],
      [
        "zero grid",
        (p) => ((first(p)["limits"] as Json)["granularity_samples"] = 0),
        "granularity must be at least 1",
      ],
      [
        "no qubits",
        (p) => ((first(p)["limits"] as Json)["n_qubits"] = 0),
        "n_qubits must be at least 1",
      ],
      [
        "coupling out of range",
        (p) => ((first(p)["limits"] as Json)["coupling_map"] = [[0, 9]]),
        "must be two distinct physical qubits",
      ],
      [
        "self coupling",
        (p) => ((first(p)["limits"] as Json)["coupling_map"] = [[1, 1]]),
        "must be two distinct physical qubits",
      ],
    ];
    for (const [name, edit, reason] of rows) {
      const payload = copy();
      edit(payload);
      const result = parsePulseInstrument(payload);
      expect(result.ok, name).toBe(false);
      if (!result.ok) expect(result.reason, name).toContain(reason);
    }
  });
});
