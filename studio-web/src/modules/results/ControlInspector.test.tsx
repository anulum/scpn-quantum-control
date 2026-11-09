// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — control proposals are shown, never applied

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import committed from "../../../../data/studio/control_proposal_instrument_20261109.json";
import { ControlInspector } from "./ControlInspector";
import { controlInstrument, parseControlInstrument } from "./controlInstrument";

afterEach(cleanup);

type Data = typeof committed;
const copy = (): Data => structuredClone(committed);
const proposals = (data: Data) => data.codesign.proposals as Record<string, unknown>[];

function at<T>(items: readonly T[], index: number): T {
  const item = items[index];
  if (item === undefined) throw new Error(`missing item ${index}`);
  return item;
}

function refusal(mutate: (data: Data) => void): string {
  const data = copy();
  mutate(data);
  const parsed = parseControlInstrument(data);
  if (parsed.ok) throw new Error("expected a refusal");
  return parsed.reason;
}

function selectProposal(label: string): void {
  const select = screen.getByLabelText("Co-design proposal") as HTMLSelectElement;
  const option = within(select).getByText(label) as HTMLOptionElement;
  fireEvent.change(select, { target: { value: option.value } });
}

describe("control proposals and co-design", () => {
  it("test_control_proposal_instrument_01: signed and rotated targets show the source objective", () => {
    render(<ControlInspector instrument={controlInstrument} />);
    const rows = within(screen.getByRole("table", { name: "Tracking targets" })).getAllByRole(
      "row",
    );
    const signed = rows.find((row) => row.textContent?.startsWith("signed target"));
    expect(signed?.textContent).toBe("signed target[-1]140Proposed, not applied");
    const flipped = rows.find((row) => row.textContent?.startsWith("sign-flipped target"));
    expect(flipped?.textContent).toBe("sign-flipped target[1]101Proposed, not applied");
    const rotated = rows.find((row) => row.textContent?.startsWith("rotated target"));
    expect(rotated?.textContent).toBe("rotated target[-1, 1]240Proposed, not applied");
    expect(
      refusal((data) => {
        at(data.mpc_targets, 0).proposed_action = 1;
      }),
    ).toBe("Proposed action differs from the lowest source cost");
    expect(
      refusal((data) => {
        at(data.mpc_targets, 0).action_costs = [1];
      }),
    ).toBe("A one-step target needs two action costs");
  });

  it("test_control_proposal_instrument_02: the graph highlights binding and violated caps", () => {
    render(<ControlInspector instrument={controlInstrument} />);
    const graph = screen.getByRole("img", { name: "Coupling graph constraints" });
    const state = () =>
      Array.from(graph.querySelectorAll("line")).map((line) => line.getAttribute("data-state"));
    expect(state()).toEqual(["binding", "within cap", "within cap"]);
    expect(screen.getByRole("table", { name: "Constraint values" }).textContent).toContain(
      "held-out disturbance 1",
    );
    selectProposal("desired plan, edge 0-1 over its cap");
    expect(state()).toEqual(["violated", "within cap", "within cap"]);
    expect(screen.getByLabelText("Edge constraint states").textContent).toContain(
      "Edge 0-1: violated",
    );
    expect(
      refusal((data) => {
        at(proposals(data), 0)["binding_edges"] = [7];
      }),
    ).toBe("binding_edges names an unknown edge");
    expect(
      refusal((data) => {
        data.codesign.graph.edges[0] = [0, 9];
      }),
    ).toBe("An edge must join two known nodes");
    expect(
      refusal((data) => {
        data.codesign.graph.edges[0] = [0, 1, 2];
      }),
    ).toBe("An edge must join two known nodes");
  });

  it("test_control_proposal_instrument_03: vetoed or out-of-domain proposals never show as applied", () => {
    render(<ControlInspector instrument={controlInstrument} />);
    for (const label of ["plan outside the tested domain", "hardware-mode request"]) {
      selectProposal(label);
      const outcome = screen.getByLabelText("Proposal outcome").textContent ?? "";
      expect(outcome).toContain("Applied");
      expect(outcome).toContain("No, no runtime actuation evidence");
    }
    expect(screen.getByLabelText("Proposal outcome").textContent).toContain("vetoed");
    expect(screen.getByLabelText("Proposal outcome").textContent).toContain("No plan");
    expect(screen.getByLabelText("Proposal outcome").textContent).toContain("None accepted");
    expect(screen.queryByRole("table", { name: "Constraint values" })).toBeNull();
    expect(
      refusal((data) => {
        at(proposals(data), 3)["applied"] = true;
      }),
    ).toBe("A co-design proposal cannot be shown as applied or actuated");
    expect(
      refusal((data) => {
        at(proposals(data), 0)["hardware_actuation"] = true;
      }),
    ).toBe("A co-design proposal cannot be shown as applied or actuated");
    expect(
      refusal((data) => {
        at(data.mpc_targets, 1).applied = true;
      }),
    ).toBe("A proposed action cannot be shown as applied");
    expect(
      refusal((data) => {
        at(proposals(data), 2)["adopted_plan"] = [0.5, -0.1, 0.2];
      }),
    ).toBe("Only an accepted proposal may change the adopted plan");
    expect(
      refusal((data) => {
        at(proposals(data), 3)["plan"] = [0.1, 0.1, 0.1];
      }),
    ).toBe("A vetoed proposal has no plan");
    expect(
      refusal((data) => {
        at(proposals(data), 1)["status"] = "applied";
      }),
    ).toBe("Unknown proposal status");
  });

  it("test_control_proposal_instrument_04: timestamps are the source integrator's own", () => {
    render(<ControlInspector instrument={controlInstrument} />);
    const trace = screen.getByRole("table", { name: "Simulated closed-loop trace" });
    const times = within(trace)
      .getAllByRole("row")
      .slice(1)
      .map((row) => row.firstElementChild?.textContent);
    expect(times).toEqual(committed.trace.times.map(String));
    expect(times[3]).toBe("0.8999999999999999");
    expect(screen.getByRole("status").textContent).toContain(
      "Observed hardware telemetry: unavailable",
    );
    expect(
      refusal((data) => {
        data.trace.times = [0, 0.3, 0.3, 0.9, 1.2];
      }),
    ).toBe("Trace timestamps must increase");
    expect(
      refusal((data) => {
        data.trace.times = [0, 0.3];
      }),
    ).toBe("Every trace sample needs one timestamp");
    expect(
      refusal((data) => {
        (data.trace as Record<string, unknown>)["observed_hardware_telemetry"] = { rounds: 4 };
      }),
    ).toBe("This instrument shows simulated traces only; observed telemetry is unsupported");
    expect(
      refusal((data) => {
        data.trace.simulated = false;
      }),
    ).toBe("The trace must be declared simulated");
  });

  it("refuses an over-budget latency claim and inconsistent verdicts", () => {
    render(<ControlInspector instrument={controlInstrument} />);
    const latency =
      screen.getByRole("table", { name: "Latency budget verdicts" }).textContent ?? "";
    expect(latency).toContain("within budget (replayed_observed_round_latencies_s)");
    expect(latency).toContain("latency claim refused: max round latency 0.015000000s exceeds");
    expect(
      refusal((data) => {
        at(data.latency, 1).passes = true;
      }),
    ).toBe("A latency verdict and its blockers disagree");
  });

  it("refuses an empty proposal list and misaligned constraint values", () => {
    expect(
      refusal((data) => {
        data.codesign.proposals = [];
      }),
    ).toBe("The co-design section needs at least one proposal");
    expect(
      refusal((data) => {
        (at(proposals(data), 0)["constraint_values"] as number[][])[0] = [0.1];
      }),
    ).toBe("Every constraint case needs one value per constraint row");
  });

  it("refuses an unsupported schema and renders the refusal visibly", () => {
    const reason = refusal((data) => {
      data.schema = "scpn_qc_studio_control_proposal_instrument_v0";
    });
    expect(reason).toBe("Unsupported control instrument schema");
    render(<ControlInspector instrument={{ ok: false, reason }} />);
    expect(screen.getByText(reason, { exact: false })).toBeTruthy();
    expect(screen.queryByRole("table", { name: "Tracking targets" })).toBeNull();
  });
});
