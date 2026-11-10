// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — control proposal and co-design data projection

import committed from "../../../../data/studio/control_proposal_instrument_20261109.json";
import type { Loaded } from "../../shared/browser/data";
import { count, finite, flag, list, numbers, record, text } from "../../shared/browser/jsonFields";

/** One tracking target and the controller's proposed binary action. */
export interface TrackingTarget {
  /** Source case label. */ readonly id: string;
  /** Target state; its sign and direction are part of the objective. */ readonly target: readonly number[];
  /** Source costs of action 0 and action 1, from the controller's own Hamiltonian. */ readonly costs: readonly [
    number,
    number,
  ];
  /** Lowest-cost action as proposed by the source; never applied here. */ readonly proposed: 0 | 1;
}
/** Every outcome a co-design proposal can have. */
export type ProposalStatus =
  | "accepted"
  | "infeasible"
  | "nonconvergent"
  | "out_of_domain"
  | "over_budget"
  | "vetoed";
/** One proposal as the source session verified it. */
export interface CodesignProposal {
  /** Source label of the attempt. */ readonly label: string;
  /** Session outcome; only `accepted` changes the adopted plan. */ readonly status: ProposalStatus;
  /** Source reason text. */ readonly reason: string;
  /** Proposed coupling weights, or null when no plan was produced. */ readonly plan:
    | readonly number[]
    | null;
  /** Recomputed objective, or null. */ readonly objective: number | null;
  /** Constraint values, nominal case first, then each held-out disturbance. */ readonly constraints: readonly (readonly number[])[];
  /** Plan in force after this proposal, or null before any acceptance. */ readonly adopted:
    | readonly number[]
    | null;
  /** Edges at their cap in some checked case. */ readonly binding: readonly number[];
  /** Edges over their cap in some checked case. */ readonly violated: readonly number[];
}
/** A replayed latency series judged against the source budget. */
export interface LatencyVerdict {
  /** Source case label. */ readonly id: string;
  /** Whether every budget ceiling holds. */ readonly passes: boolean;
  /** Source reasons a latency claim is refused. */ readonly blockers: readonly string[];
  /** Declared replay samples in seconds, not measurements of this host. */ readonly samples: readonly number[];
  /** Provenance of the samples. */ readonly clock: string;
}
/** A simulated closed-loop trace exactly as the source integrator produced it. */
export interface ControlTrace {
  /** Integrator sample times, strictly increasing and never regenerated here. */ readonly times: readonly number[];
  /** Global order parameter at each sample time. */ readonly order: readonly number[];
  /** Source adapter that produced the trace. */ readonly source: string;
  /** Source claim boundary of the simulation. */ readonly boundary: string;
}
/** Source-owned control data parsed at the browser boundary. */
export interface ControlInstrument {
  /** Source artefact identity. */ readonly id: string;
  /** Source claim boundary. */ readonly boundary: string;
  /** Why no proposal is shown as applied. */ readonly appliedRule: string;
  /** Tracking targets in source order. */ readonly targets: readonly TrackingTarget[];
  /** Graph edges as node pairs, in plan order. */ readonly edges: readonly (readonly [
    number,
    number,
  ])[];
  /** Constraint row names, edges first. */ readonly constraintRows: readonly string[];
  /** Co-design proposals in source order. */ readonly proposals: readonly CodesignProposal[];
  /** Latency verdicts in source order. */ readonly latency: readonly LatencyVerdict[];
  /** Simulated trace with the integrator's own timestamps. */ readonly trace: ControlTrace;
}

const statuses: readonly ProposalStatus[] = [
  "accepted",
  "infeasible",
  "nonconvergent",
  "out_of_domain",
  "over_budget",
  "vetoed",
];

function nullableNumbers(value: unknown, field: string): number[] | null {
  return value === null ? null : numbers(value, field);
}

function same(left: readonly number[] | null, right: readonly number[] | null): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function target(raw: unknown): TrackingTarget {
  const row = record(raw, "mpc target");
  if (row["applied"] !== false) throw new Error("A proposed action cannot be shown as applied");
  const costs = numbers(row["action_costs"], "action_costs");
  if (costs.length !== 2) throw new Error("A one-step target needs two action costs");
  const proposed = count(row["proposed_action"], "proposed_action");
  const [idle, active] = costs as [number, number];
  if (proposed !== (active < idle ? 1 : 0))
    throw new Error("Proposed action differs from the lowest source cost");
  return {
    id: text(row["case_id"], "case_id"),
    target: numbers(row["target"], "target"),
    costs: [idle, active],
    proposed: proposed as 0 | 1,
  };
}

function proposal(raw: unknown, edges: number): CodesignProposal {
  const row = record(raw, "proposal");
  if (row["applied"] !== false || row["hardware_actuation"] !== false)
    throw new Error("A co-design proposal cannot be shown as applied or actuated");
  const status = row["status"];
  if (!statuses.includes(status as ProposalStatus)) throw new Error("Unknown proposal status");
  const plan = nullableNumbers(row["plan"], "plan");
  if (status === "vetoed" && plan !== null) throw new Error("A vetoed proposal has no plan");
  const objective = row["objective"] === null ? null : finite(row["objective"], "objective");
  const indices = (field: string) =>
    numbers(row[field], field).map((index) => {
      if (!Number.isInteger(index) || index < 0 || index >= edges)
        throw new Error(`${field} names an unknown edge`);
      return index;
    });
  return {
    label: text(row["label"], "label"),
    status: status as ProposalStatus,
    reason: text(row["reason"], "reason"),
    plan,
    objective,
    constraints: list(row["constraint_values"], "constraint_values").map((values) =>
      numbers(values, "constraint_values"),
    ),
    adopted: nullableNumbers(row["adopted_plan"], "adopted_plan"),
    binding: indices("binding_edges"),
    violated: indices("violated_edges"),
  };
}

function latency(raw: unknown): LatencyVerdict {
  const row = record(raw, "latency");
  const passes = flag(row["passes"], "passes");
  const blockers = list(row["blockers"], "blockers").map((item) => text(item, "blocker"));
  if (passes === blockers.length > 0)
    throw new Error("A latency verdict and its blockers disagree");
  return {
    id: text(row["case_id"], "case_id"),
    passes,
    blockers,
    samples: numbers(row["round_latencies_s"], "round_latencies_s"),
    clock: text(row["clock_source"], "clock_source"),
  };
}

/**
 * Parse source control data, refusing applied proposals and fabricated telemetry.
 * @param raw - Decoded source JSON; extra fields are discarded.
 * @returns The parsed instrument or a visible refusal.
 */
export function parseControlInstrument(raw: unknown): Loaded<ControlInstrument> {
  try {
    const row = record(raw, "instrument");
    if (row["schema"] !== committed.schema)
      throw new Error("Unsupported control instrument schema");
    const codesign = record(row["codesign"], "codesign");
    const graph = record(codesign["graph"], "graph");
    const nodes = count(graph["nodes"], "nodes");
    const edges = list(graph["edges"], "edges").map((edge) => {
      const pair = numbers(edge, "edge");
      if (
        pair.length !== 2 ||
        pair.some((node) => !Number.isInteger(node) || node < 0 || node >= nodes)
      )
        throw new Error("An edge must join two known nodes");
      return [pair[0], pair[1]] as [number, number];
    });
    const proposals = list(codesign["proposals"], "proposals").map((item) =>
      proposal(item, edges.length),
    );
    if (proposals.length === 0)
      throw new Error("The co-design section needs at least one proposal");
    const constraintRows = list(codesign["constraint_rows"], "constraint_rows").map((name) =>
      text(name, "constraint_rows"),
    );
    if (
      proposals.some((item) =>
        item.constraints.some((values) => values.length !== constraintRows.length),
      )
    )
      throw new Error("Every constraint case needs one value per constraint row");
    let inForce: readonly number[] | null = null;
    for (const item of proposals) {
      if (item.status === "vetoed") continue;
      if (item.status === "accepted") inForce = item.plan;
      if (!same(item.adopted, inForce))
        throw new Error("Only an accepted proposal may change the adopted plan");
    }
    const trace = record(row["trace"], "trace");
    if (trace["observed_hardware_telemetry"] !== null)
      throw new Error(
        "This instrument shows simulated traces only; observed telemetry is unsupported",
      );
    if (trace["simulated"] !== true) throw new Error("The trace must be declared simulated");
    const times = numbers(trace["times"], "times");
    const order = numbers(trace["global_order"], "global_order");
    if (times.length === 0 || times.length !== order.length)
      throw new Error("Every trace sample needs one timestamp");
    if (times.some((time, index) => index > 0 && time <= (times[index - 1] as number)))
      throw new Error("Trace timestamps must increase");
    return {
      ok: true,
      value: {
        id: text(row["artifact_id"], "artifact_id"),
        boundary: text(row["claim_boundary"], "claim_boundary"),
        appliedRule: text(row["applied_status_rule"], "applied_status_rule"),
        targets: list(row["mpc_targets"], "mpc_targets").map(target),
        edges,
        constraintRows,
        proposals,
        latency: list(row["latency"], "latency").map(latency),
        trace: {
          times,
          order,
          source: text(trace["source"], "source"),
          boundary: text(trace["claim_boundary"], "claim_boundary"),
        },
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** Source-owned, committed control data parsed once at the boundary. */
export const controlInstrument = parseControlInstrument(committed);
