// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — control proposal and topology co-design instrument

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import type { CodesignProposal, ControlInstrument } from "./controlInstrument";

const nodePosition = (node: number, total: number) => {
  const angle = (2 * Math.PI * node) / total - Math.PI / 2;
  return { x: 60 + 45 * Math.cos(angle), y: 60 + 45 * Math.sin(angle) };
};

function edgeState(
  proposal: CodesignProposal,
  edge: number,
): "violated" | "binding" | "within cap" {
  if (proposal.violated.includes(edge)) return "violated";
  if (proposal.binding.includes(edge)) return "binding";
  return "within cap";
}

/**
 * Inspect source control proposals, constraints, latency verdicts and a simulated trace.
 * The browser shows source values only; it never applies a proposal or owns actuation.
 * @param props - Component inputs.
 * @param props.instrument - Parsed source data or its refusal.
 * @returns Read-only control and co-design views.
 */
export function ControlInspector({
  instrument,
}: {
  readonly instrument: Loaded<ControlInstrument>;
}) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  if (!instrument.ok)
    return <Unverifiable surface="control instrument" reason={instrument.reason} />;
  const data = instrument.value;
  const nodeIds = [...new Set(data.edges.flat())].sort((a, b) => a - b);
  const nodes = Math.max(...nodeIds) + 1;
  const selected = data.proposals[index] as CodesignProposal;
  return (
    <section aria-label="Control proposals and co-design" className="qsp-panel">
      <h4>Control proposals and co-design</h4>
      <p>{data.boundary}.</p>
      <p role="note">{data.appliedRule}.</p>
      <table aria-label="Tracking targets">
        <thead>
          <tr>
            <th scope="col">Target case</th>
            <th scope="col">Target</th>
            <th scope="col">Cost, action 0</th>
            <th scope="col">Cost, action 1</th>
            <th scope="col">Proposed action</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {data.targets.map((item) => (
            <tr key={item.id}>
              <th scope="row">{item.id}</th>
              <td>[{item.target.join(", ")}]</td>
              <td>{item.costs[0]}</td>
              <td>{item.costs[1]}</td>
              <td>{item.proposed}</td>
              <td>Proposed, not applied</td>
            </tr>
          ))}
        </tbody>
      </table>
      <label htmlFor={selectId}>Co-design proposal</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {data.proposals.map((item, at) => (
          <option key={item.label} value={at}>
            {item.label}
          </option>
        ))}
      </select>
      <dl aria-label="Proposal outcome">
        <dt>Status</dt>
        <dd>{selected.status}</dd>
        <dt>Reason</dt>
        <dd>{selected.reason}</dd>
        <dt>Proposed plan</dt>
        <dd>{selected.plan === null ? "No plan" : `[${selected.plan.join(", ")}]`}</dd>
        <dt>Plan in force</dt>
        <dd>{selected.adopted === null ? "None accepted" : `[${selected.adopted.join(", ")}]`}</dd>
        <dt>Applied</dt>
        <dd>No, no runtime actuation evidence</dd>
      </dl>
      <svg
        role="img"
        aria-label="Coupling graph constraints"
        viewBox="0 0 120 120"
        width="160"
        height="160"
      >
        {data.edges.map(([from, to], edge) => {
          const a = nodePosition(from, nodes);
          const b = nodePosition(to, nodes);
          const state = edgeState(selected, edge);
          return (
            <line
              key={`${from}-${to}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              data-edge={`${from}-${to}`}
              data-state={state}
              stroke={
                state === "violated" ? "#b3261e" : state === "binding" ? "#b26a00" : "#5f6368"
              }
              strokeWidth={state === "within cap" ? 2 : 5}
              strokeDasharray={state === "violated" ? "6 3" : undefined}
            />
          );
        })}
        {nodeIds.map((node) => {
          const point = nodePosition(node, nodes);
          return <circle key={`node-${node}`} cx={point.x} cy={point.y} r={7} fill="#1a73e8" />;
        })}
      </svg>
      <ul aria-label="Edge constraint states">
        {data.edges.map(([from, to], edge) => (
          <li key={`${from}-${to}`}>
            Edge {from}-{to}: {edgeState(selected, edge)}
          </li>
        ))}
      </ul>
      {selected.constraints.length > 0 && (
        <table aria-label="Constraint values">
          <thead>
            <tr>
              <th scope="col">Case</th>
              {data.constraintRows.map((name) => (
                <th key={name} scope="col">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {selected.constraints.map((values, at) => (
              <tr key={at === 0 ? "nominal" : `disturbance-${at}`}>
                <th scope="row">{at === 0 ? "nominal" : `held-out disturbance ${at}`}</th>
                {data.constraintRows.map((name, column) => (
                  <td key={name}>{values[column]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <table aria-label="Latency budget verdicts">
        <thead>
          <tr>
            <th scope="col">Series</th>
            <th scope="col">Replayed samples (s)</th>
            <th scope="col">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {data.latency.map((item) => (
            <tr key={item.id}>
              <th scope="row">{item.id}</th>
              <td>{item.samples.join(", ")}</td>
              <td>
                {item.passes
                  ? `within budget (${item.clock})`
                  : `latency claim refused: ${item.blockers.join("; ")}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <table aria-label="Simulated closed-loop trace">
        <caption>Simulated trace: {data.trace.source}</caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Global order</th>
          </tr>
        </thead>
        <tbody>
          {data.trace.times.map((time, at) => (
            <tr key={time}>
              <td>{String(time)}</td>
              <td>{String(data.trace.order[at])}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p role="status">Observed hardware telemetry: unavailable. {data.trace.boundary}.</p>
    </section>
  );
}
