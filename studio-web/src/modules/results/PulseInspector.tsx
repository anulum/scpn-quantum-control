// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — pulse and analog feasibility instrument

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import type { PulseCase, PulseInstrument } from "./pulseInstrument";

const nanoseconds = (seconds: number) => `${(seconds * 1e9).toPrecision(4)} ns`;
const rate = (radPerSecond: number) => `${radPerSecond.toExponential(3)} rad/s`;

function laneOf(qubits: readonly number[]): string {
  return qubits.length === 1 ? `q${qubits[0]}` : `q${qubits.join("–q")}`;
}

/** Design pulses on one lane per drive or coupler, with violated pulses marked. */
function DesignTimeline({ item }: { readonly item: PulseCase }) {
  const lanes = [...new Set(item.design.map((pulse) => laneOf(pulse.qubits)))];
  const end = Math.max(...item.design.map((pulse) => pulse.startSeconds + pulse.durationSeconds));
  const x = (seconds: number) => 80 + (seconds / end) * 300;
  const violated = new Set(item.violations.map((violation) => violation.pulse));
  return (
    <svg
      viewBox={`0 0 400 ${30 + 30 * lanes.length}`}
      role="img"
      aria-label="Design timeline by drive and coupler"
      className="qsp-pulse-timeline"
    >
      {lanes.map((lane, row) => (
        <text key={lane} x={4} y={30 + 30 * row}>
          {lane}
        </text>
      ))}
      {item.design.map((pulse) => {
        const row = lanes.indexOf(laneOf(pulse.qubits));
        const marked = violated.has(pulse.index);
        return (
          <rect
            key={pulse.index}
            data-pulse={pulse.index}
            data-violated={marked}
            x={x(pulse.startSeconds)}
            width={x(pulse.startSeconds + pulse.durationSeconds) - x(pulse.startSeconds)}
            y={16 + 30 * row}
            height={18}
            fill="currentColor"
            opacity={marked ? 0.9 : 0.4}
            stroke={marked ? "currentColor" : "none"}
            strokeDasharray={marked ? "4 2" : undefined}
          >
            <title>
              {`pulse ${pulse.index} on ${laneOf(pulse.qubits)}: ${nanoseconds(pulse.startSeconds)} + ${nanoseconds(pulse.durationSeconds)}${marked ? " (violation)" : ""}`}
            </title>
          </rect>
        );
      })}
    </svg>
  );
}

function CaseView({ item, assessedAt }: { readonly item: PulseCase; readonly assessedAt: string }) {
  const limits = item.limits;
  const dossier =
    item.dossier === null
      ? null
      : `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(item.dossier, null, 2))}`;
  return (
    <>
      <p>{item.title}.</p>
      <dl aria-label="Limits">
        <dt>Limits from</dt>
        <dd>{limits.source === "design" ? "the designer" : "a provider capability snapshot"}</dd>
        <dt>Sample period</dt>
        <dd>{nanoseconds(limits.dtSeconds)}</dd>
        <dt>Grid</dt>
        <dd>{limits.granularitySamples} samples</dd>
        <dt>Drive conversion</dt>
        <dd>{rate(limits.rabiRatePerAmplitude)} per unit amplitude</dd>
        <dt>Coupler conversion</dt>
        <dd>
          {limits.couplerRatePerAmplitude === null
            ? "not declared"
            : `${rate(limits.couplerRatePerAmplitude)} per unit amplitude`}
        </dd>
        <dt>Largest amplitude</dt>
        <dd>{limits.maxAmplitude}</dd>
        <dt>Largest step</dt>
        <dd>{limits.maxSlew === null ? "not checked" : limits.maxSlew}</dd>
        <dt>Coupling map</dt>
        <dd>{limits.couplingMap.map((pair) => `q${pair.join("–q")}`).join(", ")}</dd>
      </dl>
      {item.snapshot !== null && (
        <p role="note" aria-label="Capability snapshot">
          {item.snapshot.provider} / {item.snapshot.backendName}: observed{" "}
          {item.snapshot.observedAt}, valid until {item.snapshot.expiresAt};{" "}
          {item.snapshot.validAtAssessment
            ? `valid at the assessment (${assessedAt})`
            : `expired before the assessment (${assessedAt})`}
          .
        </p>
      )}
      <table aria-label="Placement">
        <thead>
          <tr>
            <th scope="col">Logical qubit</th>
            <th scope="col">Physical qubit</th>
          </tr>
        </thead>
        <tbody>
          {item.placement.map((pair) => (
            <tr key={pair.logical}>
              <td>q{pair.logical}</td>
              <td>{pair.physical}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <DesignTimeline item={item} />
      <table aria-label="Design pulses">
        <thead>
          <tr>
            <th scope="col">Pulse</th>
            <th scope="col">Drives</th>
            <th scope="col">Start</th>
            <th scope="col">Duration</th>
            <th scope="col">Rates</th>
          </tr>
        </thead>
        <tbody>
          {item.design.map((pulse) => (
            <tr key={pulse.index}>
              <th scope="row">{pulse.index}</th>
              <td>{laneOf(pulse.qubits)}</td>
              <td>{nanoseconds(pulse.startSeconds)}</td>
              <td>{nanoseconds(pulse.durationSeconds)}</td>
              <td>{pulse.ratesRadPerSecond.map(rate).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {item.violations.length > 0 && (
        <ul aria-label="Violations">
          {item.violations.map((violation) => (
            <li key={violation.text}>
              <strong>{violation.code}</strong>
              {violation.pulse !== null && ` (pulse ${violation.pulse})`}
              {violation.channel !== null && ` (channel ${violation.channel})`}: {violation.text}
            </li>
          ))}
        </ul>
      )}
      {item.native.length > 0 && (
        <table aria-label="Native pulses">
          <caption>Native samples, {nanoseconds(limits.dtSeconds)} each</caption>
          <thead>
            <tr>
              <th scope="col">Channel</th>
              <th scope="col">Physical qubits</th>
              <th scope="col">Start sample</th>
              <th scope="col">Samples</th>
            </tr>
          </thead>
          <tbody>
            {item.native.map((pulse) => (
              <tr key={`${pulse.channel}:${pulse.t0}`}>
                <td>{pulse.channel}</td>
                <td>{pulse.physicalQubits.join(", ")}</td>
                <td>{pulse.t0}</td>
                <td>{pulse.samples.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {dossier === null ? (
        <p role="alert" aria-label="Dossier refused">
          No dossier: {item.dossierRefusal}.
        </p>
      ) : (
        <a href={dossier} download={`pulse-dossier-${item.caseId}.json`}>
          Download no-submit dossier
        </a>
      )}
    </>
  );
}

/**
 * Pulse and analog designs checked against declared or snapshot limits.
 *
 * Every design is shown in its own units with its placement and violations;
 * only a feasible design with a current snapshot offers a no-submit dossier.
 * Nothing here submits, and design units never make a schedule executable.
 *
 * @param props - Component properties.
 * @param props.instrument - Parsed instrument data, or the reason it was refused.
 * @returns The instrument, or an unverifiable block when its data was refused.
 */
export function PulseInspector({ instrument }: { readonly instrument: Loaded<PulseInstrument> }) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  if (!instrument.ok)
    return <Unverifiable surface="pulse_instrument_20261031.json" reason={instrument.reason} />;
  const item = instrument.value.cases[index] as PulseCase;
  return (
    <section aria-label="Pulse feasibility" className="qsp-panel">
      <h4>Pulse feasibility</h4>
      <p>{instrument.value.claimBoundary}.</p>
      <p role="note" aria-label="Design units">
        Design units only: {instrument.value.designClaimBoundary}.
      </p>
      <label htmlFor={selectId}>Design</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {instrument.value.cases.map((option, at) => (
          <option key={option.caseId} value={at}>
            {option.caseId}
          </option>
        ))}
      </select>
      <CaseView item={item} assessedAt={instrument.value.assessedAt} />
    </section>
  );
}
