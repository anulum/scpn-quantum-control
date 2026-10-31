// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — QEC decoder, readiness and future-resource instrument

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { decoderSampleAt, distinct, intervalNote } from "./qecInstrument";
import type { QecInstrument, ResourceRow } from "./qecInstrument";

/** The three separate views of the instrument. */
type Tab = "decoder" | "readiness" | "resources";

const TABS: readonly { readonly id: Tab; readonly label: string }[] = [
  { id: "decoder", label: "Decoder simulation" },
  { id: "readiness", label: "Offline readiness" },
  { id: "resources", label: "Future resources" },
];

const fraction = (value: number) => value.toPrecision(3);
const seconds = (value: number) => `${value.toPrecision(3)} s`;

function DecoderView({ instrument }: { readonly instrument: QecInstrument }) {
  const distances = distinct(instrument.decoder.experiments, (row) => row.distance);
  const rates = distinct(instrument.decoder.experiments, (row) => row.physicalErrorRate);
  const [distance, setDistance] = useState(distances[0] as number);
  const [rate, setRate] = useState(rates[0] as number);
  const distanceId = useId();
  const rateId = useId();
  const sample = decoderSampleAt(instrument, distance, rate);
  const syndromes = instrument.decoder.syndromes.filter((row) => row.distance === distance);
  return (
    <section aria-label="Decoder simulation">
      <p>{instrument.decoder.claimBoundary}.</p>
      <label htmlFor={distanceId}>Code distance</label>
      <select
        id={distanceId}
        value={distance}
        onChange={(event) => setDistance(Number(event.target.value))}
      >
        {distances.map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </select>
      <label htmlFor={rateId}>Physical error rate</label>
      <select id={rateId} value={rate} onChange={(event) => setRate(Number(event.target.value))}>
        {rates.map((value) => (
          <option key={value} value={value}>
            {value}
          </option>
        ))}
      </select>
      {sample === null ? (
        <p role="status">
          No sample was run at distance {distance} and rate {rate}.
        </p>
      ) : (
        <dl aria-label="Selected decoder sample">
          <dt>Logical failures</dt>
          <dd>
            {sample.failures} of {sample.shots}
          </dd>
          <dt>Failure rate</dt>
          <dd>{fraction(sample.rate)}</dd>
          <dt>{Math.round(sample.confidenceLevel * 100)} % interval</dt>
          <dd>
            [{fraction(sample.lower)}, {fraction(sample.upper)}]
          </dd>
          <dt>Interval method</dt>
          <dd>{intervalNote({ ...sample, trials: sample.shots })}</dd>
          <dt>Seed</dt>
          <dd>{sample.seed}</dd>
        </dl>
      )}
      <details>
        <summary>Single-error syndromes at distance {distance}</summary>
        <table aria-label="Single-error syndromes">
          <thead>
            <tr>
              <th scope="col">Flipped data qubit</th>
              <th scope="col">Defects</th>
              <th scope="col">Corrected</th>
            </tr>
          </thead>
          <tbody>
            {syndromes.map((row) => (
              <tr key={row.dataQubit}>
                <td>{row.dataQubit}</td>
                <td>{row.defects.join(", ")}</td>
                <td>{row.corrected ? "yes" : "no"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}

function ReadinessView({ instrument }: { readonly instrument: QecInstrument }) {
  const readiness = instrument.readiness;
  return (
    <section aria-label="Offline readiness">
      <p role="status" aria-label="Hardware logical-error reduction">
        Hardware logical-error reduction: {readiness.hardwareLogicalErrorReduction}
      </p>
      <dl>
        <dt>Decision</dt>
        <dd>{readiness.decision}</dd>
        <dt>Federation gate</dt>
        <dd>{readiness.admitted ? "admitted" : "refused"}</dd>
        <dt>Source</dt>
        <dd>
          <code>{readiness.source}</code>
        </dd>
      </dl>
      <h5>Supported</h5>
      <ul aria-label="Supported claims">
        {readiness.supported.map((claim) => (
          <li key={claim}>{claim}</li>
        ))}
      </ul>
      <h5>Blocked</h5>
      <ul aria-label="Blocked claims">
        {readiness.blocked.map((claim) => (
          <li key={claim}>{claim}</li>
        ))}
      </ul>
      <table aria-label="Offline decoder aggregates">
        <caption>Offline distance-{readiness.distance} decoder comparison</caption>
        <thead>
          <tr>
            <th scope="col">Circuit</th>
            <th scope="col">Decoder</th>
            <th scope="col">Noise</th>
            <th scope="col">Failures</th>
            <th scope="col">Interval</th>
          </tr>
        </thead>
        <tbody>
          {readiness.aggregates.map((row) => (
            <tr key={`${row.label}:${row.decoder}:${row.noiseModel}`}>
              <td>{row.label}</td>
              <td>{row.decoder}</td>
              <td>{row.noiseModel}</td>
              <td>
                {row.failures} of {row.trials}
              </td>
              <td title={intervalNote(row)}>
                [{fraction(row.lower)}, {fraction(row.upper)}]
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/** Log-log scaling plot of data-patch and scenario qubits against the physical error rate. */
function ScalingPlot({ rows }: { readonly rows: readonly ResourceRow[] }) {
  const estimated = rows.filter((row) => row.status === "estimated");
  const xs = estimated.map((row) => Math.log10(row.physicalErrorRate));
  const totals = estimated.flatMap((row) => [
    row.dataPatchPhysicalQubits,
    ...row.scenarios.map((scenario) => scenario.totalPhysicalQubits),
  ]);
  const ys = totals.map((value) => Math.log10(value));
  const [xMin, xMax] = [Math.min(...xs), Math.max(...xs)];
  const [yMin, yMax] = [Math.min(...ys), Math.max(...ys)];
  const x = (value: number) => 40 + ((Math.log10(value) - xMin) / (xMax - xMin || 1)) * 340;
  // Every scenario adds factories to routed patches, so its total exceeds the
  // data patches and the qubit axis always has a positive range.
  const y = (value: number) => 170 - ((Math.log10(value) - yMin) / (yMax - yMin)) * 150;
  return (
    <svg
      viewBox="0 0 400 200"
      role="img"
      aria-label="Physical qubits against physical error rate, data patches and the scenario range"
      className="qsp-qec-scaling"
    >
      {estimated.map((row) => {
        const spread = row.scenarios.map((scenario) => scenario.totalPhysicalQubits);
        return (
          <g key={row.physicalErrorRate}>
            <line
              x1={x(row.physicalErrorRate)}
              x2={x(row.physicalErrorRate)}
              y1={y(Math.min(...spread))}
              y2={y(Math.max(...spread))}
              stroke="currentColor"
              strokeWidth={6}
              opacity={0.35}
            />
            <circle
              cx={x(row.physicalErrorRate)}
              cy={y(row.dataPatchPhysicalQubits)}
              r={4}
              fill="currentColor"
            />
          </g>
        );
      })}
    </svg>
  );
}

function ResourceView({ instrument }: { readonly instrument: QecInstrument }) {
  const resources = instrument.resources;
  const [index, setIndex] = useState(0);
  const rateId = useId();
  const row = resources.rows[index] as ResourceRow;
  return (
    <section aria-label="Future resources">
      <p>{resources.claimBoundary}</p>
      <label htmlFor={rateId}>Assumed physical error rate</label>
      <select id={rateId} value={index} onChange={(event) => setIndex(Number(event.target.value))}>
        {resources.rows.map((option, at) => (
          <option key={option.physicalErrorRate} value={at}>
            {option.physicalErrorRate}
          </option>
        ))}
      </select>
      {row.status === "refused" ? (
        <p role="alert" aria-label="Resource estimate refused">
          Refused by the source: {row.reason}
        </p>
      ) : (
        <>
          <dl aria-label="Resource estimate">
            <dt>Code distance</dt>
            <dd>{row.codeDistance}</dd>
            <dt>Data-patch physical qubits</dt>
            <dd>{row.dataPatchPhysicalQubits}</dd>
            <dt>T count</dt>
            <dd>{row.totalTCount}</dd>
            <dt>Syndrome time floor</dt>
            <dd>{seconds(row.syndromeTimeFloorSeconds)}</dd>
            <dt>Provenance</dt>
            <dd>
              <code>{row.payloadSha256}</code>
            </dd>
          </dl>
          <table aria-label="Overhead scenarios">
            <caption>{resources.scenarioClaimBoundary}</caption>
            <thead>
              <tr>
                <th scope="col">Scenario</th>
                <th scope="col">Routing factor</th>
                <th scope="col">Factories</th>
                <th scope="col">Total qubits</th>
                <th scope="col">T-state time floor</th>
                <th scope="col">Decoding time floor</th>
                <th scope="col">Scenario time floor</th>
              </tr>
            </thead>
            <tbody>
              {row.scenarios.map((scenario) => (
                <tr key={scenario.label}>
                  <td>{scenario.label}</td>
                  <td>{scenario.routingOverheadFactor}</td>
                  <td>{scenario.factoryCount}</td>
                  <td>{scenario.totalPhysicalQubits}</td>
                  <td>{seconds(scenario.tStateTimeFloorSeconds)}</td>
                  <td>{seconds(scenario.decodingTimeFloorSeconds)}</td>
                  <td>{seconds(scenario.scenarioTimeFloorSeconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      <h5>Excluded from the base estimate</h5>
      <ul aria-label="Excluded overheads">
        {resources.excluded.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <ScalingPlot rows={resources.rows} />
      <p>
        Dots are data patches; bars span the stated overhead scenarios. No named device is assessed.
      </p>
    </section>
  );
}

/**
 * Decoder samples, the offline readiness decision and future-resource estimates,
 * each in its own tab, with every value taken from the committed source data.
 *
 * @param props - Component properties.
 * @param props.instrument - Parsed instrument data, or the reason it was refused.
 * @returns The instrument, or an unverifiable block when its data was refused.
 */
export function QecInspector({ instrument }: { readonly instrument: Loaded<QecInstrument> }) {
  const [tab, setTab] = useState<Tab>("decoder");
  if (!instrument.ok)
    return (
      <Unverifiable surface="qec_resource_instrument_20261031.json" reason={instrument.reason} />
    );
  return (
    <section aria-label="QEC and future resources" className="qsp-panel">
      <h4>QEC and future resources</h4>
      <p>{instrument.value.claimBoundary}.</p>
      <div role="tablist" aria-label="QEC views">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={tab === item.id}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {tab === "decoder" && <DecoderView instrument={instrument.value} />}
      {tab === "readiness" && <ReadinessView instrument={instrument.value} />}
      {tab === "resources" && <ResourceView instrument={instrument.value} />}
    </section>
  );
}
