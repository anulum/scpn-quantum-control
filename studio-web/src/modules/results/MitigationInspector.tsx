// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — readout correction, extrapolation and hardware-pair instrument

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { describeMap, readoutCaseCsv } from "./mitigationInstrument";
import type {
  Calibration,
  MitigationInstrument,
  ReadoutCase,
  Zne,
  Hardware,
} from "./mitigationInstrument";

/** The three separate views of the instrument. */
type Tab = "readout" | "zne" | "hardware";

const TABS: readonly { readonly id: Tab; readonly label: string }[] = [
  { id: "readout", label: "Readout correction" },
  { id: "zne", label: "Zero-noise extrapolation" },
  { id: "hardware", label: "Hardware pairs" },
];

const value = (number: number) => number.toPrecision(4);

function CalibrationDetails({
  calibration,
  labels,
  shots,
}: {
  readonly calibration: Calibration;
  readonly labels: readonly string[];
  readonly shots: number;
}) {
  const calibrationShots = calibration.shotsByPreparedState.reduce((sum, item) => sum + item, 0);
  return (
    <details>
      <summary>Calibration matrix and shot accounting</summary>
      <table aria-label="Calibration matrix">
        <caption>Rows are observed labels, columns prepared labels</caption>
        <thead>
          <tr>
            <th scope="col">Observed</th>
            {labels.map((label) => (
              <th key={label} scope="col">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {calibration.matrix.map((row, index) => (
            <tr key={labels[index]}>
              <th scope="row">{labels[index]}</th>
              {row.map((entry, column) => (
                <td key={labels[column]}>{value(entry)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <dl aria-label="Shot accounting">
        <dt>Calibration shots</dt>
        <dd>
          {calibrationShots} over {labels.length} prepared states
        </dd>
        <dt>Measurement shots</dt>
        <dd>{shots}</dd>
        <dt>Condition number</dt>
        <dd>{value(calibration.conditionNumber)}</dd>
      </dl>
    </details>
  );
}

/** Raw frequencies beside signed corrected values, with a zero line and error bars. */
function ReadoutPlot({ item }: { readonly item: ReadoutCase }) {
  const corrected = item.status === "corrected" ? item.corrected : null;
  const extent = Math.max(
    1,
    ...item.rawFrequencies,
    ...(corrected === null
      ? []
      : corrected.quasiprobabilities.map(
          (entry, index) => Math.abs(entry) + (corrected.totalStandardError[index] as number),
        )),
  );
  const width = 80 * item.labels.length + 40;
  const y = (number: number) => 110 - (number / extent) * 90;
  return (
    <svg
      viewBox={`0 0 ${width} 230`}
      role="img"
      aria-label="Raw and corrected values by label"
      className="qsp-mitigation-plot"
    >
      <line x1={20} x2={width - 20} y1={y(0)} y2={y(0)} stroke="currentColor" />
      {item.labels.map((label, index) => {
        const left = 40 + 80 * index;
        const raw = item.rawFrequencies[index] as number;
        const signed = corrected?.quasiprobabilities[index];
        return (
          <g key={label} data-label={label}>
            <rect x={left} width={24} y={y(raw)} height={y(0) - y(raw)} opacity={0.4}>
              <title>{`${label} raw ${value(raw)}`}</title>
            </rect>
            {signed !== undefined && corrected !== null && (
              <>
                <rect
                  x={left + 28}
                  width={24}
                  y={Math.min(y(signed), y(0))}
                  height={Math.abs(y(signed) - y(0))}
                  fill="currentColor"
                >
                  <title>{`${label} corrected ${value(signed)}`}</title>
                </rect>
                <line
                  x1={left + 40}
                  x2={left + 40}
                  y1={y(signed + (corrected.totalStandardError[index] as number))}
                  y2={y(signed - (corrected.totalStandardError[index] as number))}
                  stroke="currentColor"
                />
              </>
            )}
            <text x={left + 26} y={225} textAnchor="middle">
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ReadoutView({ cases }: { readonly cases: readonly ReadoutCase[] }) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  const item = cases[index] as ReadoutCase;
  const csv = `data:text/csv;charset=utf-8,${encodeURIComponent(readoutCaseCsv(item))}`;
  return (
    <section aria-label="Readout correction">
      <label htmlFor={selectId}>Readout case</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {cases.map((option, at) => (
          <option key={option.caseId} value={at}>
            {option.caseId}
          </option>
        ))}
      </select>
      <p>{item.title}.</p>
      <p role="note" aria-label="Measurement map">
        {describeMap(item.measurementMap)}
      </p>
      {item.unmeasuredQubits.length > 0 && (
        <p>Not measured: {item.unmeasuredQubits.map((qubit) => `q${qubit}`).join(", ")}</p>
      )}
      {item.status === "refused" ? (
        <p role="alert" aria-label="Correction refused">
          Refused by the source: {item.reason}. The raw counts below are unchanged.
        </p>
      ) : (
        <p role="status" aria-label="Signed values">
          Signed values sum to {value(item.corrected.sum)}; they are not clipped or renormalised.
          {item.corrected.negativeLabels.length > 0 &&
            ` Negative at ${item.corrected.negativeLabels.join(", ")}: these are not probabilities.`}
        </p>
      )}
      <table aria-label="Raw and corrected values">
        <thead>
          <tr>
            <th scope="col">Label</th>
            <th scope="col">Raw count</th>
            <th scope="col">Raw frequency</th>
            <th scope="col">Corrected</th>
            <th scope="col">Standard error</th>
          </tr>
        </thead>
        <tbody>
          {item.labels.map((label, at) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{item.rawCounts[at]}</td>
              <td>{value(item.rawFrequencies[at] as number)}</td>
              {item.status === "corrected" ? (
                <>
                  <td>
                    {value(item.corrected.quasiprobabilities[at] as number)}
                    {(item.corrected.quasiprobabilities[at] as number) < 0 && " (negative)"}
                  </td>
                  <td
                    title={`sampling ${value(item.corrected.samplingStandardError[at] as number)}, calibration ${value(item.corrected.calibrationStandardError[at] as number)}`}
                  >
                    {value(item.corrected.totalStandardError[at] as number)}
                  </td>
                </>
              ) : (
                <>
                  <td>refused</td>
                  <td>—</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <ReadoutPlot item={item} />
      <CalibrationDetails calibration={item.calibration} labels={item.labels} shots={item.shots} />
      <a href={csv} download={`readout-${item.caseId}.csv`}>
        Download CSV
      </a>
    </section>
  );
}

/** Raw and corrected points against noise scale, and the extrapolated interval at zero. */
function ZnePlot({ zne }: { readonly zne: Zne }) {
  const scales = zne.points.map((point) => point.noiseScale);
  const values = zne.points.flatMap((point) => [
    point.rawExpectation,
    point.correctedExpectation + point.correctedStandardError,
    point.correctedExpectation - point.correctedStandardError,
  ]);
  const bottom = Math.min(...values, zne.extrapolation.low);
  const top = Math.max(...values, zne.extrapolation.high);
  const x = (scale: number) => 40 + (scale / Math.max(...scales)) * 320;
  const y = (number: number) => 180 - ((number - bottom) / (top - bottom || 1)) * 160;
  return (
    <svg
      viewBox="0 0 400 200"
      role="img"
      aria-label="Raw and corrected observable against noise scale"
      className="qsp-mitigation-plot"
    >
      <line
        x1={x(0)}
        x2={x(0)}
        y1={y(zne.extrapolation.low)}
        y2={y(zne.extrapolation.high)}
        stroke="currentColor"
        strokeWidth={4}
      />
      <circle cx={x(0)} cy={y(zne.extrapolation.zeroNoiseEstimate)} r={4} fill="currentColor" />
      {zne.points.map((point) => (
        <g key={point.noiseScale}>
          <circle
            cx={x(point.noiseScale)}
            cy={y(point.rawExpectation)}
            r={4}
            fill="none"
            stroke="currentColor"
          />
          <line
            x1={x(point.noiseScale)}
            x2={x(point.noiseScale)}
            y1={y(point.correctedExpectation - point.correctedStandardError)}
            y2={y(point.correctedExpectation + point.correctedStandardError)}
            stroke="currentColor"
          />
          <circle
            cx={x(point.noiseScale)}
            cy={y(point.correctedExpectation)}
            r={3}
            fill="currentColor"
          />
        </g>
      ))}
    </svg>
  );
}

function ZneView({ zne }: { readonly zne: Zne }) {
  const fit = zne.extrapolation;
  return (
    <section aria-label="Zero-noise extrapolation">
      <p>{zne.claimBoundary}.</p>
      <p role="note" aria-label="Measurement map">
        {describeMap(zne.measurementMap)}
      </p>
      <table aria-label="Observable at each noise scale">
        <caption>{zne.observable}</caption>
        <thead>
          <tr>
            <th scope="col">Noise scale</th>
            <th scope="col">Shots</th>
            <th scope="col">Raw</th>
            <th scope="col">Corrected</th>
            <th scope="col">Standard error</th>
          </tr>
        </thead>
        <tbody>
          {zne.points.map((point) => (
            <tr key={point.noiseScale}>
              <th scope="row">{point.noiseScale}</th>
              <td>{point.shots}</td>
              <td>{value(point.rawExpectation)}</td>
              <td>{value(point.correctedExpectation)}</td>
              <td>{value(point.correctedStandardError)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl aria-label="Extrapolation">
        <dt>Zero-noise estimate</dt>
        <dd>
          {value(fit.zeroNoiseEstimate)} ± {value(fit.standardError)}
        </dd>
        <dt>{Math.round(fit.coverage * 100)} % interval</dt>
        <dd>
          [{value(fit.low)}, {value(fit.high)}]
        </dd>
        <dt>Method</dt>
        <dd>
          {fit.method}, polynomial order {fit.order}
        </dd>
        <dt>Fit residual</dt>
        <dd>{value(fit.fitResidual)}</dd>
      </dl>
      <ZnePlot zne={zne} />
      <p>
        Hollow circles are raw values, filled ones corrected with their standard error; the bar at
        zero is the extrapolated interval.
      </p>
      <CalibrationDetails
        calibration={zne.calibration}
        labels={zne.labels}
        shots={zne.points.reduce((sum, point) => sum + point.shots, 0)}
      />
    </section>
  );
}

function HardwareView({ hardware }: { readonly hardware: Hardware }) {
  return (
    <section aria-label="Hardware pairs">
      <p role="note" aria-label="Correction caveat">
        {hardware.method}: {hardware.caveat}
      </p>
      <dl>
        <dt>Federation gate</dt>
        <dd>admitted</dd>
        <dt>Source</dt>
        <dd>
          <code>{hardware.source}</code>
        </dd>
        <dt>Bundle digest</dt>
        <dd>
          <code>{hardware.bundleDigest}</code>
        </dd>
      </dl>
      <table aria-label="Measured parity pairs">
        <thead>
          <tr>
            <th scope="col">Dataset</th>
            <th scope="col">Comparison</th>
            <th scope="col">Depth</th>
            <th scope="col">Raw left, right</th>
            <th scope="col">Corrected left, right</th>
            <th scope="col">Raw relative</th>
            <th scope="col">Corrected relative</th>
            <th scope="col">Samples</th>
          </tr>
        </thead>
        <tbody>
          {hardware.pairs.map((pair) => (
            <tr key={`${pair.dataset}:${pair.comparison}:${pair.depth}`}>
              <td>{pair.dataset}</td>
              <td>{pair.comparison}</td>
              <td>{pair.depth}</td>
              <td>
                {value(pair.leftRaw)}, {value(pair.rightRaw)}
              </td>
              <td>
                {value(pair.leftCorrected)}, {value(pair.rightCorrected)}
              </td>
              <td>{value(pair.rawRelative)}</td>
              <td>{value(pair.correctedRelative)}</td>
              <td>
                {pair.nLeft}, {pair.nRight}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/**
 * Readout correction cases, a zero-noise extrapolation and measured hardware
 * pairs, each in its own tab, with every value taken from the committed source data.
 *
 * @param props - Component properties.
 * @param props.instrument - Parsed instrument data, or the reason it was refused.
 * @returns The instrument, or an unverifiable block when its data was refused.
 */
export function MitigationInspector({
  instrument,
}: {
  readonly instrument: Loaded<MitigationInstrument>;
}) {
  const [tab, setTab] = useState<Tab>("readout");
  if (!instrument.ok)
    return (
      <Unverifiable surface="mitigation_instrument_20261031.json" reason={instrument.reason} />
    );
  return (
    <section aria-label="Mitigation and readout" className="qsp-panel">
      <h4>Mitigation and readout</h4>
      <p>{instrument.value.claimBoundary}.</p>
      <p>{instrument.value.labelOrder}.</p>
      <div role="tablist" aria-label="Mitigation views">
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
      {tab === "readout" && <ReadoutView cases={instrument.value.readout.cases} />}
      {tab === "zne" && <ZneView zne={instrument.value.zne} />}
      {tab === "hardware" && <HardwareView hardware={instrument.value.hardware} />}
    </section>
  );
}
