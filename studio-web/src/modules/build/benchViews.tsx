// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — live bench views: phase circle, R(t) chart, statistics and bench panel

import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { niceTicks, orderVector, phaseBin, timeToFraction } from "./benchModel";
import type { BenchRun } from "./benchModel";

const CHART = { width: 560, height: 200, left: 40, right: 12, top: 10, bottom: 26 } as const;

/**
 * Oscillator phases on the unit circle with the order vector R·e^{iψ}.
 *
 * @param props - Phases and what they are.
 * @returns The phase circle figure.
 */
export function PhaseCircle({
  phases,
  label,
}: {
  /** Oscillator phases in radians; at least one. */
  readonly phases: ArrayLike<number>;
  /** What the phases are, such as "Final phases". */
  readonly label: string;
}) {
  const { r, psi } = orderVector(phases);
  const radius = 70;
  const centre = 90;
  const dots = Array.from(phases, (theta, index) => ({ theta, index }));
  return (
    <figure className="qsp-phase-circle">
      <svg
        viewBox="0 0 180 180"
        role="img"
        aria-label={`${label} of ${phases.length} oscillators; R = ${r.toFixed(3)}, ψ = ${psi.toFixed(3)} rad`}
      >
        <circle className="qsp-phase-ring" cx={centre} cy={centre} r={radius} />
        <line
          className="qsp-phase-axis"
          x1={centre - radius}
          y1={centre}
          x2={centre + radius}
          y2={centre}
        />
        <line
          className="qsp-phase-axis"
          x1={centre}
          y1={centre - radius}
          x2={centre}
          y2={centre + radius}
        />
        {dots.map(({ theta, index }) => (
          <circle
            key={`phase:${index}`}
            className={`qsp-phase-dot qsp-phase-${phaseBin(theta)}`}
            cx={centre + radius * Math.cos(theta)}
            cy={centre - radius * Math.sin(theta)}
            r="5"
          />
        ))}
        <line
          className="qsp-order-arrow"
          x1={centre}
          y1={centre}
          x2={centre + radius * r * Math.cos(psi)}
          y2={centre - radius * r * Math.sin(psi)}
        />
        <circle
          className="qsp-order-tip"
          cx={centre + radius * r * Math.cos(psi)}
          cy={centre - radius * r * Math.sin(psi)}
          r="3"
        />
      </svg>
      <figcaption>
        {label} · R <span className="qsp-mono">{r.toFixed(3)}</span>
      </figcaption>
    </figure>
  );
}

/** One series drawn on the order-parameter chart. */
export interface ChartSeries {
  /** Order-parameter samples. */
  readonly values: ArrayLike<number>;
  /** Time between samples. */
  readonly dt: number;
  /** Series colour index, or `null` for the current run in the accent colour. */
  readonly colour: number | null;
  /** Name in the legend. */
  readonly label: string;
}

/**
 * Order parameter R(t) with axes, grid and overlaid runs.
 *
 * @param props - Series to draw; the first is the current run.
 * @returns The chart figure.
 */
export function OrderChart({ series }: { readonly series: readonly ChartSeries[] }) {
  const end = Math.max(...series.map((entry) => (entry.values.length - 1) * entry.dt), 0);
  const timeTicks = niceTicks(0, end > 0 ? end : 1, 5);
  const span = timeTicks.at(-1) as number;
  const rTicks = niceTicks(0, 1, 5);
  const plotWidth = CHART.width - CHART.left - CHART.right;
  const plotHeight = CHART.height - CHART.top - CHART.bottom;
  const x = (time: number) => CHART.left + (time / span) * plotWidth;
  const y = (value: number) => CHART.top + (1 - Math.min(1, Math.max(0, value))) * plotHeight;
  const path = (entry: ChartSeries) =>
    Array.from(
      entry.values,
      (value, index) =>
        `${index === 0 ? "M" : "L"}${x(index * entry.dt).toFixed(2)} ${y(value).toFixed(2)}`,
    ).join(" ");
  return (
    <figure className="qsp-order-chart">
      <svg
        viewBox={`0 0 ${CHART.width} ${CHART.height}`}
        role="img"
        aria-label={`Order parameter R over time for ${series.map((entry) => entry.label).join(", ")}`}
      >
        {rTicks.map((tick) => (
          <g key={`r:${tick}`} className="qsp-chart-grid">
            <line x1={CHART.left} x2={CHART.width - CHART.right} y1={y(tick)} y2={y(tick)} />
            <text x={CHART.left - 6} y={y(tick) + 3} textAnchor="end">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        {timeTicks.map((tick) => (
          <g key={`t:${tick}`} className="qsp-chart-grid">
            <line x1={x(tick)} x2={x(tick)} y1={CHART.top} y2={CHART.height - CHART.bottom} />
            <text x={x(tick)} y={CHART.height - 8} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        {[...series].reverse().map((entry) => (
          <path
            key={entry.label}
            className={
              entry.colour === null ? "qsp-series-current" : `qsp-series qsp-series-${entry.colour}`
            }
            d={path(entry)}
          />
        ))}
      </svg>
      <figcaption>
        <span>R(t) · time in model units</span>
        <span className="qsp-legend">
          {series.map((entry) => (
            <span key={entry.label}>
              <span
                className={
                  entry.colour === null
                    ? "qsp-swatch qsp-swatch-current"
                    : `qsp-swatch qsp-swatch-${entry.colour}`
                }
                aria-hidden="true"
              />
              {entry.label}
            </span>
          ))}
        </span>
      </figcaption>
    </figure>
  );
}

/**
 * Statistics read from one run's order parameter and final phases.
 *
 * @param props - Order-parameter samples, sample spacing and final phases.
 * @returns A definition list of the run statistics.
 */
export function RunStatistics({
  values,
  dt,
  thetaFinal,
}: {
  /** Order-parameter samples, index 0 at time 0. */
  readonly values: ArrayLike<number>;
  /** Time between samples. */
  readonly dt: number;
  /** Final phases of the run. */
  readonly thetaFinal: ArrayLike<number>;
}) {
  const reach = timeToFraction(values, dt, 0.9);
  return (
    <dl className="qsp-summary qsp-run-stats">
      <dt>R initial</dt>
      <dd>{(values[0] as number).toFixed(3)}</dd>
      <dt>R final</dt>
      <dd>{(values[values.length - 1] as number).toFixed(3)}</dd>
      <dt>Time to 90 % of final R</dt>
      <dd>{reach === null ? "not reached" : reach.toFixed(2)}</dd>
      <dt>R of final phases</dt>
      <dd>{orderVector(thetaFinal).r.toFixed(3)}</dd>
    </dl>
  );
}

/** The four views of the bench panel. */
export type BenchView = "messages" | "progress" | "log" | "runs";

const views: readonly (readonly [BenchView, string])[] = [
  ["messages", "Messages"],
  ["progress", "Progress"],
  ["log", "Log"],
  ["runs", "Runs"],
];

/**
 * Messages, progress, the run log and the run table under the bench.
 *
 * @param props - What each view shows and how runs are compared.
 * @returns The bench panel.
 */
export function BenchPanel({
  messages,
  progress,
  runs,
  toggle,
  clear,
}: {
  /** Current refusals and verification states, in order. */
  readonly messages: readonly { readonly tone: "refusal" | "note"; readonly text: string }[];
  /** What the instrument is doing now. */
  readonly progress: string;
  /** Session runs, newest first. */
  readonly runs: readonly BenchRun[];
  /** Draw or stop drawing a run over the current result. */
  readonly toggle: (number: number) => void;
  /** Forget the session runs. */
  readonly clear: () => void;
}) {
  const [view, setView] = useState<BenchView>("runs");
  const refs = useRef(new Map<BenchView, HTMLButtonElement>());
  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (step === 0) return;
    event.preventDefault();
    const [next] = views[(index + step + views.length) % views.length] as (typeof views)[number];
    setView(next);
    refs.current.get(next)?.focus();
  };
  const refusals = messages.filter((message) => message.tone === "refusal").length;
  return (
    <section className="qsp-card qsp-bench-panel" aria-label="Bench panel">
      <div role="tablist" aria-label="Bench views" className="qsp-tabs">
        {views.map(([id, label], index) => (
          <button
            key={id}
            ref={(element) => {
              if (element !== null) refs.current.set(id, element);
            }}
            type="button"
            role="tab"
            id={`bench-${id}-tab`}
            aria-selected={view === id}
            aria-controls={`bench-${id}`}
            tabIndex={view === id ? 0 : -1}
            onClick={() => setView(id)}
            onKeyDown={(event) => move(event, index)}
          >
            {label}
            {id === "messages" && refusals > 0 ? ` (${refusals})` : null}
            {id === "runs" && runs.length > 0 ? ` (${runs.length})` : null}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id="bench-messages"
        aria-labelledby="bench-messages-tab"
        hidden={view !== "messages"}
      >
        {messages.length === 0 ? (
          <p className="qsp-empty">No messages.</p>
        ) : (
          <ul className="qsp-messages">
            {messages.map((message) => (
              <li key={message.text} data-tone={message.tone}>
                {message.text}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div
        role="tabpanel"
        id="bench-progress"
        aria-labelledby="bench-progress-tab"
        hidden={view !== "progress"}
      >
        <p role="status">{progress}</p>
      </div>
      <div role="tabpanel" id="bench-log" aria-labelledby="bench-log-tab" hidden={view !== "log"}>
        {runs.length === 0 ? (
          <p className="qsp-empty">
            No runs yet in this session. Run the simulation to record one.
          </p>
        ) : (
          <ol className="qsp-log">
            {runs.map((run) => (
              <li key={`log:${run.number}`} className="qsp-mono">
                run {run.number} · {run.controls.mode} · N={run.controls.n} K=
                {run.controls.coupling.toFixed(2)} spread=
                {run.controls.spread.toFixed(2)} steps={run.controls.steps} · R final{" "}
                {(run.orderParameter[run.orderParameter.length - 1] as number).toFixed(3)} ·{" "}
                {run.wallMs.toFixed(0)} ms browser wall time
              </li>
            ))}
          </ol>
        )}
      </div>
      <div
        role="tabpanel"
        id="bench-runs"
        aria-labelledby="bench-runs-tab"
        hidden={view !== "runs"}
      >
        {runs.length === 0 ? (
          <p className="qsp-empty">
            No runs yet in this session. Run the simulation to record one.
          </p>
        ) : (
          <>
            <table className="qsp-runs">
              <caption>Session runs, newest first; nothing is stored</caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Mode</th>
                  <th scope="col">N</th>
                  <th scope="col">K</th>
                  <th scope="col">Steps</th>
                  <th scope="col">R final</th>
                  <th scope="col">Overlay</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={`run:${run.number}`}>
                    <td>
                      <span className={`qsp-swatch qsp-swatch-${run.colour}`} aria-hidden="true" />
                      {run.number}
                    </td>
                    <td>{run.controls.mode}</td>
                    <td>{run.controls.n}</td>
                    <td>{run.controls.coupling.toFixed(2)}</td>
                    <td>{run.controls.steps}</td>
                    <td>
                      {(run.orderParameter[run.orderParameter.length - 1] as number).toFixed(3)}
                    </td>
                    <td>
                      <label>
                        <input
                          type="checkbox"
                          checked={run.overlaid}
                          onChange={() => toggle(run.number)}
                        />
                        <span className="qsp-visually-hidden">Overlay run {run.number}</span>
                      </label>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button type="button" onClick={clear}>
              Clear session runs
            </button>
          </>
        )}
      </div>
    </section>
  );
}
