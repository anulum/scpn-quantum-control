// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — static plots of source-owned learning values

import type { NamedMetric } from "./learningDomains";

const WIDTH = 320;
const HEIGHT = 120;
const PAD = 8;

function scale(value: number, low: number, high: number, span: number): number {
  return high === low ? span / 2 : ((value - low) / (high - low)) * span;
}

/**
 * Plot the variational energy per iteration against the exact ground energy.
 * @param props - Component inputs.
 * @param props.history - Source energy after each iteration.
 * @param props.exact - Source exact ground energy, drawn as a reference line.
 * @returns A static labelled plot; the browser computes no energy.
 */
export function EnergyHistoryPlot({
  history,
  exact,
}: {
  readonly history: readonly number[];
  readonly exact: number;
}) {
  const low = Math.min(exact, ...history);
  const high = Math.max(exact, ...history);
  const y = (value: number) => PAD + (HEIGHT - 2 * PAD) - scale(value, low, high, HEIGHT - 2 * PAD);
  const x = (at: number) => PAD + scale(at, 0, Math.max(history.length - 1, 1), WIDTH - 2 * PAD);
  const points = history.map((value, at) => `${x(at)},${y(value)}`).join(" ");
  return (
    <svg
      role="img"
      aria-label="Variational energy per iteration against the exact ground energy"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width={WIDTH}
      height={HEIGHT}
    >
      <title>Variational energy per iteration against the exact ground energy</title>
      <line
        data-series="exact"
        x1={PAD}
        x2={WIDTH - PAD}
        y1={y(exact)}
        y2={y(exact)}
        stroke="currentColor"
        strokeDasharray="4 3"
      />
      <polyline data-series="variational" points={points} fill="none" stroke="currentColor" />
    </svg>
  );
}

/**
 * Plot one benchmark's proxy metrics, each labelled as a proxy.
 * @param props - Component inputs.
 * @param props.dataset - Packaged dataset identity.
 * @param props.metrics - Source proxy metrics.
 * @returns A static bar plot whose every label states that the value is a proxy.
 */
export function ProxyMetricsPlot({
  dataset,
  metrics,
}: {
  readonly dataset: string;
  readonly metrics: readonly NamedMetric[];
}) {
  const high = Math.max(1, ...metrics.map((metric) => Math.abs(metric.value)));
  const row = 22;
  return (
    <svg
      role="img"
      aria-label={`${dataset} proxy metrics, not physical outcomes`}
      viewBox={`0 0 ${WIDTH} ${row * metrics.length + PAD}`}
      width={WIDTH}
      height={row * metrics.length + PAD}
    >
      <title>{`${dataset} proxy metrics, not physical outcomes`}</title>
      {metrics.map((metric, at) => (
        <g key={metric.name} data-metric={metric.name}>
          <rect
            x={PAD}
            y={PAD + at * row}
            width={scale(Math.abs(metric.value), 0, high, WIDTH / 2)}
            height={row - 8}
            fill="currentColor"
            opacity={0.35}
          />
          <text x={PAD + 4} y={PAD + at * row + row - 12} fontSize={11}>
            {`${metric.name} (proxy): ${metric.value}`}
          </text>
        </g>
      ))}
    </svg>
  );
}
