// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — accessible source simulation samples

import { useState } from "react";
import type { PhaseTrajectory } from "./phaseTrajectory";
import { LAB_MAX_OSCILLATORS } from "./phaseTrajectory";

const PAGE_ROWS = 20;

/** Display-only views of the existing admitted numerical arrays. */
export interface SimulationDataTableProps {
  /** Accessible caption identifying the source chart. */
  readonly label: string;
  /** Original kernel samples; no rounding, interpolation or recomputation. */
  readonly orderParameter: Float64Array;
  /** Original captured phases, in row-major order and radians, when available. */
  readonly phases?: Pick<PhaseTrajectory, "n" | "theta">;
}

/** Preserve signed zero when displaying a source binary64 value. */
function sampleText(value: number): string {
  return Object.is(value, -0) ? "-0" : String(value);
}

/** Read every source sample through bounded pages without changing the numerical arrays. */
export function SimulationDataTable({ label, orderParameter, phases }: SimulationDataTableProps) {
  const [offset, setOffset] = useState(0);
  if (orderParameter.length === 0) return <p role="status">No simulation samples are available.</p>;
  if (phases && (!Number.isSafeInteger(phases.n) || phases.n < 1 || phases.n > LAB_MAX_OSCILLATORS
    || phases.theta.length !== orderParameter.length * phases.n)) {
    return <p role="alert">Simulation table unavailable: inconsistent phase dimensions.</p>;
  }
  const lastOffset = Math.floor((orderParameter.length - 1) / PAGE_ROWS) * PAGE_ROWS;
  const first = Math.min(offset, lastOffset);
  const end = Math.min(first + PAGE_ROWS, orderParameter.length);
  const oscillators = Array.from({ length: phases?.n ?? 0 }, (_, index) => index);
  return <section className="qsp-simulation-table" aria-label={label + " samples"}>
    <div className="qsp-data-scroll" role="region" aria-label={label + " table scrolling"} tabIndex={0}>
      <table>
        <caption>{label}</caption>
        <thead><tr><th scope="col">Step</th><th scope="col">R (dimensionless)</th>
          {oscillators.map(index => <th scope="col" key={index}>{"θ" + String(index).replace(/[0-9]/g, digit => "₀₁₂₃₄₅₆₇₈₉"[Number(digit)]!) + " (rad)"}</th>)}
        </tr></thead>
        <tbody>{Array.from({ length: end - first }, (_, local) => {
          const step = first + local;
          return <tr key={step} data-step={step}>
            <th scope="row">{step}</th><td>{sampleText(orderParameter[step]!)}</td>
            {oscillators.map(index => <td key={index} data-oscillator={index}>{sampleText(phases!.theta[step * phases!.n + index]!)}</td>)}
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p role="status">Samples {first}–{end - 1} of {orderParameter.length}. Every source sample is available through these pages.</p>
    <div className="qsp-workspace-actions">
      <button type="button" disabled={first === 0} onClick={() => setOffset(first - PAGE_ROWS)}>Previous samples</button>
      <button type="button" disabled={end === orderParameter.length} onClick={() => setOffset(first + PAGE_ROWS)}>Next samples</button>
    </div>
  </section>;
}
