// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web transform-algebra support-matrix grid

import type { SupportMatrixView } from "./data";
import { displayClassLabel, presentSupportStatus } from "./honesty";
import { ALL_LANES, usePanelStore } from "./store";

function residualCell(residual: number | null): string {
  return residual === null ? "n/a" : residual.toExponential(3);
}

/**
 * The transform-algebra support matrix, statuses verbatim. Blocked rows are
 * first-class fail-closed boundaries, not failures; the lane filter narrows
 * the grid without ever hiding that a boundary exists (the counts stay
 * visible in the header).
 */
export function SupportMatrixGrid({ matrix }: { matrix: SupportMatrixView }) {
  const laneFilter = usePanelStore((state) => state.laneFilter);
  const setLaneFilter = usePanelStore((state) => state.setLaneFilter);
  const lanes = [ALL_LANES, ...new Set(matrix.rows.map((row) => row.lane))];
  const visible =
    laneFilter === ALL_LANES
      ? matrix.rows
      : matrix.rows.filter((row) => row.lane === laneFilter);
  const blockedCount = matrix.rows.filter((row) => row.status === "blocked").length;
  return (
    <section className="qsp-support-matrix">
      <h3>Transform-algebra support matrix</h3>
      <p className="qsp-meta">
        <code>{matrix.artifactId}</code> · {matrix.rows.length} rows ·{" "}
        {matrix.rows.length - blockedCount} supported · {blockedCount} fail-closed boundaries
      </p>
      <label className="qsp-lane-filter">
        Lane{" "}
        <select value={laneFilter} onChange={(event) => setLaneFilter(event.target.value)}>
          {lanes.map((lane) => (
            <option key={lane} value={lane}>
              {lane}
            </option>
          ))}
        </select>
      </label>
      <table>
        <thead>
          <tr>
            <th scope="col">Row</th>
            <th scope="col">Lane</th>
            <th scope="col">Transform stack</th>
            <th scope="col">Status</th>
            <th scope="col">Residual</th>
            <th scope="col">Boundary / notes</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => {
            const displayClass = presentSupportStatus(row.status);
            return (
              <tr key={row.rowId} className={`qsp-row-${displayClass}`}>
                <td>
                  <code>{row.rowId}</code>
                </td>
                <td>{row.lane}</td>
                <td>
                  <code>{row.transformStack.join(" ∘ ")}</code>
                </td>
                <td>
                  <span className={`qsp-badge qsp-badge-${displayClass}`}>
                    {row.status} · {displayClassLabel(displayClass)}
                  </span>
                </td>
                <td>{residualCell(row.residual)}</td>
                <td>{[...row.blockedReasons, ...row.notes].join("; ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="qsp-boundary">{matrix.claimBoundary}</p>
    </section>
  );
}
