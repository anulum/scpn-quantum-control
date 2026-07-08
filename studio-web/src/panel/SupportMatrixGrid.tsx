// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web transform-algebra support-matrix grid

import { useMemo, useState } from "react";

import type { SupportMatrixView } from "./data";
import { displayClassLabel, presentSupportStatus } from "./honesty";
import {
  SUPPORT_MATRIX_ALL,
  buildSupportMatrixExplorer,
  filterSupportMatrixRows,
  type SupportMatrixFilters,
} from "./supportMatrixExplorer";

function residualCell(residual: number | null): string {
  return residual === null ? "n/a" : residual.toExponential(3);
}

/**
 * The transform-algebra support-matrix cockpit, statuses verbatim. Blocked rows
 * are first-class fail-closed boundaries, not failures; filters narrow the grid
 * without mutating the artifact counts or upgrading any row.
 */
export function SupportMatrixGrid({ matrix }: { matrix: SupportMatrixView }) {
  const explorer = useMemo(() => buildSupportMatrixExplorer(matrix), [matrix]);
  const [filters, setFilters] = useState<SupportMatrixFilters>({
    operationQuery: "",
    framework: SUPPORT_MATRIX_ALL,
    backend: SUPPORT_MATRIX_ALL,
    exactness: SUPPORT_MATRIX_ALL,
    claimStatus: SUPPORT_MATRIX_ALL,
  });
  const visible = useMemo(
    () => filterSupportMatrixRows(explorer.rows, filters),
    [explorer.rows, filters],
  );
  const setFilter = (key: keyof SupportMatrixFilters, value: string): void => {
    setFilters((current) => ({ ...current, [key]: value }));
  };
  return (
    <section className="qsp-support-matrix">
      <h3>Differentiate support explorer</h3>
      <p className="qsp-meta">
        <code>{explorer.artifactId}</code> · {explorer.rows.length} rows ·{" "}
        {explorer.supportedCount} supported · {explorer.failClosedCount} fail-closed boundaries
      </p>
      <div className="qsp-support-summary" aria-label="support matrix summary">
        <span>frameworks {explorer.frameworks.length}</span>
        <span>backends {explorer.backends.length}</span>
        <span>exactness levels {explorer.exactnessLevels.length}</span>
        <span>claim states {explorer.claimStatuses.length}</span>
      </div>
      <div className="qsp-filter-grid">
        <label>
          Operation
          <input
            value={filters.operationQuery}
            onChange={(event) => setFilter("operationQuery", event.target.value)}
          />
        </label>
        <label>
          Framework
          <select
            value={filters.framework}
            onChange={(event) => setFilter("framework", event.target.value)}
          >
            <option value={SUPPORT_MATRIX_ALL}>all</option>
            {explorer.frameworks.map((framework) => (
              <option key={framework} value={framework}>
                {framework}
              </option>
            ))}
          </select>
        </label>
        <label>
          Backend
          <select
            value={filters.backend}
            onChange={(event) => setFilter("backend", event.target.value)}
          >
            <option value={SUPPORT_MATRIX_ALL}>all</option>
            {explorer.backends.map((backend) => (
              <option key={backend} value={backend}>
                {backend}
              </option>
            ))}
          </select>
        </label>
        <label>
          Exactness
          <select
            value={filters.exactness}
            onChange={(event) => setFilter("exactness", event.target.value)}
          >
            <option value={SUPPORT_MATRIX_ALL}>all</option>
            {explorer.exactnessLevels.map((exactness) => (
              <option key={exactness} value={exactness}>
                {exactness}
              </option>
            ))}
          </select>
        </label>
        <label>
          Claim status
          <select
            value={filters.claimStatus}
            onChange={(event) => setFilter("claimStatus", event.target.value)}
          >
            <option value={SUPPORT_MATRIX_ALL}>all</option>
            {explorer.claimStatuses.map((claimStatus) => (
              <option key={claimStatus} value={claimStatus}>
                {claimStatus}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="qsp-table-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Operation</th>
              <th scope="col">Framework</th>
              <th scope="col">Backend</th>
              <th scope="col">Exactness</th>
              <th scope="col">Claim status</th>
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
                    <span className="qsp-row-detail">{row.operation}</span>
                  </td>
                  <td>{row.framework}</td>
                  <td>{row.backend}</td>
                  <td>{row.exactness}</td>
                  <td>
                    <span className={`qsp-badge qsp-badge-${displayClass}`}>
                      {row.status} · {row.claimStatus} · {displayClassLabel(displayClass)}
                    </span>
                  </td>
                  <td>
                    {residualCell(row.residual)}
                    <span className="qsp-row-detail">tol {row.tolerance.toExponential(1)}</span>
                  </td>
                  <td>{[...row.blockedReasons, ...row.notes].join("; ") || "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {visible.length === 0 ? (
        <p className="qsp-boundary">No committed support row matches these filters.</p>
      ) : null}
      <p className="qsp-boundary">{explorer.claimBoundary}</p>
    </section>
  );
}
