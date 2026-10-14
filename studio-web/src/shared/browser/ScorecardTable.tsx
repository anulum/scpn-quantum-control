// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web baseline-scorecard view

import type { ScorecardView } from "./data";
import { displayClassLabel, presentScorecardStatus } from "./honesty";

/**
 * The differentiable baseline scorecard, category statuses verbatim.
 * `behind_baseline` renders at the boundary — explicit hardening work,
 * never failure noise and never a green badge.
 */
export function ScorecardTable({ scorecard }: { scorecard: ScorecardView }) {
  return (
    <section className="qsp-scorecard">
      <h3>Baseline scorecard</h3>
      <p className="qsp-meta">
        <code>{scorecard.artifactId}</code> · {scorecard.rows.length} categories
      </p>
      <table>
        <thead>
          <tr>
            <th scope="col">Category</th>
            <th scope="col">Status</th>
            <th scope="col">Blockers</th>
          </tr>
        </thead>
        <tbody>
          {scorecard.rows.map((row) => {
            const displayClass = presentScorecardStatus(row.status);
            return (
              <tr key={row.category} className={`qsp-row-${displayClass}`}>
                <td>
                  <code>{row.category}</code>
                </td>
                <td>
                  <span className={`qsp-badge qsp-badge-${displayClass}`}>
                    {row.status} · {displayClassLabel(displayClass)}
                  </span>
                </td>
                <td>{row.blockers.join("; ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="qsp-boundary">{scorecard.claimBoundary}</p>
    </section>
  );
}
