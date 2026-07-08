// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web gradient-plan explanation view

import { useMemo, useState } from "react";

import type {
  GradientPlanExplanationRowView,
  GradientPlanExplanationView,
} from "./data";
import { displayClassLabel, presentSupportStatus } from "./honesty";

function uniqueSorted(values: readonly string[]): readonly string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function firstRow(rows: readonly GradientPlanExplanationRowView[]): GradientPlanExplanationRowView {
  const row = rows[0];
  if (row === undefined) {
    throw new Error("gradient-plan artefact has no rows");
  }
  return row;
}

/**
 * Planner explanation view for the committed gradient-support audit.
 *
 * The selected cell renders the method the Python planner chose and the reasons
 * or fail-closed boundaries it emitted. This component does not execute a
 * differentiate run; browser execution remains owned by the bounded WASM tier.
 */
export function GradientPlanExplanation({
  plans,
}: {
  plans: GradientPlanExplanationView;
}) {
  const frameworks = useMemo(
    () => ["all", ...uniqueSorted(plans.rows.map((row) => row.framework))],
    [plans.rows],
  );
  const [framework, setFramework] = useState("all");
  const visible = useMemo(
    () =>
      framework === "all"
        ? plans.rows
        : plans.rows.filter((row) => row.framework === framework),
    [framework, plans.rows],
  );
  const [selectedCell, setSelectedCell] = useState(firstRow(plans.rows).cellId);
  const selected =
    visible.find((row) => row.cellId === selectedCell) ?? firstRow(visible);
  const selectedClass = presentSupportStatus(selected.supported ? "passed" : "blocked");
  const selectedReasons = selected.why.length > 0 ? selected.why : selected.failClosedBoundaries;
  return (
    <section className="qsp-gradient-plan">
      <h3>Gradient-plan explanation</h3>
      <p className="qsp-meta">
        <code>{plans.artifactId}</code> · {plans.rows.length} cells ·{" "}
        {plans.methodFamilies.join(", ")}
      </p>
      <div className="qsp-filter-grid">
        <label>
          Planner framework
          <select
            value={framework}
            onChange={(event) => {
              const nextFramework = event.target.value;
              setFramework(nextFramework);
              const nextRows =
                nextFramework === "all"
                  ? plans.rows
                  : plans.rows.filter((row) => row.framework === nextFramework);
              setSelectedCell(firstRow(nextRows).cellId);
            }}
          >
            {frameworks.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          Planner cell
          <select
            value={selected.cellId}
            onChange={(event) => setSelectedCell(event.target.value)}
          >
            {visible.map((row) => (
              <option key={row.cellId} value={row.cellId}>
                {row.operation} / {row.framework}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="qsp-plan-grid">
        <div>
          <span className="qsp-row-detail">Selected method</span>
          <code>{selected.selectedMethod}</code>
        </div>
        <div>
          <span className="qsp-row-detail">Method family</span>
          <code>{selected.methodFamily}</code>
        </div>
        <div>
          <span className="qsp-row-detail">Backend</span>
          <code>{selected.backend}</code>
        </div>
        <div>
          <span className="qsp-row-detail">Mode</span>
          <code>{selected.evaluationMode}</code>
        </div>
        <div>
          <span className="qsp-row-detail">Evaluations</span>
          <code>{selected.backendEvaluations}</code>
        </div>
        <div>
          <span className="qsp-row-detail">Status</span>
          <span className={`qsp-badge qsp-badge-${selectedClass}`}>
            {selected.status} · {displayClassLabel(selectedClass)}
          </span>
        </div>
      </div>
      <div className="qsp-plan-detail">
        <h4>Why</h4>
        <ul>
          {selectedReasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
        {selected.warnings.length > 0 ? (
          <>
            <h4>Warnings</h4>
            <ul>
              {selected.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </>
        ) : null}
        {selected.alternatives.length > 0 ? (
          <>
            <h4>Alternatives</h4>
            <p>{selected.alternatives.join(", ")}</p>
          </>
        ) : null}
      </div>
      <p className="qsp-boundary">{selected.claimBoundary}</p>
      <p className="qsp-boundary">{plans.claimBoundary}</p>
    </section>
  );
}
