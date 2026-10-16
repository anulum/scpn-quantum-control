// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — complete retained-population comparison view

import captured from "../../../../data/studio/method_comparison_cases.json";
import { writeJson } from "../../shared/contracts/jsonTransport";
import { resultLimits } from "../../shared/results/resultModel";
import { methodMeaningFields } from "./methodComparisonArtifact";
import type { MethodComparisonArtifact, MethodInterval } from "./methodComparisonArtifact";
import { useMethodComparison } from "./useMethodComparison";

/** Format original lossless scalars; an unknown cost remains explicitly unavailable. */
function valueText(value: unknown): string {
  return value === null ? "unavailable" : typeof value === "string" ? value : writeJson(value);
}
/** Display the original estimator interval without deriving a new confidence claim. */
function intervalText(interval: MethodInterval | null): string {
  return interval === null
    ? "Not estimated"
    : `${interval.method}; nominal ${valueText(interval.coverage)}; [${valueText(interval.low)}, ${valueText(interval.high)}]; SE ${valueText(interval.standard_error)}; shots ${valueText(interval.n_shots)}; resamples ${valueText(interval.n_resamples)}`;
}
/** Download the exact admitted source bytes and release the browser object URL. */
function download(original: MethodComparisonArtifact): void {
  const url = URL.createObjectURL(
    new Blob([original.text], { type: "application/json;charset=utf-8" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "method-comparison.json";
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Inspect source-owned all-run statistics and intervals; selection changes only the displayed repetition. */
export function MethodComparison() {
  const controller = useMethodComparison(),
    admitted = controller.admitted;
  const comparison = admitted?.body.comparison;
  const rows =
    comparison?.runs.slice(
      controller.page * resultLimits.tableRows,
      (controller.page + 1) * resultLimits.tableRows,
    ) ?? [];
  const selected = comparison?.runs.find((run) => run.run_id === controller.selected);
  return (
    <section className="qsp-workspace" aria-label="Method comparison and uncertainty">
      <h4>Method comparison and uncertainty</h4>
      <p>
        Read a complete native comparison export. Every repetition, failure and negative result
        stays in its declared population. Selecting a repetition never replaces an all-run
        aggregate.
      </p>
      <label>
        Native comparison file
        <input
          aria-label="Native comparison file"
          type="file"
          accept="application/json,.json"
          disabled={controller.pending}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void controller.inspect(file);
            event.target.value = "";
          }}
        />
      </label>
      <label>
        Native method comparison JSON
        <textarea
          aria-label="Native method comparison JSON"
          value={controller.draft}
          disabled={controller.pending}
          spellCheck={false}
          onChange={(event) => controller.edit(event.target.value)}
        />
      </label>
      <button
        type="button"
        disabled={controller.pending || controller.draft === ""}
        onClick={() => {
          void controller.inspect(controller.draft);
        }}
      >
        Read native method comparison
      </button>
      <button
        type="button"
        disabled={controller.pending}
        onClick={() => {
          void controller.inspect(captured.cases.population);
        }}
      >
        Load captured population example
      </button>
      <button
        type="button"
        disabled={controller.pending}
        onClick={() => {
          void controller.inspect(captured.cases.intervals);
        }}
      >
        Load captured interval example
      </button>
      <button
        type="button"
        disabled={controller.pending}
        onClick={() => {
          void controller.inspect(captured.cases["original-solver"]);
        }}
      >
        Load captured original solver example
      </button>
      <p role="status" aria-label="Method comparison status">
        {controller.pending
          ? "Reading original comparison…"
          : admitted === null
            ? "No native comparison admitted."
            : "Native comparison admitted; showing the last admitted source."}
      </p>
      {controller.error !== null && (
        <p role="alert" aria-label="Method comparison import error">
          {controller.error}
        </p>
      )}
      {comparison !== undefined && admitted !== null && (
        <>
          <p>
            Reference: {comparison.baseline_method}. Missing-data policy:{" "}
            {comparison.missing_data_policy}.
          </p>
          <p>
            Imported statistics are source declarations. The content digest does not authenticate
            their producer or certify scientific validity.
          </p>
          <details>
            <summary>Original source identity and claim boundary</summary>
            <dl>
              <dt>Declared producer</dt>
              <dd>{admitted.body.producer}</dd>
              <dt>Content digest</dt>
              <dd>
                <output aria-label="Method comparison digest">
                  <code>{admitted.sha256}</code>
                </output>
              </dd>
            </dl>
            <p>{comparison.claim_boundary}</p>
          </details>
          <button type="button" onClick={() => download(admitted)}>
            Download original method comparison
          </button>
          {comparison.blockers.length > 0 ? (
            <div role="alert" aria-label="Native comparison refusal">
              <p>Source refuses ranking:</p>
              <ul>
                {comparison.blockers.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p>
              Source permits comparison of the declared populations and full known costs. No
              significance or advantage is inferred.
            </p>
          )}
          <table aria-label="All-run method summaries">
            <thead>
              <tr>
                <th>Method</th>
                <th>Retained</th>
                <th>Observed</th>
                <th>All-run mean</th>
                <th>Sample variance (n−1)</th>
                <th>Full cost (ms)</th>
                <th>Difference vs reference</th>
              </tr>
            </thead>
            <tbody>
              {comparison.methods.map((method) => (
                <tr key={method.method}>
                  <th>{method.method}</th>
                  <td>{valueText(method.population_size)}</td>
                  <td>{valueText(method.observed_size)}</td>
                  <td>{valueText(method.mean)}</td>
                  <td>{valueText(method.sample_variance)}</td>
                  <td>{valueText(method.total_cost_ms)}</td>
                  <td>{valueText(method.difference_vs_reference)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {admitted.sha256 === captured.interval_protocol.artifact_sha256 && (
            <section aria-label="Captured fixed-cohort interval evidence">
              <h5>Captured fixed-cohort check</h5>
              <p>{captured.interval_protocol.status}</p>
              <p>
                {captured.interval_protocol.description}. All 200 count samples and all 400
                estimator rows are retained in the captured source.
              </p>
              <table aria-label="Captured fixed-cohort interval check">
                <thead>
                  <tr>
                    <th>Estimator</th>
                    <th>Nominal confidence</th>
                    <th>Observed coverage</th>
                    <th>Covered / retained</th>
                    <th>Wilson lower</th>
                    <th>Wilson upper</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(captured.interval_protocol.methods).map(([method, evidence]) => (
                    <tr key={method}>
                      <th>{method}</th>
                      <td>{valueText(evidence.nominal_coverage)}</td>
                      <td>{valueText(evidence.observed_coverage)}</td>
                      <td>
                        {evidence.covered}/{evidence.replicates}
                      </td>
                      <td>{valueText(evidence.wilson_interval.lower)}</td>
                      <td>{valueText(evidence.wilson_interval.upper)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
          <label>
            Inspect retained repetition
            <select
              aria-label="Inspect retained repetition"
              value={controller.selected}
              onChange={(event) => controller.select(event.target.value)}
            >
              <option value="">All-run overview; no individual selection</option>
              {comparison.runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} · {run.status} · seed {valueText(run.seed)}
                </option>
              ))}
            </select>
          </label>
          <table aria-label="All retained repetitions">
            <thead>
              <tr>
                <th>Original run</th>
                <th>Method</th>
                <th>Seed</th>
                <th>Status</th>
                <th>Original scalar</th>
                <th>Original estimator interval</th>
                <th>Full cost (ms)</th>
                <th>Failure or missing reason</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((run) => (
                <tr
                  key={run.run_id}
                  aria-current={controller.selected === run.run_id ? "true" : undefined}
                >
                  <th>{run.run_id}</th>
                  <td>{run.method}</td>
                  <td>{valueText(run.seed)}</td>
                  <td>{run.status}</td>
                  <td>{valueText(run.value)}</td>
                  <td>{intervalText(run.interval)}</td>
                  <td>{valueText(run.budget.total_cost_ms)}</td>
                  <td>{run.reason ?? "none"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Showing retained repetitions {controller.page * resultLimits.tableRows + 1}–
            {Math.min((controller.page + 1) * resultLimits.tableRows, comparison.runs.length)} of{" "}
            {comparison.runs.length}
          </p>
          <button type="button" disabled={controller.page === 0} onClick={controller.previousPage}>
            Previous retained repetitions
          </button>
          <button
            type="button"
            disabled={(controller.page + 1) * resultLimits.tableRows >= comparison.runs.length}
            onClick={controller.nextPage}
          >
            Next retained repetitions
          </button>
          {selected !== undefined && (
            <section aria-label="Selected original repetition">
              <h5>{selected.run_id}</h5>
              <p>
                Backend: {selected.backend}; seed {valueText(selected.seed)}; {selected.status}.
              </p>
              <dl>
                {methodMeaningFields.map((field) => (
                  <div key={field}>
                    <dt>{field}</dt>
                    <dd>{valueText(selected.meaning[field])}</dd>
                  </div>
                ))}
              </dl>
              <table aria-label="Selected repetition resource budget">
                <thead>
                  <tr>
                    <th>Original resource</th>
                    <th>Source value</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["Compilation (ms)", selected.budget.compilation_ms],
                      ["Data movement (ms)", selected.budget.transfer_ms],
                      ["Execution (ms)", selected.budget.execution_ms],
                      ["Full cost (ms)", selected.budget.total_cost_ms],
                      ["Unallocated advisory wall time (ms)", selected.budget.observed_wall_ms],
                      ["Peak bytes", selected.budget.peak_bytes],
                      ["Memory ceiling bytes", selected.budget.memory_limit_bytes],
                      ["Wall-time ceiling (ms)", selected.budget.wall_limit_ms],
                      ["Within declared budget", selected.budget.within_budget],
                    ] as const
                  ).map(([label, value]) => (
                    <tr key={label}>
                      <th>{label}</th>
                      <td>{valueText(value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p>{intervalText(selected.interval)}</p>
            </section>
          )}
        </>
      )}
    </section>
  );
}
