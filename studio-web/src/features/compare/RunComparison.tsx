// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — read-only immutable revision and run comparison

import { useState } from "react";
import { writeJson } from "../../shared/contracts";
import type { RawCodec } from "../../shared/contracts";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { rawResultText, resultLimits } from "../results/resultModel";
import { comparisonKeyFields } from "./comparisonModel";
import { useRunComparison } from "./useRunComparison";
import type { ComparisonSideIndex } from "./useRunComparison";

/** Original currently selected workspace text and trusted host producer registry. */
export interface RunComparisonProps {
  /** Original source text for explicit local reading; never automatically saved. */ readonly sourceJson?: string;
  /** Original offline verifiers; an imported source cannot install executable verifiers. */ readonly rawCodecs?: ReadonlyMap<
    string,
    RawCodec
  >;
}

function semanticText(value: unknown): string {
  return value === undefined ? "(absent)" : writeJson(value);
}
function scalarText(value: number | null): string {
  return value === null ? "unavailable" : rawResultText(value);
}

/** Show semantic differences and baseline matching before any per-observation arithmetic. */
export function RunComparison({
  sourceJson = "",
  rawCodecs = localExperimentCodecs,
}: RunComparisonProps) {
  const controller = useRunComparison(sourceJson, rawCodecs),
    [page, setPage] = useState(0);
  const comparison = controller.comparison;
  const rowCount = comparison?.result.rows.length ?? 0;
  const pages = Math.max(1, Math.ceil(rowCount / resultLimits.tableRows));
  const currentPage = Math.min(page, pages - 1);
  const rows =
    comparison?.result.rows.slice(
      currentPage * resultLimits.tableRows,
      (currentPage + 1) * resultLimits.tableRows,
    ) ?? [];
  return (
    <section className="qsp-workspace" aria-label="Immutable revision and run comparison">
      <h4>Compare immutable revisions and runs</h4>
      <p>
        Read original workspace archives locally. A comparison never changes a saved revision,
        executes imported code or submits a job. Select a baseline and candidate explicitly.
      </p>
      {([0, 1] as const).map((index: ComparisonSideIndex) => {
        const side = controller.sides[index],
          name = index === 0 ? "Baseline" : "Candidate";
        return (
          <fieldset key={name} disabled={controller.busy}>
            <legend>{name} source</legend>
            <label>
              {name} archive file
              <input
                aria-label={`${name} archive file`}
                type="file"
                accept="application/json,.json"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void controller.read(index, file);
                  event.target.value = "";
                }}
              />
            </label>
            <label>
              {name} archive JSON
              <textarea
                aria-label={`${name} archive JSON`}
                value={side.json}
                spellCheck={false}
                onChange={(event) => controller.edit(index, event.target.value)}
              />
            </label>
            <button
              type="button"
              disabled={sourceJson === ""}
              onClick={() => controller.edit(index, sourceJson)}
            >
              Use current workspace as {name.toLowerCase()}
            </button>
            <button
              type="button"
              disabled={side.json === ""}
              onClick={() => {
                void controller.inspect(index);
              }}
            >
              Read {name.toLowerCase()} archive
            </button>
            {side.admitted !== null && (
              <>
                <p>
                  {name} admitted archive:{" "}
                  <code>{side.admitted.archive.preview.archiveDigest}</code>
                </p>
                <label>
                  {name} revision
                  <select
                    aria-label={`${name} revision`}
                    value={side.revisionHash}
                    onChange={(event) => controller.selectRevision(index, event.target.value)}
                  >
                    {side.admitted.revisions.map((revision) => (
                      <option key={revision.hash} value={revision.hash}>
                        {revision.hash}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {name} recorded run
                  <select
                    aria-label={`${name} recorded run`}
                    value={side.runHash ?? ""}
                    onChange={(event) =>
                      controller.selectRun(
                        index,
                        event.target.value === "" ? null : event.target.value,
                      )
                    }
                  >
                    <option value="">Revision metadata only</option>
                    {side.admitted.runs
                      .filter((run) => run.revisionHash === side.revisionHash)
                      .map((run) => (
                        <option key={run.hash} value={run.hash}>
                          {run.attemptId} · {run.state} · {run.hash}
                        </option>
                      ))}
                  </select>
                </label>
              </>
            )}
          </fieldset>
        );
      })}
      <button
        type="button"
        disabled={controller.busy || !controller.canCompare}
        onClick={() => {
          setPage(0);
          void controller.compare();
        }}
      >
        Compare selected immutable sources
      </button>
      <p role="status" aria-label="Comparison status">
        {controller.message}
      </p>
      {comparison !== null && (
        <>
          <dl aria-label="Compared immutable identities">
            <dt>Baseline revision</dt>
            <dd>{comparison.baseline.revisionHash}</dd>
            <dt>Baseline run</dt>
            <dd>{comparison.baseline.runHash ?? "No run selected"}</dd>
            <dt>Candidate revision</dt>
            <dd>{comparison.candidate.revisionHash}</dd>
            <dt>Candidate run</dt>
            <dd>{comparison.candidate.runHash ?? "No run selected"}</dd>
            <dt>Baseline source archive</dt>
            <dd>{comparison.baselineArchiveDigest}</dd>
            <dt>Candidate source archive</dt>
            <dd>{comparison.candidateArchiveDigest}</dd>
          </dl>
          <h5>Semantic differences</h5>
          {comparison.result.semanticDiff.length === 0 ? (
            <p>No original semantic fields differ.</p>
          ) : (
            <table aria-label="Original semantic differences">
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Baseline original</th>
                  <th>Candidate original</th>
                </tr>
              </thead>
              <tbody>
                {comparison.result.semanticDiff.map((difference) => (
                  <tr key={difference.field}>
                    <th>{difference.field}</th>
                    <td>
                      <pre>{semanticText(difference.baseline)}</pre>
                    </td>
                    <td>
                      <pre>{semanticText(difference.candidate)}</pre>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <h5>Comparable meanings</h5>
          <table aria-label="Declared comparison meanings">
            <thead>
              <tr>
                <th>Meaning</th>
                <th>Baseline</th>
                <th>Candidate</th>
              </tr>
            </thead>
            <tbody>
              {comparisonKeyFields.map((field) => (
                <tr key={field}>
                  <th>{field}</th>
                  <td>{comparison.baseline.key[field]}</td>
                  <td>{comparison.candidate.key[field]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h5>Baseline matching status</h5>
          <p>
            Exact object and requested-time alignment. No interpolation or nearest-time
            substitution. Matched: {comparison.result.matched}; baseline only:{" "}
            {comparison.result.baselineOnly}; candidate only: {comparison.result.candidateOnly}.
          </p>
          {comparison.result.blockers.length !== 0 ? (
            <div role="alert">
              <p>Numerical differences blocked:</p>
              <ul>
                {comparison.result.blockers.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p>
              All declared comparison meanings match. Differences are candidate minus baseline; no
              uncertainty, aggregate ranking or scientific equivalence is inferred.
            </p>
          )}
          <table aria-label="Original matched and unmatched values">
            <thead>
              <tr>
                <th>Object</th>
                <th>Requested time</th>
                <th>Status</th>
                <th>Baseline raw value</th>
                <th>Candidate raw value</th>
                <th>Candidate − baseline</th>
                <th>Difference state</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={JSON.stringify([row.key, rawResultText(row.time), row.status])}>
                  <th>{row.key}</th>
                  <td>{rawResultText(row.time)}</td>
                  <td>{row.status}</td>
                  <td>{scalarText(row.baseline)}</td>
                  <td>{scalarText(row.candidate)}</td>
                  <td>{scalarText(row.delta)}</td>
                  <td>{row.deltaState}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Original observations {rowCount === 0 ? 0 : currentPage * resultLimits.tableRows + 1}–
            {Math.min((currentPage + 1) * resultLimits.tableRows, rowCount)} of {rowCount}; every
            unmatched observation is retained.
          </p>
          <button
            type="button"
            disabled={currentPage === 0}
            onClick={() => setPage(currentPage - 1)}
          >
            Previous comparison values
          </button>
          <button
            type="button"
            disabled={currentPage + 1 >= pages}
            onClick={() => setPage(currentPage + 1)}
          >
            Next comparison values
          </button>
        </>
      )}
    </section>
  );
}
