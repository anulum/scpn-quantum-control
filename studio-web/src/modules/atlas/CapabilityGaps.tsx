// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — a capability's separate kinds of evidence and its gaps

import type { CapabilityEvidence, GapTest } from "./capabilityGaps";

function testState(test: GapTest): string {
  if (!test.present) return "missing";
  const behaviour =
    test.behaviour === null
      ? "behaviour not judged"
      : test.behaviour
        ? "defines tests"
        : "defines no test";
  const owner =
    test.ciState === "explicit"
      ? `run by ${test.ciOwners[0]}`
      : test.ciState === "duplicate"
        ? `run by ${test.ciOwners.length} jobs: ${test.ciOwners.join(", ")}`
        : "no CI job names it";
  return `${behaviour}; ${owner}`;
}

/**
 * Show one capability's tests, documentation, UI, coverage, support and science apart, with its gaps.
 *
 * Nothing is scored and nothing is combined: each kind of evidence keeps its own state, and
 * what cannot be known is said to be unknown.
 *
 * @param props - The capability's evidence from the committed report.
 * @returns The panel.
 */
export function CapabilityGaps({ evidence }: { readonly evidence: CapabilityEvidence }) {
  return (
    <section aria-label="Coverage and gaps">
      <h4>Coverage and gaps</h4>
      <dl aria-label="Kinds of evidence">
        <dt>Code coverage</dt>
        <dd>{evidence.codeCoverage}</dd>
        <dt>Capability support</dt>
        <dd>
          {`${evidence.supportReady ? "ready" : "not ready"}${evidence.supportBlockers.length > 0 ? `; blockers: ${evidence.supportBlockers.join("; ")}` : ""}`}
        </dd>
        <dt>Scientific validation</dt>
        <dd>
          {evidence.scientific.length === 0
            ? "unassessed"
            : evidence.scientific.map(([claim, status]) => `${claim}: ${status}`).join("; ")}
        </dd>
        <dt>User routes</dt>
        <dd>
          {evidence.ui.length === 0
            ? "unknown: no UI surface is declared"
            : evidence.ui
                .map((ui) => `${ui.path} (${ui.present ? "present" : "missing"})`)
                .join("; ")}
        </dd>
      </dl>
      <div className="qsp-data-scroll">
        <table aria-label="Declared tests">
          <thead>
            <tr>
              <th scope="col">Test</th>
              <th scope="col">State</th>
            </tr>
          </thead>
          <tbody>
            {evidence.tests.map((test) => (
              <tr key={test.path} data-ci={test.ciState}>
                <th scope="row">{test.path}</th>
                <td>{testState(test)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul aria-label="Declared documentation">
        {evidence.docs.map((doc) => (
          <li key={doc.path}>{`${doc.path} (${doc.present ? "present" : "missing"})`}</li>
        ))}
      </ul>
      {evidence.gaps.length === 0 ? (
        <p>No gap in the declared mappings.</p>
      ) : (
        <ul aria-label="Gaps">
          {evidence.gaps.map((gap) => (
            <li key={`${gap.kind} ${gap.surface}`} data-gap={gap.kind}>
              {`${gap.kind}: ${gap.surface}, declared in ${gap.declaredIn}. ${gap.detail}.`}
            </li>
          ))}
        </ul>
      )}
      <ul aria-label="Unknown">
        {evidence.unknowns.map((statement) => (
          <li key={statement}>{statement}</li>
        ))}
      </ul>
    </section>
  );
}
