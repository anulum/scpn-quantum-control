// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — claim comparisons with a filter that never hides counter-evidence

import { useEffect, useState } from "react";
import {
  type ClaimComparison,
  type ClaimComparisons as ComparisonRecord,
  type ClaimComparisonsAdmission,
  type ComparisonStatus,
  admitCommittedClaimComparisons,
  comparisonStatuses,
} from "./claimComparisons";

function Rows({
  label,
  items,
  sources,
}: {
  readonly label: string;
  readonly items: readonly ClaimComparison[];
  readonly sources: ComparisonRecord["sources"];
}) {
  return (
    <div className="qsp-data-scroll">
      <table aria-label={label}>
        <thead>
          <tr>
            <th scope="col">Values</th>
            <th scope="col">Criterion</th>
            <th scope="col">Verdict</th>
            <th scope="col">Why</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={`${item.left} ${item.right} ${item.criterion}`} data-status={item.status}>
              <th scope="row">
                {`${item.left} (${sources.get(item.left) ?? "unknown source"}) · ${item.right} (${sources.get(item.right) ?? "unknown source"})${item.negative ? " · counter-evidence" : ""}${item.federated ? " · federated" : ""}`}
              </th>
              <td>{item.criterion}</td>
              <td>{item.status}</td>
              <td>{item.reasons.join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Comparisons({ record }: { readonly record: ComparisonRecord }) {
  const [shown, setShown] = useState<ReadonlySet<ComparisonStatus>>(new Set(comparisonStatuses));
  const selected = record.comparisons.filter((item) => shown.has(item.status));
  // A filter never hides counter-evidence: a negative value or a disagreement stays listed.
  const counter = record.comparisons.filter(
    (item) => !shown.has(item.status) && (item.negative || item.status === "disagreement"),
  );
  return (
    <>
      <p>
        {`Local producer ${record.home}. Accepted producer equivalence: ${record.accepted.length === 0 ? "none, so comparisons across producers are disabled" : record.accepted.join(", ")}. Record ${record.recordDigest.slice(0, 12)} verified.`}
      </p>
      <fieldset>
        <legend>Show comparisons</legend>
        {comparisonStatuses.map((status) => (
          <label key={status}>
            <input
              type="checkbox"
              checked={shown.has(status)}
              onChange={(event) => {
                const next = new Set(shown);
                if (event.target.checked) next.add(status);
                else next.delete(status);
                setShown(next);
              }}
            />
            {status}
          </label>
        ))}
      </fieldset>
      <Rows label="Selected comparisons" items={selected} sources={record.sources} />
      {counter.length > 0 ? (
        <Rows label="Counter-evidence kept visible" items={counter} sources={record.sources} />
      ) : null}
    </>
  );
}

/**
 * Show claim comparisons judged only under compatible assumptions.
 *
 * Nothing is promoted: each verdict comes from the sealed record, and the status
 * filter never hides a comparison that involves counter-evidence or disagrees.
 *
 * @param props - The admission to show; by default the record committed with this build.
 * @returns The panel.
 */
export function ClaimComparisons({
  admit = admitCommittedClaimComparisons,
}: {
  readonly admit?: () => Promise<ClaimComparisonsAdmission>;
}) {
  const [admission, setAdmission] = useState<ClaimComparisonsAdmission | null>(null);
  useEffect(() => {
    let live = true;
    void admit().then((result) => {
      if (live) setAdmission(result);
    });
    return () => {
      live = false;
    };
  }, [admit]);
  return (
    <section aria-label="Claim comparisons">
      <h4>Claim comparisons</h4>
      {admission === null ? (
        <p role="status">Verifying the claim comparison record…</p>
      ) : admission.ok ? (
        <Comparisons record={admission.record} />
      ) : (
        <p role="alert">{admission.reason}</p>
      )}
    </section>
  );
}
