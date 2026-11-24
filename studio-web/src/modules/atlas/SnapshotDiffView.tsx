// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — an admitted snapshot comparison and its potential impact

import { useRef, useState } from "react";
import { ScrollRegion } from "../../shared/browser/ScrollRegion";
import { type SnapshotDiff, type SnapshotDiffAdmission, admitSnapshotDiff } from "./snapshotDiff";

function short(digest: string): string {
  return digest.slice(0, 12);
}

function Comparison({ diff, open }: { readonly diff: SnapshotDiff; readonly open: string | null }) {
  const counts = new Map<string, number>();
  for (const node of diff.nodes) counts.set(node.change, (counts.get(node.change) ?? 0) + 1);
  return (
    <>
      <p>
        {`Snapshot ${short(diff.before.snapshotDigest)} (commit ${short(diff.before.sourceCommit)}) compared with ${short(diff.after.snapshotDigest)} (commit ${short(diff.after.sourceCommit)}). Record ${short(diff.recordDigest)} verified.`}
      </p>
      <p>
        {diff.after.snapshotDigest === open
          ? "The later snapshot is the one open here."
          : "The later snapshot is not the one open here."}
      </p>
      <p role="status" aria-label="Comparison summary">
        {`${["added", "removed", "tombstoned", "changed"].map((change) => `${counts.get(change) ?? 0} ${change}`).join(", ")}; ${diff.unchanged} unchanged; ${diff.edges.length} relations differ; ${diff.impact.length} nodes potentially impacted.`}
      </p>
      <p role="note" data-complete={diff.complete}>
        {diff.complete
          ? "The impact follows only observed relations of complete extractions. It is potential impact, not a proof."
          : `The impact analysis is incomplete, so it is not exhaustive: ${diff.incompleteReasons.join("; ")}.`}
      </p>
      <ScrollRegion
        className="qsp-data-scroll"
        name="Changed objects table scrolling"
        action="Scroll Changed objects"
      >
        <table aria-label="Changed objects">
          <thead>
            <tr>
              <th scope="col">Object</th>
              <th scope="col">Change</th>
              <th scope="col">Category</th>
              <th scope="col">Detail</th>
            </tr>
          </thead>
          <tbody>
            {diff.nodes.map((node) => (
              <tr key={node.id}>
                <th scope="row">{node.id}</th>
                <td>{node.change}</td>
                <td>{node.category}</td>
                <td>{node.reason ?? node.fields.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <ScrollRegion
        className="qsp-data-scroll"
        name="Changed relations table scrolling"
        action="Scroll Changed relations"
      >
        <table aria-label="Changed relations">
          <thead>
            <tr>
              <th scope="col">From</th>
              <th scope="col">Relation</th>
              <th scope="col">To</th>
              <th scope="col">Change</th>
            </tr>
          </thead>
          <tbody>
            {diff.edges.map((edge) => (
              <tr key={`${edge.source} ${edge.kind} ${edge.target}`}>
                <td>{edge.source}</td>
                <td>{edge.kind}</td>
                <td>{edge.target}</td>
                <td>{edge.change}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <ScrollRegion
        className="qsp-data-scroll"
        name="Potential impact table scrolling"
        action="Scroll Potential impact"
      >
        <table aria-label="Potential impact">
          <thead>
            <tr>
              <th scope="col">Object</th>
              <th scope="col">From the change of</th>
              <th scope="col">Path</th>
            </tr>
          </thead>
          <tbody>
            {diff.impact.map((item) => (
              <tr key={item.id}>
                <th scope="row">{item.id}</th>
                <td>{item.origin}</td>
                <td>
                  {item.path
                    .map((step) => `${step.source} ${step.kind} ${step.target}`)
                    .join(" → ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
    </>
  );
}

/**
 * Open a snapshot comparison written by `tools/studio_atlas_snapshot_diff.py` and show it.
 *
 * The record is verified by its canonical digest and shown as its source wrote it; the
 * browser does not recompute the comparison.
 *
 * @param props - The digest of the public snapshot open in the Atlas, or null.
 * @returns The chooser and, once a record is admitted, its changes and potential impact.
 */
export function SnapshotDiffView({ open }: { readonly open: string | null }) {
  const [admission, setAdmission] = useState<SnapshotDiffAdmission | null>(null);
  const latest = useRef(0);
  const choose = (file: File): void => {
    const sequence = latest.current + 1;
    latest.current = sequence;
    void file
      .text()
      .then(admitSnapshotDiff, () => ({
        ok: false as const,
        reason: "The snapshot comparison could not be read.",
      }))
      .then((result) => {
        if (sequence === latest.current) setAdmission(result);
      });
  };
  return (
    <section aria-label="Snapshot comparison">
      <h4>Snapshot comparison</h4>
      <label>
        Open a snapshot comparison
        <input
          type="file"
          accept=".json,application/json"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file !== undefined) choose(file);
            event.target.value = "";
          }}
        />
      </label>
      {admission === null ? null : admission.ok ? (
        <Comparison diff={admission.diff} open={open} />
      ) : (
        <p role="alert">{admission.reason}</p>
      )}
    </section>
  );
}
