// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence lineage of the selected Atlas node

import { useUnitBoundRun } from "../../shared/browser/useUnitBoundRun";
import { projectEvidenceBundle } from "../../shared/evidence/evidence";
import {
  EvidenceInspector,
  type EvidenceVerification,
} from "../../shared/evidence/EvidenceInspector";
import type {
  LineageClaim,
  LineagePack,
  LineageRecord,
  LineageReplication,
} from "./evidenceLineage";

/** Verifies one record; the default recomputes its canonical digest. */
export type LineageVerifier = (record: LineageRecord) => Promise<EvidenceVerification>;

function Evidence({ claim }: { readonly claim: LineageClaim }) {
  return (
    <table aria-label="Declared evidence">
      <thead>
        <tr>
          <th scope="col">Identifier</th>
          <th scope="col">Role</th>
          <th scope="col">Custody</th>
        </tr>
      </thead>
      <tbody>
        {claim.evidence.map((item) => (
          <tr key={`${item.role} ${item.artifactId}`} data-state={item.state}>
            <th scope="row">{item.artifactId}</th>
            <td>{item.role}</td>
            <td>
              {item.state === "missing" ? (
                "Missing custody: no committed file carries this identifier"
              ) : (
                <>
                  {item.state === "ambiguous"
                    ? `Ambiguous: ${item.candidates.length} committed files carry this identifier`
                    : "Resolved"}
                  <ul>
                    {item.candidates.map((candidate) => (
                      <li key={candidate.path}>
                        <code>{candidate.path}</code> sha256 <code>{candidate.sha256}</code>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ClaimLineage({
  claim,
  verify,
}: {
  readonly claim: LineageClaim;
  readonly verify: LineageVerifier;
}) {
  return (
    <>
      <dl aria-label="Claim statuses">
        <dt>Ledger promotion status</dt>
        <dd>{claim.promotionStatus}</dd>
        <dt>Reference-validation certificate</dt>
        <dd>
          {claim.absentReason === null
            ? `Certified: ${JSON.stringify(claim.certificate)}`
            : `Absent: ${claim.absentReason}`}
        </dd>
      </dl>
      <EvidenceInspector
        view={projectEvidenceBundle(claim.bundle)}
        inputRevision={claim.recordDigest}
        verify={() => verify(claim)}
        actionLabel="Verify lineage record"
      />
      <Evidence claim={claim} />
    </>
  );
}

function PackLineage({
  pack,
  verify,
}: {
  readonly pack: LineagePack;
  readonly verify: LineageVerifier;
}) {
  return (
    <>
      <dl aria-label="Pack statuses">
        <dt>Manifest status</dt>
        <dd>{pack.manifestStatus}</dd>
        <dt>Scope</dt>
        <dd>{pack.claimScope}</dd>
        <dt>Does not claim</dt>
        <dd>{pack.nonClaims.join("; ")}</dd>
        <dt>Provider jobs</dt>
        <dd>{pack.jobIds.join(", ")}</dd>
      </dl>
      <EvidenceInspector
        view={projectEvidenceBundle(pack.bundle)}
        inputRevision={pack.recordDigest}
        verify={() => verify(pack)}
        actionLabel="Verify lineage record"
      />
      <table aria-label="Pack artefacts">
        <thead>
          <tr>
            <th scope="col">File</th>
            <th scope="col">Role</th>
            <th scope="col">Recorded sha256</th>
            <th scope="col">Custody</th>
          </tr>
        </thead>
        <tbody>
          {pack.artefacts.map((artefact) => (
            <tr key={artefact.path}>
              <th scope="row">
                <code>{artefact.path}</code>
              </th>
              <td>{artefact.role}</td>
              <td>
                <code>{artefact.recordedSha256}</code>
              </td>
              <td>{artefact.custody}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function ReplicationLineage({
  replication,
  verify,
}: {
  readonly replication: LineageReplication;
  readonly verify: LineageVerifier;
}) {
  const { state, run } = useUnitBoundRun<EvidenceVerification>(replication.recordDigest);
  return (
    <>
      <dl aria-label="Replication decision">
        <dt>Campaign</dt>
        <dd>{replication.campaign}</dd>
        <dt>{replication.decisionField}</dt>
        <dd data-negative={!replication.decision}>{String(replication.decision)}</dd>
        <dt>Analysis</dt>
        <dd>
          <code>{replication.path}</code> sha256 <code>{replication.sha256}</code>
        </dd>
      </dl>
      <button
        type="button"
        disabled={state.phase === "running"}
        onClick={() => {
          void run(() => verify(replication), "verification failed");
        }}
      >
        Verify lineage record
      </button>
      {state.phase === "done" ? (
        <p role="status" data-verdict={state.verdict.display}>
          {state.verdict.detail}
        </p>
      ) : null}
    </>
  );
}

/**
 * Show the evidence lineage of the selected node, every source-owned status verbatim.
 *
 * @param props - The node's lineage record, the verifier and the route's context suffix.
 * @returns The statuses, the declared evidence or artefacts, and links to the evidence instruments.
 */
export function EvidenceLineage({
  record,
  verify,
  contextSuffix,
}: {
  readonly record: LineageRecord;
  readonly verify: LineageVerifier;
  readonly contextSuffix: string;
}) {
  return (
    <section aria-label="Evidence lineage">
      <h4>Evidence lineage</h4>
      <p>
        Each status below is its source owner&apos;s, shown as written. None is merged, raised or
        relabelled here.
      </p>
      {record.kind === "claim" ? <ClaimLineage claim={record} verify={verify} /> : null}
      {record.kind === "pack" ? <PackLineage pack={record} verify={verify} /> : null}
      {record.kind === "replication" ? (
        <ReplicationLineage replication={record} verify={verify} />
      ) : null}
      <nav aria-label="Evidence instruments">
        <a href={`#/evidence${contextSuffix}`}>Open the evidence catalogue</a>{" "}
        <a href={`#/evidence/attestation${contextSuffix}`}>Verify an attestation</a>
      </nav>
    </section>
  );
}
