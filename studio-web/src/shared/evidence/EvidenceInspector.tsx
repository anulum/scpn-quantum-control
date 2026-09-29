// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — evidence inspector

import { useUnitBoundRun } from "../../panel/useUnitBoundRun";
import type { EvidenceView } from "./evidence";

/** A result from a registered owning verifier, orthogonal to scientific status. */
export interface EvidenceVerification {
  /** Exact bounded verification outcome. */ readonly display: "match" | "mismatch" | "unverifiable" | "tampered";
  /** Owner-provided description of what was actually checked. */ readonly detail: string;
}
/** Inputs for one immutable evidence view and its optional offline verifier. */
export interface EvidenceInspectorProps {
  /** Source projection including complete snapshot identity. */ readonly view: EvidenceView;
  /** Current input revision, even when the evidence object itself is unchanged. */ readonly inputRevision: string;
  /** Trusted application callback; never taken from imported data. */ readonly verify?: () => Promise<EvidenceVerification>;
  /** Product action label for the owning verifier. */ readonly actionLabel?: string;
  /** Product label while the verifier is running. */ readonly runningLabel?: string;
  /** Visible fallback when the verifier rejects without an Error. */ readonly failureReason?: string;
}

/** Display independent evidence axes and bind asynchronous results to snapshot plus revision. */
export function EvidenceInspector({ view, inputRevision, verify, actionLabel = "Verify evidence", runningLabel = "Verifying…", failureReason = "verification failed" }: EvidenceInspectorProps) {
  const { state, run } = useUnitBoundRun<EvidenceVerification>(JSON.stringify([view, inputRevision]));
  const fields = [
    ["Schema", view.schema], ["Source", view.source], ["Source digest", view.digest],
    ["Scientific kind", view.kind], ["Claim status", view.claim], ["Admission", view.admission],
    ["Claim boundary", view.boundary], ["Freshness", view.freshness], ["Substrate", view.substrate],
    ["Numeric parity declarations", view.exactness], ["Provenance", view.provenance],
  ] as const;
  return (
    <section className="qsp-evidence-inspector" aria-label="Evidence inspector" data-verification-phase={state.phase}>
      <h4>Evidence details</h4>
      <dl>
        {fields.map(([label, value]) => <div key={label}><dt>{label}</dt><dd data-axis={label} data-negative={(label === "Scientific kind" || label === "Claim status") && (value === "falsified" || value === "refuted")}>{value ?? "Not supplied"}</dd></div>)}
        <div><dt>Seal</dt><dd>{view.seal === "missing" ? "Missing seal" : view.seal === "malformed" ? "Malformed seal" : "Seal present — not verified"}</dd></div>
      </dl>
      {view.issues.length > 0 && <div className="qsp-boundary"><p>Partial or unsupported evidence</p><ul aria-label="Evidence limitations">{view.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div>}
      <p className="qsp-meta">Custody and local verification do not change the producer’s scientific claim or freshness.</p>
      {verify ? <button type="button" disabled={!view.identifiable || state.phase === "running"}
        onClick={() => { void run(verify, failureReason); }}>{state.phase === "running" ? runningLabel : actionLabel}</button>
        : <p className="qsp-boundary">Verification is unavailable for this evidence format.</p>}
      {state.phase === "done" && <p role="status" data-verdict={state.verdict.display}
        className={`qsp-badge qsp-badge-${state.verdict.display === "match" ? "boundary" : "unverifiable"}`}>{state.verdict.detail}</p>}
      {state.phase === "error" && <p role="alert" className="qsp-badge qsp-badge-unverifiable">unverifiable — {state.reason}</p>}
    </section>
  );
}
