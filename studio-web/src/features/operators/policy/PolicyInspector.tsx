// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned operator policy inspection

import { useEffect, useRef, useState } from "react";
import example from "../../../../../data/studio/operator_policy_decisions.json?raw";
import { writeJson } from "../../../shared/contracts/jsonTransport";
import { parseOperatorDecision } from "./policyDecision";
import type { OperatorDecisionSnapshot } from "./policyDecision";

/** Display the source's dated verdict and original values without rerunning policy. */
function DecisionDetails({ snapshot }: {
  /** Exact admitted source, independent of the current JSON draft. */
  readonly snapshot: OperatorDecisionSnapshot;
}) {
  const { decision, settings } = snapshot;
  const requested = settings.body["requested"] as Readonly<Record<string, unknown>>;
  const effective = settings.body["effective"] as Readonly<Record<string, unknown>>;
  const origins = settings.body["origins"] as Readonly<Record<string, unknown>>;
  return <section aria-label="Admitted operator decision">
    <p aria-label="Core policy verdict">{decision.allowed ? "allowed plan" : "refused plan"}</p>
    <p>Assessment time: {decision.assessed_at}. This is a dated source verdict; HAL rechecks current policy and the exact workload at submit.</p>
    <p aria-label="Policy refusal reasons">{decision.reasons.length === 0 ? "none" : decision.reasons.join(", ")}</p>
    <p aria-label="Rejected substitutions">{decision.rejected_substitutions.length === 0 ? "none" : decision.rejected_substitutions.join(", ")}</p>
    <p>Envelope SHA-256: <code>{snapshot.sha256}</code></p>
    <p>Original settings SHA-256: <code>{snapshot.settingsSha256}</code></p>
    <table><caption>Operator requested and effective settings</caption>
      <thead><tr><th scope="col">Field</th><th scope="col">Requested</th><th scope="col">Effective</th><th scope="col">Origin</th></tr></thead>
      <tbody>{Object.keys(requested).sort().map(key => <tr key={key}><th scope="row">{key}</th>
        <td>{writeJson(requested[key])}</td><td>{writeJson(effective[key])}</td><td>{writeJson(origins[key])}</td>
      </tr>)}</tbody>
    </table>
    <p>Policy source: <code>{writeJson(settings.body["policy_ref"])}</code></p>
    <p>Environment source: <code>{writeJson(settings.body["environment_ref"])}</code></p>
    <dl aria-label="Bound operator request">{Object.entries(decision.request).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{writeJson(value)}</dd></div>)}</dl>
    <dl aria-label="Policy limits">{Object.entries(decision.policy).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{writeJson(value)}</dd></div>)}</dl>
    <p>Concurrency and time ceilings apply to the declared plan. They do not certify a global job quota or predict elapsed execution time.</p>
    <section aria-label="Dated pricing inputs"><h4>Supplied price estimate</h4>
      <p aria-label="Estimated cost">{decision.estimate === null || decision.estimate["amount"] === null ? "unknown" : String(decision.estimate["amount"])}</p>
      {decision.estimate !== null && <dl>{Object.entries(decision.estimate).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{writeJson(value)}</dd></div>)}</dl>}
      <p>Unknown pricing cannot authorise unattended external execution. Supplied synthetic examples are conformance inputs, not live prices or account charges.</p>
    </section>
  </section>;
}

/** Inspect and verbatim-export native policy metadata without submission or storage writes. */
export function PolicyInspector() {
  const [draft, setDraft] = useState(""), [snapshot, setSnapshot] = useState<OperatorDecisionSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null), [pending, setPending] = useState(false);
  const ticket = useRef(0), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; ticket.current++; }; }, []);
  function change(value: string) { ticket.current++; setDraft(value); setError(null); setPending(false); }
  async function inspect(raw: string) {
    const current = ++ticket.current; setPending(true); setError(null);
    const result = await parseOperatorDecision(raw);
    if (!mounted.current || current !== ticket.current) return;
    setPending(false);
    if (!result.ok) { setError(result.message); return; }
    setSnapshot(result.value);
  }
  function download(value: OperatorDecisionSnapshot) {
    const url = URL.createObjectURL(new Blob([value.text], { type: "application/json;charset=utf-8" }));
    try { const anchor = document.createElement("a"); anchor.href = url; anchor.download = "operator-policy-decision.json"; anchor.click(); }
    finally { URL.revokeObjectURL(url); }
  }
  return <section className="qsp-workspace" aria-label="Operator policy inspector">
    <h3>Operator policy limits</h3>
    <p>Inspect original core verdicts and provenance. This view performs no provider calls and grants no run approval.</p>
    <label>Operator policy JSON<textarea aria-label="Operator policy JSON" value={draft} rows={6} spellCheck={false} onChange={event => change(event.target.value)} /></label>
    <button type="button" disabled={pending} onClick={() => { void inspect(draft); }}>{pending ? "Inspecting policy…" : "Inspect policy decision"}</button>
    <button type="button" onClick={() => { change(example); void inspect(example); }}>Open policy example</button>
    <button type="button" disabled={snapshot === null} onClick={snapshot === null ? undefined : () => download(snapshot)}>Export admitted policy</button>
    {error !== null && <p role="alert">{error}</p>}
    {snapshot === null ? <p role="status">No operator policy decision admitted.</p> : <DecisionDetails snapshot={snapshot} />}
  </section>;
}
