// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original operator review evidence

import { useEffect, useRef, useState } from "react";
import example from "../../../../../data/studio/operator_review_dossier.json?raw";
import { writeJson } from "../../../shared/contracts/jsonTransport";
import { parseOperatorDossier, reviewDocument, reviewStatus } from "./operatorDossier";
import type { HumanReview, OperatorDossierSnapshot } from "./operatorDossier";

function download(text: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  try { const link = document.createElement("a"); link.href = url; link.download = filename; link.click(); }
  finally { URL.revokeObjectURL(url); }
}

/** Display immutable source and record separate in-memory human review choices. */
export function OperatorDossier() {
  const [draft, setDraft] = useState(""), [snapshot, setSnapshot] = useState<OperatorDossierSnapshot | null>(null);
  const [review, setReview] = useState<HumanReview | null>(null), [reviewText, setReviewText] = useState<string | null>(null);
  const [reviewedSource, setReviewedSource] = useState<OperatorDossierSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null), [pending, setPending] = useState(false), [clean, setClean] = useState(false), [clock, setClock] = useState(() => new Date());
  const ticket = useRef(0), mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; ticket.current++; }; }, []);
  useEffect(() => {
    if (snapshot === null) return;
    const interval = window.setInterval(() => setClock(new Date()), 250);
    return () => window.clearInterval(interval);
  }, [snapshot]);
  function change(value: string) { ticket.current++; setDraft(value); setClean(false); setError(null); setPending(false); }
  async function inspect(raw: string) {
    const current = ++ticket.current; setPending(true); setClean(false); setError(null);
    const result = await parseOperatorDossier(raw);
    if (!mounted.current || current !== ticket.current) return;
    setPending(false);
    if (!result.ok) { setError(result.message); return; }
    setSnapshot(result.value); setClean(true); setClock(new Date());
  }
  async function decide(source: OperatorDossierSnapshot, choice: HumanReview["choice"]) {
    if (reviewStatus(source, null) !== "pending") return;
    const current = ++ticket.current;
    const record = Object.freeze({ dossierSha256: source.sha256, executionSha256: source.executionSha256, choice, recordedAt: new Date().toISOString().slice(0, 19) + "Z" });
    const text = await reviewDocument(record);
    if (!mounted.current || current !== ticket.current || reviewStatus(source, null) !== "pending") return;
    setReview(record); setReviewText(text); setReviewedSource(source); setClock(new Date());
  }
  const status = snapshot === null ? "no dossier" : !clean ? "draft changed" : reviewStatus(snapshot, review, clock);
  const eligible = snapshot !== null && clean && !pending && reviewStatus(snapshot, null, clock) === "pending";
  return <section className="qsp-workspace" aria-label="Operator review dossier">
    <h3>Operator review dossier</h3>
    <p>Review the original deployment, exact payload, dated price and calibration. Human review grants no provider submission authority. Synthetic examples retain their source dates.</p>
    <label>Operator dossier JSON<textarea aria-label="Operator dossier JSON" value={draft} rows={6} spellCheck={false} onChange={event => change(event.target.value)} /></label>
    <button type="button" disabled={pending} onClick={() => { void inspect(draft); }}>{pending ? "Inspecting dossier…" : "Inspect operator dossier"}</button>
    <button type="button" onClick={() => { change(example); void inspect(example); }}>Open dossier example</button>
    <button type="button" disabled={snapshot === null} onClick={snapshot === null ? undefined : () => download(snapshot.dossierText, "operator-review-dossier.json")}>Export admitted dossier</button>
    <button type="button" disabled={snapshot === null} onClick={snapshot === null ? undefined : () => download(snapshot.script, "verify_operator_review.py")}>Export native verifier</button>
    <button type="button" disabled={!eligible} onClick={eligible && snapshot !== null ? () => { void decide(snapshot, "approved"); } : undefined}>Approve human review</button>
    <button type="button" disabled={!eligible} onClick={eligible && snapshot !== null ? () => { void decide(snapshot, "denied"); } : undefined}>Deny human review</button>
    <button type="button" disabled={reviewText === null || reviewedSource === null} onClick={reviewText === null || reviewedSource === null ? undefined : () => download(writeJson({ original_export: reviewedSource.text, review_text: reviewText }) + "\n", "operator-human-review.json")}>Export human review</button>
    <p aria-label="Operator review status" role="status">{status}</p>
    {error !== null && <p role="alert">{error}</p>}
    {snapshot !== null && <section aria-label="Admitted operator dossier">
      <p>Dossier SHA-256: <code aria-label="Admitted dossier identity">{snapshot.sha256}</code></p>
      <p>Execution SHA-256: <code aria-label="Admitted execution identity">{snapshot.executionSha256}</code></p>
      <dl aria-label="Original deployment plan">{Object.entries(snapshot.body["plan"] as Record<string, unknown>).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{writeJson(value)}</dd></div>)}</dl>
      <dl aria-label="Original compiled payload">{Object.entries(snapshot.body["payload"] as Record<string, unknown>).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{writeJson(value)}</dd></div>)}</dl>
      <p aria-label="Original backend profile">{writeJson(snapshot.body["profile"])}</p>
      <p aria-label="Original resolved settings">{writeJson(snapshot.body["settings"])}</p>
      <p aria-label="Original policy verdict">{writeJson(snapshot.body["policy_decision"])}</p>
      <p aria-label="Original price estimate">{snapshot.policy.decision.estimate === null || snapshot.policy.decision.estimate["amount"] === null ? "unknown" : writeJson(snapshot.policy.decision.estimate)}</p>
      <p aria-label="Original calibration">{snapshot.body["calibration"] === null ? "unknown" : writeJson(snapshot.body["calibration"])}</p>
      <p>Source creation: <time>{String(snapshot.body["created_at"])}</time>. Exclusive expiry: <time>{String(snapshot.body["expires_at"])}</time>.</p>
      {review !== null && <p aria-label="Original human review reference">{review.choice}: {review.dossierSha256}, recorded {review.recordedAt}</p>}
    </section>}
  </section>;
}
