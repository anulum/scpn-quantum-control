// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — existing provider job journal

import { useEffect, useRef, useState } from "react";
import recorded from "../../../../../data/studio/provider_observation_cases.json";
import { writeJson } from "../../../shared/contracts/jsonTransport";
import { isProviderObservationRefresh, parseProviderObservation } from "./providerObservation";
import type { ProviderObservation } from "./providerObservation";

/** Inspect and refresh exact native exports while preserving original job identity and evidence. */
export function JobJournal() {
  const [draft, setDraft] = useState("");
  const [admitted, setAdmitted] = useState<ProviderObservation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const ticket = useRef(0);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      ticket.current++;
    };
  }, []);

  async function inspect(raw: string, previous: ProviderObservation | null) {
    const current = ++ticket.current;
    setPending(true);
    setError(null);
    const result = await parseProviderObservation(raw);
    if (!mounted.current || current !== ticket.current) return;
    setPending(false);
    if (!result.ok) {
      setError(result.message);
      return;
    }
    if (previous !== null && !isProviderObservationRefresh(previous, result.value)) {
      setError("Refresh refused: original request, provider identity or retained history differs.");
      return;
    }
    setAdmitted(result.value);
  }

  function download(snapshot: ProviderObservation) {
    const url = URL.createObjectURL(
      new Blob([snapshot.text], { type: "application/json;charset=utf-8" }),
    );
    try {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "provider-observation.json";
      anchor.click();
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  return (
    <section className="qsp-workspace" aria-label="Existing provider job journal">
      <h3>Existing provider job</h3>
      <p>
        Paste a native journal export to inspect an existing job. For a newer observation,
        explicitly observe or reconcile the original job with the native runner, export a new
        snapshot, then refresh here. Refresh reads supplied evidence and never submits a job.
      </p>
      <label>
        Provider observation JSON
        <textarea
          aria-label="Provider observation JSON"
          value={draft}
          rows={6}
          spellCheck={false}
          onChange={(event) => {
            ticket.current++;
            setDraft(event.target.value);
            setError(null);
            setPending(false);
          }}
        />
      </label>
      <button
        type="button"
        disabled={pending}
        onClick={() => {
          void inspect(draft, null);
        }}
      >
        Import existing job observation
      </button>
      <button
        type="button"
        disabled={pending || admitted === null}
        onClick={
          admitted === null
            ? undefined
            : () => {
                void inspect(draft, admitted);
              }
        }
      >
        Refresh existing job observation
      </button>
      <button
        type="button"
        onClick={() => {
          setDraft(recorded.cases.late_partial);
          void inspect(recorded.cases.late_partial, null);
        }}
      >
        Open recorded local sampler job
      </button>
      <p>
        The recorded example is a local Aer simulation with 16 shots; it is not a physical device
        run.
      </p>
      <button
        type="button"
        disabled={admitted === null}
        onClick={admitted === null ? undefined : () => download(admitted)}
      >
        Export admitted job observation
      </button>
      {pending && <p role="status">Inspecting provider observation…</p>}
      {error !== null && <p role="alert">{error}</p>}
      {admitted === null && <p role="status">No existing job observation admitted.</p>}
      {admitted !== null && (
        <section aria-label="Admitted provider job">
          <dl>
            <dt>Attempt</dt>
            <dd>
              <span role="note" aria-label="Provider attempt identity">
                {admitted.body.attempt_id}
              </span>
            </dd>
            <dt>Provider job</dt>
            <dd>
              <span role="note" aria-label="Provider job identity">
                {admitted.body.provider_job_id ?? "unknown — explicit reconciliation required"}
              </span>
            </dd>
            <dt>Target</dt>
            <dd>{admitted.body.target}</dd>
            <dt>Local state</dt>
            <dd>
              <span role="note" aria-label="Local job state">
                {admitted.body.local_state}
              </span>
            </dd>
            <dt>Provider state</dt>
            <dd>
              <span role="note" aria-label="Provider job state">
                {admitted.body.provider_state ?? "unknown — no status retained"}
              </span>
            </dd>
            <dt>Billing</dt>
            <dd>{admitted.body.billing}</dd>
            <dt>Requested shots per circuit</dt>
            <dd>{admitted.body.shots.toString()}</dd>
            <dt>Original circuit count</dt>
            <dd>{admitted.body.circuits.toString()}</dd>
            <dt>Original payload SHA-256</dt>
            <dd>
              <span role="note" aria-label="Provider payload digest">
                {admitted.body.payload_sha256}
              </span>
            </dd>
            <dt>Observation SHA-256</dt>
            <dd>
              <span role="note" aria-label="Provider observation digest">
                {admitted.sha256}
              </span>
            </dd>
          </dl>
          <p>
            Supplied observations and content digests do not authenticate a provider or prove a
            charge. Cancellation intent and confirmed completion remain separate in the history. The
            original journal does not retain event timestamps.
          </p>
          <ol aria-label="Original provider observation history">
            {admitted.body.events.map((event) => (
              <li key={event.sequence.toString()}>
                <p>
                  Sequence {event.sequence.toString()} · {event.state}
                </p>
                <pre>{writeJson(event.observation)}</pre>
              </li>
            ))}
          </ol>
        </section>
      )}
    </section>
  );
}
