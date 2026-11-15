// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — sections of the local job panel

import { useId } from "react";
import type { LocalServiceConnection } from "../../shared/transport/localService";
import type { LocalJobs } from "./useLocalJobs";

/** Address and session token of the shared service connection. */
export function ServiceFields({ connection }: { readonly connection: LocalServiceConnection }) {
  const ids = { url: useId(), token: useId() };
  return (
    <>
      <label htmlFor={ids.url}>Service address</label>
      <input
        id={ids.url}
        value={connection.url}
        onChange={(event) => connection.setUrl(event.target.value)}
      />
      <label htmlFor={ids.token}>Session token</label>
      <input
        id={ids.token}
        type="password"
        autoComplete="off"
        value={connection.token}
        onChange={(event) => connection.setToken(event.target.value)}
      />
    </>
  );
}

/** The editable request, its plan and run actions, and reconnection by attempt id. */
export function RequestFields({ jobs }: { readonly jobs: LocalJobs }) {
  const ids = { request: useId(), attempt: useId() };
  return (
    <>
      <label htmlFor={ids.request}>Job request</label>
      <textarea
        id={ids.request}
        rows={12}
        value={jobs.requestText}
        onChange={(event) => jobs.setRequestText(event.target.value)}
      />
      <div>
        <button type="button" onClick={() => void jobs.preview()}>
          Preview plan
        </button>
        <button type="button" onClick={() => void jobs.run()}>
          Run job
        </button>
      </div>
      <label htmlFor={ids.attempt}>Attempt</label>
      <input
        id={ids.attempt}
        value={jobs.attemptText}
        onChange={(event) => jobs.setAttemptText(event.target.value)}
      />
      <div>
        <button type="button" onClick={() => void jobs.reconnect()}>
          Reconnect
        </button>
      </div>
    </>
  );
}

/** The refusal and the resolved plan, when there are any. */
export function JobNotices({ jobs }: { readonly jobs: LocalJobs }) {
  return (
    <>
      {jobs.refusal !== null && (
        <p role="alert" aria-label="Local job refused">
          {jobs.refusal.code}: {jobs.refusal.message}
        </p>
      )}
      {jobs.plan !== null && (
        <details open>
          <summary>Plan</summary>
          <section aria-label="Local job plan">
            <pre>{JSON.stringify(jobs.plan, null, 2)}</pre>
          </section>
        </details>
      )}
    </>
  );
}

/** The followed attempt: state, cancel, ordered events and the sealed record. */
export function AttemptSection({ jobs }: { readonly jobs: LocalJobs }) {
  const attempt = jobs.attempt;
  if (attempt === null) return null;
  return (
    <>
      <p role="status" aria-label="Local job">
        {attempt.attemptId}: {attempt.state}
        {attempt.reason !== null && ` (${attempt.reason})`}
      </p>
      <button
        type="button"
        disabled={attempt.terminal}
        onClick={() => void jobs.cancel(attempt.attemptId)}
      >
        Cancel job
      </button>
      <table aria-label="Local job events">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Event</th>
          </tr>
        </thead>
        <tbody>
          {jobs.events.map((event) => (
            <tr key={event.seq}>
              <td>{event.seq}</td>
              <td>{event.kind}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {jobs.recordDigest !== null && (
        <p role="note" aria-label="Sealed record">
          Sealed record <code>{jobs.recordDigest}</code>
        </p>
      )}
    </>
  );
}
