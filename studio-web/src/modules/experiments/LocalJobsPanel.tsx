// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Run-stage panel of the optional local job service

import { useEffect, useId, useRef, useState } from "react";
import { LocalJobClient, appendEvents, cursorOf } from "../../shared/transport/localJobs";
import type {
  AttemptView,
  Fetch,
  JobEvent,
  JobRequest,
  JobResult,
} from "../../shared/transport/localJobs";

/** A request for the shipped three-node ``simulate`` handler, editable before running. */
export const EXAMPLE_REQUEST: JobRequest = {
  project: "studio-project",
  revision: "revision-1",
  idempotency_key: "simulate-3-node-1",
  verb: "simulate",
  action_id: "simulate-3-node",
  parameters: {
    K_nm: [
      [0, 0.5, 0.5],
      [0.5, 0, 0.5],
      [0.5, 0.5, 0],
    ],
    omega: [0.9, 1, 1.1],
    t_max: 1,
    dt: 0.1,
    trotter_per_step: 1,
    trotter_order: 1,
  },
};

/** Milliseconds between event polls while an attempt runs. */
export const POLL_MILLISECONDS = 500;

interface Refusal {
  readonly code: string;
  readonly message: string;
}

/**
 * Start, follow, reconnect to and cancel durable local jobs.
 *
 * The service is optional and started separately; this panel only talks to it
 * on the loopback interface with the session token it printed. Closing or
 * reloading the page does not stop a job: entering its attempt id reconnects
 * and replays every event from the start.
 *
 * @param props - Component properties.
 * @param props.focus - Focus the panel after it mounts, for a deep link.
 * @param props.fetcher - Fetch implementation; the browser's own when omitted.
 * @returns The panel.
 */
export default function LocalJobsPanel({
  focus = false,
  fetcher,
}: {
  readonly focus?: boolean;
  readonly fetcher?: Fetch;
}) {
  const [url, setUrl] = useState("http://127.0.0.1:8765");
  const [token, setToken] = useState("");
  const [requestText, setRequestText] = useState(JSON.stringify(EXAMPLE_REQUEST, null, 2));
  const [attemptText, setAttemptText] = useState("");
  const [attempt, setAttempt] = useState<AttemptView | null>(null);
  const [events, setEvents] = useState<readonly JobEvent[]>([]);
  const [plan, setPlan] = useState<Readonly<Record<string, unknown>> | null>(null);
  const [recordDigest, setRecordDigest] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const [following, setFollowing] = useState<{
    readonly attemptId: string;
    readonly service: LocalJobClient;
  } | null>(null);
  const section = useRef<HTMLElement>(null);
  const ids = { url: useId(), token: useId(), request: useId(), attempt: useId() };

  useEffect(() => {
    if (focus && section.current !== null) section.current.focus();
  }, [focus]);

  function client(): LocalJobClient | null {
    try {
      return new LocalJobClient(url, token, fetcher);
    } catch (cause: unknown) {
      setRefusal({ code: "invalid_service", message: (cause as Error).message });
      return null;
    }
  }

  function parsedRequest(): JobRequest | null {
    try {
      return JSON.parse(requestText) as JobRequest;
    } catch {
      setRefusal({ code: "invalid_request", message: "the request is not valid JSON" });
      return null;
    }
  }

  function settle<T>(result: JobResult<T>): T | null {
    if (result.ok) return result.value;
    setRefusal({ code: result.code, message: result.message });
    return null;
  }

  function follow(view: AttemptView, service: LocalJobClient) {
    setAttempt(view);
    setAttemptText(view.attemptId);
    setEvents([]);
    setRecordDigest(null);
    setFollowing({ attemptId: view.attemptId, service });
  }

  async function preview() {
    setRefusal(null);
    const service = client();
    const request = parsedRequest();
    if (service === null || request === null) return;
    setPlan(settle(await service.plan(request)));
  }

  async function run() {
    setRefusal(null);
    const service = client();
    const request = parsedRequest();
    if (service === null || request === null) return;
    const submitted = settle(await service.submit(request));
    if (submitted !== null) follow(submitted.attempt, service);
  }

  async function reconnect() {
    setRefusal(null);
    const service = client();
    if (service === null) return;
    const retrieved = settle(await service.retrieve(attemptText.trim()));
    if (retrieved !== null) follow(retrieved.attempt, service);
  }

  async function cancel(attemptId: string) {
    setRefusal(null);
    const service = client();
    if (service === null) return;
    const cancelled = settle(await service.cancel(attemptId));
    if (cancelled !== null) setAttempt(cancelled);
  }

  useEffect(() => {
    if (following === null) return;
    const { attemptId, service } = following;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let history: readonly JobEvent[] = [];
    // The attempt is read before its events: the service writes a terminal
    // state and its event in one transaction, so events read after a terminal
    // state always include the terminal event.
    const poll = async () => {
      const retrieved = await service.retrieve(attemptId);
      if (stopped) return;
      if (!retrieved.ok) {
        setRefusal({ code: retrieved.code, message: retrieved.message });
        return;
      }
      const page = await service.events(attemptId, cursorOf(history));
      if (stopped) return;
      if (!page.ok) {
        setRefusal({ code: page.code, message: page.message });
        return;
      }
      history = appendEvents(history, page.value);
      setEvents(history);
      setAttempt(retrieved.value.attempt);
      if (retrieved.value.attempt.terminal) {
        setRecordDigest(retrieved.value.attempt.resultDigest);
        return;
      }
      timer = setTimeout(() => void poll(), POLL_MILLISECONDS);
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [following]);

  return (
    <section
      ref={section}
      tabIndex={-1}
      aria-label="Local job service"
      className="qsp-panel"
      id="/experiments/local-jobs"
    >
      <h4>Local job service</h4>
      <p>
        Optional. Start it with{" "}
        <code>python -m scpn_quantum_control.studio_workspace.jobs.service --root DIR</code>; it
        prints its address and session token. Jobs keep running when this page closes. Only local
        verbs run there; nothing is submitted to a provider.
      </p>
      <label htmlFor={ids.url}>Service address</label>
      <input id={ids.url} value={url} onChange={(event) => setUrl(event.target.value)} />
      <label htmlFor={ids.token}>Session token</label>
      <input
        id={ids.token}
        type="password"
        autoComplete="off"
        value={token}
        onChange={(event) => setToken(event.target.value)}
      />
      <label htmlFor={ids.request}>Job request</label>
      <textarea
        id={ids.request}
        rows={12}
        value={requestText}
        onChange={(event) => setRequestText(event.target.value)}
      />
      <div>
        <button type="button" onClick={() => void preview()}>
          Preview plan
        </button>
        <button type="button" onClick={() => void run()}>
          Run job
        </button>
      </div>
      <label htmlFor={ids.attempt}>Attempt</label>
      <input
        id={ids.attempt}
        value={attemptText}
        onChange={(event) => setAttemptText(event.target.value)}
      />
      <div>
        <button type="button" onClick={() => void reconnect()}>
          Reconnect
        </button>
      </div>
      {refusal !== null && (
        <p role="alert" aria-label="Local job refused">
          {refusal.code}: {refusal.message}
        </p>
      )}
      {plan !== null && (
        <details open>
          <summary>Plan</summary>
          <section aria-label="Local job plan">
            <pre>{JSON.stringify(plan, null, 2)}</pre>
          </section>
        </details>
      )}
      {attempt !== null && (
        <>
          <p role="status" aria-label="Local job">
            {attempt.attemptId}: {attempt.state}
            {attempt.reason !== null && ` (${attempt.reason})`}
          </p>
          <button
            type="button"
            disabled={attempt.terminal}
            onClick={() => void cancel(attempt.attemptId)}
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
              {events.map((event) => (
                <tr key={event.seq}>
                  <td>{event.seq}</td>
                  <td>{event.kind}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {recordDigest !== null && (
            <p role="note" aria-label="Sealed record">
              Sealed record <code>{recordDigest}</code>
            </p>
          )}
        </>
      )}
    </section>
  );
}
