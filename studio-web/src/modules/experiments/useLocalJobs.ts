// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — state, actions and event polling of the local job panel

import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { LocalJobClient, appendEvents, cursorOf } from "../../shared/transport/localJobs";
import type {
  AttemptView,
  Fetch,
  JobEvent,
  JobRequest,
  JobResult,
} from "../../shared/transport/localJobs";
import type { LocalServiceConnection } from "../../shared/transport/localService";

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

/** Why the service or the panel refused a step. */
export interface Refusal {
  /** Refusal code. */ readonly code: string;
  /** Human-readable reason. */ readonly message: string;
}

/** The attempt the panel follows and the client that polls it. */
export interface Following {
  /** Identifier of the followed attempt. */
  readonly attemptId: string;
  /** Client that polls the attempt and its events. */
  readonly service: LocalJobClient;
}

/** The panel's state, its setters and its four actions. */
export interface LocalJobs {
  /** Editable JSON text of the job request. */
  readonly requestText: string;
  /** Attempt identifier typed for reconnecting. */
  readonly attemptText: string;
  /** The attempt as the service last reported it. */
  readonly attempt: AttemptView | null;
  /** Every event of the followed attempt, in order. */
  readonly events: readonly JobEvent[];
  /** The plan the service returned for a preview. */
  readonly plan: Readonly<Record<string, unknown>> | null;
  /** Digest of the result record once the attempt is terminal. */
  readonly recordDigest: string | null;
  /** The last refusal, shown until the next action. */
  readonly refusal: Refusal | null;
  /** The attempt being polled, if any. */
  readonly following: Following | null;
  /** Replace the request text. */
  readonly setRequestText: Dispatch<SetStateAction<string>>;
  /** Replace the attempt identifier text. */
  readonly setAttemptText: Dispatch<SetStateAction<string>>;
  /** Replace the shown attempt. */
  readonly setAttempt: Dispatch<SetStateAction<AttemptView | null>>;
  /** Replace the shown events. */
  readonly setEvents: Dispatch<SetStateAction<readonly JobEvent[]>>;
  /** Replace the shown plan. */
  readonly setPlan: Dispatch<SetStateAction<Readonly<Record<string, unknown>> | null>>;
  /** Replace the result record digest. */
  readonly setRecordDigest: Dispatch<SetStateAction<string | null>>;
  /** Show or clear a refusal. */
  readonly setRefusal: Dispatch<SetStateAction<Refusal | null>>;
  /** Start or stop following an attempt. */
  readonly setFollowing: Dispatch<SetStateAction<Following | null>>;
  /** Ask the service for the plan of the current request without running it. */
  readonly preview: () => Promise<void>;
  /** Submit the current request and follow the new attempt. */
  readonly run: () => Promise<void>;
  /** Follow the attempt named in the attempt text again. */
  readonly reconnect: () => Promise<void>;
  /** Ask the service to cancel one attempt. */
  readonly cancel: (attemptId: string) => Promise<void>;
}

function useJobState() {
  const [requestText, setRequestText] = useState(JSON.stringify(EXAMPLE_REQUEST, null, 2));
  const [attemptText, setAttemptText] = useState("");
  const [attempt, setAttempt] = useState<AttemptView | null>(null);
  const [events, setEvents] = useState<readonly JobEvent[]>([]);
  const [plan, setPlan] = useState<Readonly<Record<string, unknown>> | null>(null);
  const [recordDigest, setRecordDigest] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const [following, setFollowing] = useState<Following | null>(null);
  return {
    ...{ requestText, attemptText, attempt, events, plan, recordDigest, refusal, following },
    ...{ setRequestText, setAttemptText, setAttempt, setEvents, setPlan, setRecordDigest },
    ...{ setRefusal, setFollowing },
  };
}

type JobState = ReturnType<typeof useJobState>;

/** Settle a result: keep its value, or show and report its refusal. */
function settler(connection: LocalServiceConnection, state: JobState) {
  return <T>(result: JobResult<T>): T | null => {
    connection.report(result.ok ? { ok: true } : result);
    if (result.ok) return result.value;
    state.setRefusal({ code: result.code, message: result.message });
    return null;
  };
}

function jobActions(
  connection: LocalServiceConnection,
  fetcher: Fetch | undefined,
  state: JobState,
) {
  const settle = settler(connection, state);
  const client = (): LocalJobClient | null => {
    try {
      return new LocalJobClient(connection.url, connection.token, fetcher);
    } catch (cause: unknown) {
      const refusal = { code: "invalid_service", message: (cause as Error).message };
      state.setRefusal(refusal);
      connection.report({ ok: false, ...refusal });
      return null;
    }
  };
  const request = (): JobRequest | null => {
    try {
      return JSON.parse(state.requestText) as JobRequest;
    } catch {
      state.setRefusal({ code: "invalid_request", message: "the request is not valid JSON" });
      return null;
    }
  };
  const follow = (view: AttemptView, service: LocalJobClient) => {
    state.setAttempt(view);
    state.setAttemptText(view.attemptId);
    state.setEvents([]);
    state.setRecordDigest(null);
    state.setFollowing({ attemptId: view.attemptId, service });
  };
  return { settle, client, request, follow };
}

function panelActions(
  connection: LocalServiceConnection,
  fetcher: Fetch | undefined,
  state: JobState,
) {
  const { settle, client, request, follow } = jobActions(connection, fetcher, state);
  const begin = () => {
    state.setRefusal(null);
    return client();
  };
  return {
    preview: async () => {
      const service = begin();
      const body = request();
      if (service !== null && body !== null) state.setPlan(settle(await service.plan(body)));
    },
    run: async () => {
      const service = begin();
      const body = request();
      const submitted =
        service !== null && body !== null ? settle(await service.submit(body)) : null;
      if (submitted !== null && service !== null) follow(submitted.attempt, service);
    },
    reconnect: async () => {
      const service = begin();
      const retrieved =
        service === null ? null : settle(await service.retrieve(state.attemptText.trim()));
      if (retrieved !== null && service !== null) follow(retrieved.attempt, service);
    },
    cancel: async (attemptId: string) => {
      const service = begin();
      const cancelled = service === null ? null : settle(await service.cancel(attemptId));
      if (cancelled !== null) state.setAttempt(cancelled);
    },
  };
}

function useAttemptPolling(connection: LocalServiceConnection, state: JobState): void {
  const { following, setRefusal, setEvents, setAttempt, setRecordDigest } = state;
  const { report } = connection;
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
        report(retrieved);
        return setRefusal({ code: retrieved.code, message: retrieved.message });
      }
      const page = await service.events(attemptId, cursorOf(history));
      if (stopped) return;
      report(page.ok ? { ok: true } : page);
      if (!page.ok) return setRefusal({ code: page.code, message: page.message });
      history = appendEvents(history, page.value);
      setEvents(history);
      setAttempt(retrieved.value.attempt);
      if (retrieved.value.attempt.terminal)
        return setRecordDigest(retrieved.value.attempt.resultDigest);
      timer = setTimeout(() => void poll(), POLL_MILLISECONDS);
    };
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [following, report, setRefusal, setEvents, setAttempt, setRecordDigest]);
}

/**
 * Hold the panel's request, attempt and events, and its four actions.
 *
 * Every answer or failure of the service is reported to the shared connection,
 * so the rest of the workbench sees whether the service is reachable.
 *
 * @param connection - The shared connection to the local service.
 * @param fetcher - Fetch implementation; the browser's own when omitted.
 * @returns State, setters and the preview, run, reconnect and cancel actions.
 */
export function useLocalJobs(connection: LocalServiceConnection, fetcher?: Fetch): LocalJobs {
  const state = useJobState();
  useAttemptPolling(connection, state);
  return { ...state, ...panelActions(connection, fetcher, state) };
}
