// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — one connection to the optional loopback job service

import { useCallback, useMemo, useState } from "react";

/** Address the service prints by default. */
export const DEFAULT_SERVICE_URL = "http://127.0.0.1:8765";

/** What the last call showed about the service. */
export type LocalServiceStatus = "not-connected" | "connected" | "refused";

/** The outcome of one call, as far as the connection is concerned. */
export type CallOutcome =
  | {
      /** The service answered. */
      readonly ok: true;
    }
  | {
      /** It did not. */
      readonly ok: false;
      /** Refusal code. */
      readonly code: string;
      /** Why. */
      readonly message: string;
    };

/** The shared connection: address, token and what the service last showed. */
export interface LocalServiceConnection {
  /** Service address. */ readonly url: string;
  /** Session token the service printed. */ readonly token: string;
  /** Status from the last call. */ readonly status: LocalServiceStatus;
  /** Why the status is what it is. */ readonly reason: string;
  /** Change the address; the status returns to not connected. */ readonly setUrl: (
    url: string,
  ) => void;
  /** Change the token; the status returns to not connected. */ readonly setToken: (
    token: string,
  ) => void;
  /** Record what a call showed about the service. */ readonly report: (
    outcome: CallOutcome,
  ) => void;
}

/** Codes that mean the service did not answer as a service: down, wrong address or malformed answer. */
const unanswered = new Set(["unreachable", "invalid_answer", "invalid_service"]);
const notConnected = { status: "not-connected", reason: "not connected yet" } as const;

/**
 * Hold one connection to the optional local job service.
 *
 * Nothing is contacted here; the status follows the calls the user makes. A
 * successful call marks the service connected, a call it did not answer marks
 * it refused, and a refusal of one request leaves the status as it was.
 *
 * @returns The connection.
 */
export function useLocalServiceConnection(): LocalServiceConnection {
  const [url, changeUrl] = useState(DEFAULT_SERVICE_URL);
  const [token, changeToken] = useState("");
  const [state, setState] = useState<{ status: LocalServiceStatus; reason: string }>(notConnected);
  const setUrl = useCallback((value: string) => {
    changeUrl(value);
    setState(notConnected);
  }, []);
  const setToken = useCallback((value: string) => {
    changeToken(value);
    setState(notConnected);
  }, []);
  const report = useCallback((outcome: CallOutcome) => {
    const next = outcome.ok
      ? { status: "connected" as const, reason: "the service answered" }
      : unanswered.has(outcome.code)
        ? { status: "refused" as const, reason: outcome.message }
        : null;
    // Returning the same state object skips a re-render while polling repeats an answer.
    setState((previous) =>
      next === null || (previous.status === next.status && previous.reason === next.reason)
        ? previous
        : next,
    );
  }, []);
  return useMemo(
    () => ({ url, token, ...state, setUrl, setToken, report }),
    [url, token, state, setUrl, setToken, report],
  );
}
