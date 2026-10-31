// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — typed client of the optional local job service

import { count, flag, list, record, text } from "../browser/jsonFields";

/** Header carrying the per-run session token. */
export const TOKEN_HEADER = "X-Studio-Job-Token";

/** The only fields a job request may carry. */
export const REQUEST_FIELDS = [
  "action_id",
  "backend",
  "idempotency_key",
  "parameters",
  "project",
  "revision",
  "verb",
] as const;

/** States after which an attempt never changes. */
export const TERMINAL_STATES = [
  "cancelled",
  "expired",
  "failed",
  "interrupted",
  "succeeded",
] as const;

/** Attempt states, in the order an attempt can reach them. */
export type AttemptState = "queued" | "running" | (typeof TERMINAL_STATES)[number];

/** A job request, exactly as the service admits it. */
export interface JobRequest {
  /** Project identity. */
  readonly project: string;
  /** Revision identity. */
  readonly revision: string;
  /** Caller key; one attempt exists per key. */
  readonly idempotency_key: string;
  /** Local verb, such as ``simulate``. */
  readonly verb: string;
  /** Action identity. */
  readonly action_id: string;
  /** Verb parameters. */
  readonly parameters: Readonly<Record<string, unknown>>;
  /** Optional backend. */
  readonly backend?: string | null;
}

/** The caller-facing state of one attempt. */
export interface AttemptView {
  /** Attempt identity. */
  readonly attemptId: string;
  /** Project identity. */
  readonly project: string;
  /** Revision identity. */
  readonly revision: string;
  /** Caller key. */
  readonly idempotencyKey: string;
  /** Verb. */
  readonly verb: string;
  /** Action identity. */
  readonly actionId: string;
  /** The handler's plan. */
  readonly plan: Readonly<Record<string, unknown>>;
  /** Current state. */
  readonly state: AttemptState;
  /** Whether the attempt has ended. */
  readonly terminal: boolean;
  /** Why it ended, when it did not succeed. */
  readonly reason: string | null;
  /** Digest of its sealed record, once it has one. */
  readonly resultDigest: string | null;
}

/** One ordered attempt event. */
export interface JobEvent {
  /** Attempt identity. */
  readonly attemptId: string;
  /** Position in the attempt's history, from one. */
  readonly seq: number;
  /** Event kind. */
  readonly kind: string;
  /** Event details. */
  readonly payload: Readonly<Record<string, unknown>>;
  /** Wall-clock time in seconds. */
  readonly at: number;
}

/** Why the client or the service refused a request. */
export interface Refusal {
  /** Stable refusal category, such as ``idempotency_conflict``. */
  readonly code: string;
  /** Explanation safe to show. */
  readonly message: string;
}

/** A successful service answer. */
export interface JobAnswer<T> {
  /** Always true for an answer. */
  readonly ok: true;
  /** The parsed answer. */
  readonly value: T;
}

/** A refused request. */
export interface JobRefusal extends Refusal {
  /** Always false for a refusal. */
  readonly ok: false;
}

/** A service answer, or the reason the client or the service refused. */
export type JobResult<T> = JobAnswer<T> | JobRefusal;

/** What ``submit`` returns. */
export interface Submitted {
  /** The admitted attempt, new or repeated. */
  readonly attempt: AttemptView;
  /** Whether this request created the attempt. */
  readonly created: boolean;
}

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const STATES: readonly string[] = ["queued", "running", ...TERMINAL_STATES];

function refused(code: string, message: string): JobRefusal {
  return { ok: false, code, message };
}

function optionalText(value: unknown, field: string): string | null {
  return value === null ? null : text(value, field);
}

/**
 * Refuse a request the service would refuse for its shape alone.
 *
 * Unknown fields, missing fields, identifiers that could name a path, and
 * non-object parameters are refused before anything is sent; whether a verb
 * is local and its parameters valid is for the service to decide.
 *
 * @param request - Candidate request.
 * @returns ``null`` when the request may be sent, else the refusal.
 */
export function requestRefusal(request: unknown): Refusal | null {
  if (typeof request !== "object" || request === null || Array.isArray(request))
    return { code: "invalid_request", message: "a job request must be a JSON object" };
  const fields = Object.keys(request);
  const unknown = fields.filter((field) => !(REQUEST_FIELDS as readonly string[]).includes(field));
  if (unknown.length > 0)
    return {
      code: "invalid_request",
      message: `unknown request fields: ${unknown.sort().join(", ")}`,
    };
  const missing = REQUEST_FIELDS.filter((field) => field !== "backend" && !fields.includes(field));
  if (missing.length > 0)
    return { code: "invalid_request", message: `missing request fields: ${missing.join(", ")}` };
  const values = request as Record<string, unknown>;
  for (const field of ["project", "revision", "idempotency_key", "action_id"] as const) {
    const value = values[field];
    if (typeof value !== "string" || !IDENTIFIER.test(value))
      return { code: "invalid_request", message: `${field} must be a short identifier` };
  }
  const backend = values["backend"];
  if (
    backend !== undefined &&
    backend !== null &&
    (typeof backend !== "string" || !IDENTIFIER.test(backend))
  )
    return { code: "invalid_request", message: "backend must be a short identifier" };
  const parameters = values["parameters"];
  if (typeof parameters !== "object" || parameters === null || Array.isArray(parameters))
    return { code: "invalid_request", message: "parameters must be a JSON object" };
  if (typeof values["verb"] !== "string")
    return { code: "invalid_request", message: "verb must be a string" };
  return null;
}

/**
 * Parse an attempt view, refusing an unknown state or a terminal flag that disagrees with it.
 *
 * @param raw - Decoded service answer.
 * @returns The attempt.
 * @throws Error when the answer is malformed.
 */
export function parseAttempt(raw: unknown): AttemptView {
  const row = record(raw, "attempt");
  const state = text(row["state"], "attempt.state");
  if (!STATES.includes(state)) throw new Error(`attempt.state ${state} is not a known state`);
  const terminal = flag(row["terminal"], "attempt.terminal");
  if (terminal !== (TERMINAL_STATES as readonly string[]).includes(state))
    throw new Error("attempt.terminal disagrees with its state");
  return {
    attemptId: text(row["attempt_id"], "attempt.attempt_id"),
    project: text(row["project"], "attempt.project"),
    revision: text(row["revision"], "attempt.revision"),
    idempotencyKey: text(row["idempotency_key"], "attempt.idempotency_key"),
    verb: text(row["verb"], "attempt.verb"),
    actionId: text(row["action_id"], "attempt.action_id"),
    plan: record(row["plan"], "attempt.plan"),
    state: state as AttemptState,
    terminal,
    reason: optionalText(row["reason"], "attempt.reason"),
    resultDigest: optionalText(row["result_digest"], "attempt.result_digest"),
  };
}

/**
 * Parse an event page, refusing one that is not strictly consecutive after ``after``.
 *
 * @param raw - Decoded list of events.
 * @param attemptId - Attempt the events must belong to.
 * @param after - Cursor the page was requested after.
 * @returns The events in order.
 * @throws Error when an event is malformed, foreign, or out of sequence.
 */
export function parseEvents(raw: unknown, attemptId: string, after: number): JobEvent[] {
  return list(raw, "events").map((item, index) => {
    const row = record(item, `events[${index}]`);
    const seq = count(row["seq"], `events[${index}].seq`);
    if (seq !== after + index + 1)
      throw new Error(`events[${index}].seq ${seq} breaks the sequence`);
    const owner = text(row["attempt_id"], `events[${index}].attempt_id`);
    if (owner !== attemptId) throw new Error(`events[${index}] belongs to another attempt`);
    const at = row["at"];
    if (typeof at !== "number" || !Number.isFinite(at))
      throw new Error(`events[${index}].at must be a time`);
    return {
      attemptId: owner,
      seq,
      kind: text(row["kind"], `events[${index}].kind`),
      payload: record(row["payload"], `events[${index}].payload`),
      at,
    };
  });
}

/** What ``retrieve`` returns: the attempt and its sealed record once it has one. */
export interface Retrieved {
  /** The attempt. */
  readonly attempt: AttemptView;
  /** The sealed executive record, or ``null``. */
  readonly record: Readonly<Record<string, unknown>> | null;
}

/** Fetch function the client uses; tests and the browser pass the real one. */
export type Fetch = (input: string, init: RequestInit) => Promise<Response>;

/**
 * Typed client of one running local job service.
 *
 * The service lives on the loopback interface; any other base URL is refused
 * at construction, so a misconfigured client cannot send the token elsewhere.
 */
export class LocalJobClient {
  private readonly base: string;
  private readonly token: string;
  private readonly fetcher: Fetch;

  /**
   * @param baseUrl - ``http://127.0.0.1:<port>`` of the service.
   * @param token - Session token the service printed.
   * @param fetcher - Fetch implementation.
   * @throws Error when the URL is not the loopback service or the token is too short.
   */
  constructor(
    baseUrl: string,
    token: string,
    fetcher: Fetch = (input, init) => fetch(input, init),
  ) {
    this.token = token;
    this.fetcher = fetcher;
    const url = new URL(baseUrl);
    if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || url.port === "")
      throw new Error("the local job service must be http://127.0.0.1:<port>");
    if (url.pathname !== "/" || url.search !== "" || url.username !== "" || url.password !== "")
      throw new Error("the service URL must be only the scheme, host and port");
    if (token.length < 32) throw new Error("the session token must have at least 32 characters");
    this.base = url.origin;
  }

  private async call<T>(
    method: "GET" | "POST",
    path: string,
    body: unknown,
    parse: (raw: unknown) => T,
  ): Promise<JobResult<T>> {
    let response: Response;
    try {
      response = await this.fetcher(`${this.base}${path}`, {
        method,
        headers: { [TOKEN_HEADER]: this.token, "Content-Type": "application/json" },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
    } catch {
      return refused("unreachable", "the local job service did not answer");
    }
    let raw: unknown;
    try {
      raw = await response.json();
    } catch {
      return refused("invalid_answer", "the service answer is not JSON");
    }
    if (!response.ok) {
      const row = typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {};
      return refused(
        typeof row["error"] === "string" ? row["error"] : "http_error",
        typeof row["message"] === "string" ? row["message"] : `HTTP ${response.status}`,
      );
    }
    try {
      return { ok: true, value: parse(raw) };
    } catch (cause: unknown) {
      return refused("invalid_answer", (cause as Error).message);
    }
  }

  /**
   * Resolve the handler's plan without admitting the request.
   *
   * @param request - Job request.
   * @returns The plan or the refusal.
   */
  async plan(request: JobRequest): Promise<JobResult<Readonly<Record<string, unknown>>>> {
    const refusal = requestRefusal(request);
    if (refusal !== null) return refused(refusal.code, refusal.message);
    return this.call("POST", "/v1/plan", request, (raw) =>
      record(record(raw, "answer")["plan"], "plan"),
    );
  }

  /**
   * Admit a request once per idempotency key.
   *
   * @param request - Job request.
   * @returns The attempt and whether it was created now, or the refusal.
   */
  async submit(request: JobRequest): Promise<JobResult<Submitted>> {
    const refusal = requestRefusal(request);
    if (refusal !== null) return refused(refusal.code, refusal.message);
    return this.call("POST", "/v1/jobs", request, (raw) => ({
      attempt: parseAttempt(raw),
      created: flag(record(raw, "answer")["created"], "created"),
    }));
  }

  /**
   * Return the events after a cursor.
   *
   * @param attemptId - Attempt identity.
   * @param after - Last sequence number already seen.
   * @returns The next events in order, or the refusal.
   */
  async events(attemptId: string, after: number): Promise<JobResult<JobEvent[]>> {
    return this.call(
      "GET",
      `/v1/jobs/${encodeURIComponent(attemptId)}/events?after=${after}`,
      undefined,
      (raw) => parseEvents(record(raw, "answer")["events"], attemptId, after),
    );
  }

  /**
   * Return an attempt and its sealed record.
   *
   * @param attemptId - Attempt identity.
   * @returns The attempt and record, or the refusal.
   */
  async retrieve(attemptId: string): Promise<JobResult<Retrieved>> {
    return this.call("GET", `/v1/jobs/${encodeURIComponent(attemptId)}`, undefined, (raw) => {
      const answer = record(raw, "answer");
      return {
        attempt: parseAttempt(answer),
        record: answer["record"] === null ? null : record(answer["record"], "record"),
      };
    });
  }

  /**
   * Cancel an attempt; an ended one is returned as it is.
   *
   * @param attemptId - Attempt identity.
   * @returns The attempt after cancellation, or the refusal.
   */
  async cancel(attemptId: string): Promise<JobResult<AttemptView>> {
    return this.call("POST", `/v1/jobs/${encodeURIComponent(attemptId)}/cancel`, {}, parseAttempt);
  }
}

/**
 * Append a page of events to a history, keeping it strictly consecutive.
 *
 * @param history - Events already held.
 * @param page - Events returned after ``history``'s last sequence number.
 * @returns The extended history.
 * @throws Error when the page does not continue the history.
 */
export function appendEvents(history: readonly JobEvent[], page: readonly JobEvent[]): JobEvent[] {
  const last = history.length === 0 ? 0 : (history[history.length - 1] as JobEvent).seq;
  page.forEach((event, index) => {
    if (event.seq !== last + index + 1) throw new Error("events do not continue the history");
  });
  return [...history, ...page];
}

/** Last sequence number of a history, the cursor of the next request. */
export function cursorOf(history: readonly JobEvent[]): number {
  return history.length === 0 ? 0 : (history[history.length - 1] as JobEvent).seq;
}
