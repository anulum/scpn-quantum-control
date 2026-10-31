// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the local job client agrees with the service's shared fixtures

import { afterEach, describe, expect, it, vi } from "vitest";
import fixtures from "../../../../data/studio/local_job_transport_fixtures.json";
import {
  LocalJobClient,
  REQUEST_FIELDS,
  TERMINAL_STATES,
  TOKEN_HEADER,
  appendEvents,
  cursorOf,
  parseAttempt,
  parseEvents,
  requestRefusal,
} from "./localJobs";
import type { Fetch, JobRequest } from "./localJobs";

type Json = Record<string, unknown>;

const TOKEN = "t".repeat(40);
const BASE = "http://127.0.0.1:8765";
const attemptId = fixtures.attempt.attempt_id;
const request = (fixtures.requests[0] as { request: JobRequest }).request;

function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** A fetcher answering every call with one response, recording what was sent. */
function answering(status: number, body: unknown) {
  const sent: { url: string; init: RequestInit }[] = [];
  const fetcher: Fetch = async (url, init) => {
    sent.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  };
  return { fetcher, sent };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("local job client", () => {
  it("uses the browser's fetch unless given another", async () => {
    const { fetcher, sent } = answering(200, { plan: { verb: "simulate" } });
    vi.stubGlobal("fetch", fetcher);
    const planned = await new LocalJobClient(BASE, TOKEN).plan(request);
    expect(planned.ok).toBe(true);
    expect(sent).toHaveLength(1);
  });

  it("shares the service's request fields, token header and terminal states", () => {
    expect([...REQUEST_FIELDS]).toEqual(fixtures.request_fields);
    expect(TOKEN_HEADER).toBe(fixtures.token_header);
    expect([...TERMINAL_STATES]).toEqual(fixtures.terminal_states);
  });

  it("refuses exactly the requests the service refuses for their shape", () => {
    for (const item of fixtures.requests) {
      const refusal = requestRefusal(item.request);
      if (item.side === "both" && item.code !== null)
        expect(refusal?.code, item.name).toBe(item.code);
      else expect(refusal, item.name).toBeNull();
    }
    expect(requestRefusal(null)?.message).toBe("a job request must be a JSON object");
    expect(requestRefusal({ ...request, verb: 3 })?.message).toBe("verb must be a string");
    expect(requestRefusal({ ...request, backend: null })).toBeNull();
  });

  it("parses the shared attempt and events and refuses inconsistent ones", () => {
    const attempt = parseAttempt(fixtures.attempt);
    expect(attempt).toMatchObject({ attemptId, state: "succeeded", terminal: true, reason: null });
    const events = parseEvents(fixtures.events, attemptId, 0);
    expect(events.map((event) => event.kind)).toEqual(["admitted", "started", "succeeded"]);
    expect(parseEvents(fixtures.events.slice(1), attemptId, 1)).toEqual(events.slice(1));
    const refusals: [() => unknown, string][] = [
      [() => parseAttempt({ ...fixtures.attempt, state: "paused" }), "not a known state"],
      [() => parseAttempt({ ...fixtures.attempt, terminal: false }), "disagrees with its state"],
      [() => parseEvents(fixtures.events, attemptId, 1), "breaks the sequence"],
      [() => parseEvents(fixtures.events, "job-other", 0), "belongs to another attempt"],
      [
        () => parseEvents([{ ...fixtures.events[0], at: "now" }], attemptId, 0),
        "events[0].at must be a time",
      ],
    ];
    for (const [parse, message] of refusals) expect(parse).toThrow(message);
  });

  it("extends a history only with consecutive events", () => {
    const events = parseEvents(fixtures.events, attemptId, 0);
    expect(cursorOf([])).toBe(0);
    const history = appendEvents(appendEvents([], events.slice(0, 1)), events.slice(1));
    expect(cursorOf(history)).toBe(3);
    expect(() => appendEvents(history, events.slice(0, 1))).toThrow("do not continue");
  });

  it("talks only to the loopback service with a real token", () => {
    for (const [url, token] of [
      ["https://127.0.0.1:8765", TOKEN],
      ["http://localhost:8765", TOKEN],
      ["http://127.0.0.1", TOKEN],
      ["http://127.0.0.1:8765/v1", TOKEN],
      ["http://user:pass@127.0.0.1:8765", TOKEN],
      [BASE, "short"],
    ] as const)
      expect(() => new LocalJobClient(url, token), url).toThrow();
  });

  it("sends each operation with the token and parses the answer", async () => {
    const answers: [keyof LocalJobClient, unknown[], unknown, string, string][] = [
      ["plan", [request], { plan: { verb: "simulate" } }, "POST", "/v1/plan"],
      ["submit", [request], { ...fixtures.attempt, created: true }, "POST", "/v1/jobs"],
      [
        "events",
        [attemptId, 0],
        { events: fixtures.events },
        "GET",
        `/v1/jobs/${attemptId}/events?after=0`,
      ],
      [
        "retrieve",
        [attemptId],
        { ...fixtures.attempt, record: { digest: "d" } },
        "GET",
        `/v1/jobs/${attemptId}`,
      ],
      ["cancel", [attemptId], fixtures.attempt, "POST", `/v1/jobs/${attemptId}/cancel`],
    ];
    for (const [operation, args, body, method, path] of answers) {
      const { fetcher, sent } = answering(200, body);
      const client = new LocalJobClient(BASE, TOKEN, fetcher);
      const call = client[operation] as (...values: unknown[]) => Promise<{ ok: boolean }>;
      const result = await call.apply(client, args);
      expect(result.ok, operation).toBe(true);
      const first = sent[0];
      if (first === undefined) throw new Error(`${operation} sent nothing`);
      expect(first.url).toBe(`${BASE}${path}`);
      expect(first.init.method).toBe(method);
      expect((first.init.headers as Record<string, string>)[TOKEN_HEADER]).toBe(TOKEN);
    }
    const { fetcher } = answering(200, { ...fixtures.attempt, record: null });
    const retrieved = await new LocalJobClient(BASE, TOKEN, fetcher).retrieve(attemptId);
    expect(retrieved.ok && retrieved.value.record).toBeNull();
  });

  it("reports refusals, unreachable services and malformed answers", async () => {
    const cases: [Fetch, string][] = [
      [
        answering(409, { error: "idempotency_conflict", message: "used" }).fetcher,
        "idempotency_conflict",
      ],
      [answering(500, ["not", "an", "object"]).fetcher, "http_error"],
      [answering(503, null).fetcher, "http_error"],
      [answering(200, "not json").fetcher, "invalid_answer"],
      [
        answering(200, { ...fixtures.attempt, state: "paused", created: true }).fetcher,
        "invalid_answer",
      ],
      [
        async () => {
          throw new TypeError("connection refused");
        },
        "unreachable",
      ],
    ];
    for (const [fetcher, code] of cases) {
      const result = await new LocalJobClient(BASE, TOKEN, fetcher).submit(request);
      expect(result.ok).toBe(false);
      if (!result.ok) expect(result.code).toBe(code);
    }
    const local = await new LocalJobClient(BASE, TOKEN, answering(200, {}).fetcher).plan({
      ...copy(request),
      project: "../escape",
    });
    expect(local).toEqual({
      ok: false,
      code: "invalid_request",
      message: "project must be a short identifier",
    });
    const unsent = await new LocalJobClient(BASE, TOKEN, answering(200, {}).fetcher).submit({
      ...copy(request),
      parameters: [] as unknown as Json,
    });
    expect(unsent.ok).toBe(false);
  });
});
