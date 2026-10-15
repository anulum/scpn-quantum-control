// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native provider observation admission tests

import { expect, it } from "vitest";
import recorded from "../../../../../data/studio/provider_observation_cases.json";
import { canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson, writeJson } from "../../../shared/contracts/jsonTransport";
import {
  MAX_OBSERVATION_BYTES,
  PROVIDER_OBSERVATION_SCHEMA,
  isProviderObservationRefresh,
  parseProviderObservation,
} from "./providerObservation";
import type { ProviderObservation } from "./providerObservation";

async function admitted(raw: string): Promise<ProviderObservation> {
  const result = await parseProviderObservation(raw);
  if (!result.ok) throw new Error(result.message);
  return result.value;
}

it.each(Object.entries(recorded.cases))(
  "admits original native capture %s without numeric loss",
  async (name, raw) => {
    const result = await admitted(raw);
    expect(result.text).toBe(raw);
    expect(result.body.shots).toBe(16n);
    expect(result.body.circuits).toBe(1n);
    expect(result.body.billing).toBe("unknown");
    expect(Object.isFrozen(result.body.events)).toBe(true);
    expect(Object.isFrozen(result.body.events[0]?.observation)).toBe(true);
    if (name === "late_partial") {
      expect(result.body.local_state).toBe("completed");
      expect(result.body.events.at(-1)?.observation).toEqual({ partial_results: [] });
      const complete = result.body.events.find((event) =>
        Object.hasOwn(event.observation, "results"),
      );
      expect(complete?.observation["results"]).toMatchObject([{ counts: { "1": 16n } }]);
    }
  },
);

it("compares actual same-job refreshes, stale snapshots and distinct original identities", async () => {
  const unknown = await admitted(recorded.cases.submission_unknown);
  const submitted = await admitted(recorded.cases.submitted);
  const cancel = await admitted(recorded.cases.cancellation_requested);
  const complete = await admitted(recorded.cases.completed);
  const late = await admitted(recorded.cases.late_partial);
  expect(isProviderObservationRefresh(unknown, unknown)).toBe(true);
  expect(isProviderObservationRefresh(submitted, cancel)).toBe(true);
  expect(isProviderObservationRefresh(cancel, complete)).toBe(true);
  expect(isProviderObservationRefresh(complete, late)).toBe(true);
  expect(isProviderObservationRefresh(late, late)).toBe(true);
  expect(isProviderObservationRefresh(unknown, complete)).toBe(false);
  expect(isProviderObservationRefresh(late, complete)).toBe(false);
});

it.each([
  "attempt_id",
  "target",
  "payload_sha256",
  "payload_base64",
  "shots",
  "circuits",
  "experiment",
  "provider_job_id",
] as const)("refuses refresh of changed original %s", async (key) => {
  const original = await admitted(recorded.cases.completed);
  const value = original.body[key];
  const next = {
    ...original,
    body: { ...original.body, [key]: typeof value === "bigint" ? value + 1n : `${value}-changed` },
  };
  expect(isProviderObservationRefresh(original, next)).toBe(false);
});

it("refuses rewritten history of both different and equal canonical byte length", async () => {
  const original = await admitted(recorded.cases.completed);
  const events = original.body.events;
  const changedLength = {
    ...original,
    body: {
      ...original.body,
      events: events.map((event, index) =>
        index === 0 ? { ...event, observation: { rewritten: true } } : event,
      ),
    },
  };
  expect(isProviderObservationRefresh(original, changedLength)).toBe(false);
  const changedByte = {
    ...original,
    body: {
      ...original.body,
      events: events.map((event, index) => (index === 0 ? { ...event, state: "prepareX" } : event)),
    },
  };
  expect(isProviderObservationRefresh(original, changedByte)).toBe(false);
});

type Wire = Record<string, unknown>;
type Mutation = (wire: Wire, body: Wire, events: Wire[]) => void;
const malformed: [string, Mutation][] = [
  [
    "future envelope",
    (wire) => {
      wire["schema"] = "studio.provider-observation.v2";
    },
  ],
  [
    "unknown extension",
    (wire) => {
      wire["extensions"] = { future: true };
    },
  ],
  [
    "nonobject body",
    (wire) => {
      wire["body"] = [];
    },
  ],
  [
    "missing body field",
    (_wire, body) => {
      delete body["target"];
    },
  ],
  [
    "substituted body field",
    (_wire, body) => {
      delete body["target"];
      body["other"] = "foreign";
    },
  ],
  [
    "future journal",
    (_wire, body) => {
      body["journal_schema"] = "provider_job_journal.v2";
    },
  ],
  [
    "invalid UUID",
    (_wire, body) => {
      body["attempt_id"] = "foreign";
    },
  ],
  [
    "empty target",
    (_wire, body) => {
      body["target"] = "";
    },
  ],
  [
    "oversized label",
    (_wire, body) => {
      body["experiment"] = "x".repeat(1025);
    },
  ],
  [
    "nontext label",
    (_wire, body) => {
      body["experiment"] = true;
    },
  ],
  [
    "zero shots",
    (_wire, body) => {
      body["shots"] = 0n;
    },
  ],
  [
    "floating shots",
    (_wire, body) => {
      body["shots"] = 16.0;
    },
  ],
  [
    "overflow shots",
    (_wire, body) => {
      body["shots"] = 2n ** 63n;
    },
  ],
  [
    "oversized batch",
    (_wire, body) => {
      body["circuits"] = 257n;
    },
  ],
  [
    "unknown state",
    (_wire, body) => {
      body["local_state"] = "retries_allowed";
    },
  ],
  [
    "empty handle",
    (_wire, body) => {
      body["provider_job_id"] = "";
    },
  ],
  [
    "empty provider state",
    (_wire, body) => {
      body["provider_state"] = "";
    },
  ],
  [
    "invented billing",
    (_wire, body) => {
      body["billing"] = "free";
    },
  ],
  [
    "execution request",
    (_wire, body) => {
      body["no_submit"] = false;
    },
  ],
  [
    "empty payload",
    (_wire, body) => {
      body["payload_base64"] = "";
    },
  ],
  [
    "nontext payload",
    (_wire, body) => {
      body["payload_base64"] = 1n;
    },
  ],
  [
    "invalid base64",
    (_wire, body) => {
      body["payload_base64"] = "?";
    },
  ],
  [
    "noncanonical base64",
    (_wire, body) => {
      body["payload_base64"] = "YQ";
    },
  ],
  [
    "invalid payload digest",
    (_wire, body) => {
      body["payload_sha256"] = "bad";
    },
  ],
  [
    "changed payload digest",
    (_wire, body) => {
      body["payload_sha256"] = "0".repeat(64);
    },
  ],
  [
    "absent history",
    (_wire, body) => {
      body["events"] = null;
    },
  ],
  [
    "empty history",
    (_wire, body) => {
      body["events"] = [];
    },
  ],
  [
    "zero sequence",
    (_wire, _body, events) => {
      const first = events[0];
      if (first) first["sequence"] = 0n;
    },
  ],
  [
    "reordered events",
    (_wire, _body, events) => {
      events.reverse();
    },
  ],
  [
    "unknown event state",
    (_wire, _body, events) => {
      const first = events[0];
      if (first) first["state"] = "foreign";
    },
  ],
  [
    "nonobject observation",
    (_wire, _body, events) => {
      const first = events[0];
      if (first) first["observation"] = null;
    },
  ],
  [
    "malformed provider status",
    (_wire, _body, events) => {
      const first = events[0];
      if (first) first["observation"] = { provider_status: 1n };
    },
  ],
  [
    "mismatched local state",
    (_wire, body) => {
      body["local_state"] = "submitted";
    },
  ],
  [
    "mismatched provider state",
    (_wire, body) => {
      body["provider_state"] = "RUNNING";
    },
  ],
];
it.each(malformed)(
  "refuses %s without modifying the original native capture",
  async (_name, change) => {
    const original = recorded.cases.completed;
    const wire = readJson(original) as Wire;
    const body = wire["body"] as Wire;
    change(wire, body, body["events"] as Wire[]);
    wire["sha256"] = await canonicalDigest(PROVIDER_OBSERVATION_SCHEMA, {
      schema: wire["schema"],
      body: wire["body"],
      extensions: wire["extensions"],
    });
    expect((await parseProviderObservation(writeJson(wire))).ok).toBe(false);
    expect((await parseProviderObservation(original)).ok).toBe(true);
  },
);

it.each(["null", "[]", "{}", "{"])("refuses malformed wire %s", async (raw) => {
  expect((await parseProviderObservation(raw)).ok).toBe(false);
});

it("refuses changed content identity and both character and UTF-8 budget overflow", async () => {
  const wire = readJson(recorded.cases.completed) as Wire;
  wire["sha256"] = "0".repeat(64);
  expect((await parseProviderObservation(writeJson(wire))).ok).toBe(false);
  wire["sha256"] = 1n;
  expect((await parseProviderObservation(writeJson(wire))).ok).toBe(false);
  expect((await parseProviderObservation(" ".repeat(MAX_OBSERVATION_BYTES + 1))).ok).toBe(false);
  expect((await parseProviderObservation(`"${"é".repeat(MAX_OBSERVATION_BYTES / 2)}"`)).ok).toBe(
    false,
  );
});
