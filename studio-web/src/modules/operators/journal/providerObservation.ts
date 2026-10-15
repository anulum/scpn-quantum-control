// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original provider observation admission

import { canonicalBytes, canonicalDigest } from "../../../shared/contracts/canonical";
import { readJson } from "../../../shared/contracts/jsonTransport";

/** Exact domain of native read-only journal exports. */
export const PROVIDER_OBSERVATION_SCHEMA = "studio.provider-observation.v1";
/** UTF-8 ceiling shared with the native producer; original journals are never truncated. */
export const MAX_OBSERVATION_BYTES = 4 * 1024 * 1024;
/** Original journal states, separate from provider transport status. */
const states = [
  "prepared",
  "submission_unknown",
  "submitted",
  "cancellation_requested",
  "completed",
  "cancelled",
  "failed",
] as const;

/** An original append-only journal observation, without invented timestamps. */
export interface ProviderObservationEvent {
  /** Original global journal sequence; gaps across attempts are allowed. */ readonly sequence: bigint;
  /** Effective local state at this original event. */ readonly state: string;
  /** Original raw SDK, partial-result or status record. */ readonly observation: Readonly<
    Record<string, unknown>
  >;
}
/** Exact native request and detached observation values. */
export interface ProviderObservationBody {
  /** Original durable companion version. */ readonly journal_schema: string;
  /** Canonical UUID of this one effect. */ readonly attempt_id: string;
  /** Original selected backend; never replaced during refresh. */ readonly target: string;
  /** Digest of the original native request bytes. */ readonly payload_sha256: string;
  /** Exact native request bytes retained as base64. */ readonly payload_base64: string;
  /** Original integral shots per circuit. */ readonly shots: bigint;
  /** Original ordered native batch size. */ readonly circuits: bigint;
  /** Original caller label. */ readonly experiment: string;
  /** Current local lifecycle annotation. */ readonly local_state: string;
  /** Original provider handle, or unknown until explicit reconciliation. */ readonly provider_job_id:
    | string
    | null;
  /** Latest explicit provider status; null when the source retained none. */ readonly provider_state:
    | string
    | null;
  /** The journal does not prove provider billing. */ readonly billing: "unknown";
  /** Complete ordered original evidence history. */ readonly events: readonly ProviderObservationEvent[];
  /** This export confers no submission authority. */ readonly no_submit: true;
}
/** Immutable admitted values and their original lossless wire text. */
export interface ProviderObservation {
  /** Original export for exact read-only download. */ readonly text: string;
  /** Domain-separated canonical content identity, not provider authentication. */ readonly sha256: string;
  /** Source-owned values. */ readonly body: ProviderObservationBody;
}
/** Refusal preserves the caller's previous admitted observation. */
export type ProviderObservationResult =
  | {
      /** Successful content admission. */ readonly ok: true;
      /** Original source values. */
      readonly value: ProviderObservation;
    }
  | {
      /** Failed content admission. */ readonly ok: false;
      /** Observable refusal. */
      readonly message: string;
    };

function require(value: boolean): void {
  if (!value) throw new Error("Provider observation refused");
}
function object(value: unknown): Record<string, unknown> {
  require(typeof value === "object" && value !== null && !Array.isArray(value));
  return value as Record<string, unknown>;
}
function fields(value: unknown, names: readonly string[]): Record<string, unknown> {
  const result = object(value);
  require(
    Object.keys(result).length === names.length && names.every((key) => Object.hasOwn(result, key)),
  );
  return result;
}
function text(value: unknown): string {
  require(typeof value === "string" && [...value].length > 0 && [...value].length <= 1024);
  return value as string;
}
function hash(value: unknown): string {
  const result = text(value);
  require(/^[0-9a-f]{64}$/.test(result));
  return result;
}
function state(value: unknown): string {
  const result = text(value);
  require(states.some((item) => item === result));
  return result;
}
function count(value: unknown, maximum: bigint): bigint {
  require(typeof value === "bigint" && value > 0n && value <= maximum);
  return value as bigint;
}
function freeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

/** Admit native content identity and preserve all original evidence; no provider calls occur. */
export async function parseProviderObservation(raw: string): Promise<ProviderObservationResult> {
  try {
    require(
      raw.length <= MAX_OBSERVATION_BYTES &&
        new TextEncoder().encode(raw).length <= MAX_OBSERVATION_BYTES,
    );
    const wire = fields(readJson(raw), ["schema", "body", "extensions", "sha256"]);
    require(wire["schema"] === PROVIDER_OBSERVATION_SCHEMA);
    fields(wire["extensions"], []);
    const body = fields(wire["body"], [
      "journal_schema",
      "attempt_id",
      "target",
      "payload_sha256",
      "payload_base64",
      "shots",
      "circuits",
      "experiment",
      "local_state",
      "provider_job_id",
      "provider_state",
      "billing",
      "events",
      "no_submit",
    ]);
    require(body["journal_schema"] === "provider_job_journal.v1");
    const identity = text(body["attempt_id"]);
    require(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(identity));
    text(body["target"]);
    text(body["experiment"]);
    count(body["shots"], 2n ** 63n - 1n);
    count(body["circuits"], 256n);
    state(body["local_state"]);
    if (body["provider_job_id"] !== null) text(body["provider_job_id"]);
    if (body["provider_state"] !== null) text(body["provider_state"]);
    require(body["billing"] === "unknown" && body["no_submit"] === true);
    const encoded = body["payload_base64"];
    require(typeof encoded === "string" && encoded.length > 0);
    const decoded = atob(encoded as string);
    require(btoa(decoded) === encoded);
    const payload = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
    const digest = await crypto.subtle.digest("SHA-256", payload);
    const payloadHash = Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    require(hash(body["payload_sha256"]) === payloadHash);
    const events = body["events"];
    require(Array.isArray(events) && events.length > 0);
    let sequence = 0n;
    let providerState: unknown = null;
    let localState = "";
    for (const original of events as unknown[]) {
      const event = fields(original, ["sequence", "state", "observation"]);
      const next = count(event["sequence"], 2n ** 63n - 1n);
      require(next > sequence);
      sequence = next;
      localState = state(event["state"]);
      const observation = object(event["observation"]);
      if (Object.hasOwn(observation, "provider_status"))
        providerState = text(observation["provider_status"]);
    }
    require(localState === body["local_state"] && providerState === body["provider_state"]);
    const envelopeHash = hash(wire["sha256"]);
    require(
      (await canonicalDigest(PROVIDER_OBSERVATION_SCHEMA, {
        schema: wire["schema"],
        body,
        extensions: wire["extensions"],
      })) === envelopeHash,
    );
    return {
      ok: true,
      value: freeze({
        text: raw,
        sha256: envelopeHash,
        body: body as unknown as ProviderObservationBody,
      }),
    };
  } catch {
    return {
      ok: false,
      message:
        "Provider observation refused; the previous admitted job and evidence remain unchanged.",
    };
  }
}

/** Admit an append-only refresh of the same immutable request and original provider identity. */
export function isProviderObservationRefresh(
  previous: ProviderObservation,
  next: ProviderObservation,
): boolean {
  const before = previous.body,
    after = next.body;
  for (const key of [
    "attempt_id",
    "target",
    "payload_sha256",
    "payload_base64",
    "shots",
    "circuits",
    "experiment",
  ] as const) {
    if (before[key] !== after[key]) return false;
  }
  if (before.provider_job_id !== null && before.provider_job_id !== after.provider_job_id)
    return false;
  if (after.events.length < before.events.length) return false;
  return before.events.every((event, index) => {
    const original = canonicalBytes("studio.provider-event.v1", event);
    const refreshed = canonicalBytes("studio.provider-event.v1", after.events[index]);
    return (
      original.length === refreshed.length &&
      original.every((byte, offset) => byte === refreshed[offset])
    );
  });
}
