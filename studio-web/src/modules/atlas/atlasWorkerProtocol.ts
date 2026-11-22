// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — messages between the Atlas view and its snapshot worker

import type { AtlasQuery, AtlasQueryOutcome } from "../../shared/contracts/atlasQuery";

/** What an admitted snapshot is, without its nodes and relations. */
export interface AtlasSnapshotSummary {
  /** Canonical digest recomputed on admission. */
  readonly digest: string;
  /** Full commit the snapshot's source tree is based on. */
  readonly sourceCommit: string;
  /** Nodes of the snapshot. */
  readonly nodes: number;
  /** Relations of the snapshot. */
  readonly edges: number;
}

/** A request to the worker; the worker answers requests in the order they arrive. */
export type AtlasWorkerRequest =
  | {
      /** Admit a snapshot pin; it replaces any snapshot admitted before. */
      readonly type: "open";
      /** Request number the reply carries. */
      readonly id: number;
      /** The pin's text, or a file holding it. */
      readonly source: string | Blob;
    }
  | {
      /** Answer one bounded query on the admitted snapshot. */
      readonly type: "query";
      /** Request number the reply carries. */
      readonly id: number;
      /** The query. */
      readonly query: AtlasQuery;
    };

/** The worker's reply to one request. */
export type AtlasWorkerReply =
  | {
      /** The pin was admitted. */
      readonly type: "opened";
      /** Number of the request answered. */
      readonly id: number;
      /** The admitted snapshot. */
      readonly summary: AtlasSnapshotSummary;
    }
  | {
      /** The pin was refused by the admission checks. */
      readonly type: "refused";
      /** Number of the request answered. */
      readonly id: number;
      /** Fixed explanation of the refusal. */
      readonly reason: string;
    }
  | {
      /** The query was answered or refused by the query owner. */
      readonly type: "answer";
      /** Number of the request answered. */
      readonly id: number;
      /** The query owner's outcome. */
      readonly outcome: AtlasQueryOutcome;
    }
  | {
      /** The request could not be carried out. */
      readonly type: "failed";
      /** Number of the request answered, or -1 for a request that could not be read. */
      readonly id: number;
      /** What went wrong. */
      readonly reason: string;
    };

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function requestId(value: unknown): number | null {
  return Number.isSafeInteger(value) && (value as number) >= 0 ? (value as number) : null;
}

/**
 * Read a request the worker received.
 *
 * @param data - The structured-cloned message.
 * @returns The request, or null when it is not one.
 */
export function readAtlasWorkerRequest(data: unknown): AtlasWorkerRequest | null {
  const message = record(data);
  const id = requestId(message?.["id"]);
  if (message === null || id === null) return null;
  const source = message["source"];
  if (message["type"] === "open" && (typeof source === "string" || source instanceof Blob))
    return { type: "open", id, source };
  const query = record(message["query"]);
  if (message["type"] === "query" && query !== null)
    return { type: "query", id, query: query as AtlasQuery };
  return null;
}

/**
 * Read a reply the view received.
 *
 * @param data - The structured-cloned message.
 * @returns The reply, or null when it is not one.
 */
export function readAtlasWorkerReply(data: unknown): AtlasWorkerReply | null {
  const message = record(data);
  const id = message?.["id"];
  if (message === null || !Number.isSafeInteger(id) || (id as number) < -1) return null;
  const at = id as number;
  const summary = record(message["summary"]);
  if (
    message["type"] === "opened" &&
    typeof summary?.["digest"] === "string" &&
    typeof summary["sourceCommit"] === "string" &&
    Number.isSafeInteger(summary["nodes"]) &&
    Number.isSafeInteger(summary["edges"])
  )
    return { type: "opened", id: at, summary: summary as unknown as AtlasSnapshotSummary };
  const outcome = record(message["outcome"]);
  if (message["type"] === "answer" && typeof outcome?.["ok"] === "boolean")
    return { type: "answer", id: at, outcome: outcome as unknown as AtlasQueryOutcome };
  const reason = message["reason"];
  if ((message["type"] === "refused" || message["type"] === "failed") && typeof reason === "string")
    return { type: message["type"], id: at, reason };
  return null;
}
