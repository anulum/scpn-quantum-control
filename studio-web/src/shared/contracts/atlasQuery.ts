// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — bounded, deterministic Atlas queries (browser counterpart)

import {
  type AtlasEdge,
  type AtlasNode,
  type AtlasSnapshot,
  atlasEdgeKinds,
  atlasNodeKinds,
  atlasSnapshotDigest,
  projectAtlasPublic,
} from "./atlas";
import { canonicalDigest } from "./canonical";

/** Query document schema shared with the Python query owner. */
export const atlasQuerySchema = "quantum_atlas_query.v1";
/** Largest admitted hop bound. */
export const atlasMaxDepth = 6;
/** Largest admitted page size. */
export const atlasMaxPageSize = 500;
/** Most edges one answer carries. */
export const atlasMaxVisibleEdges = 2000;
/** Largest snapshot, in nodes, that is queried. */
export const atlasNodeCeiling = 10_000;
/** Largest snapshot, in edges, that is queried. */
export const atlasEdgeCeiling = 100_000;

const nodeStatuses = ["current", "stale", "incomplete"];

/** One bounded query; omitted fields take the Python owner's defaults. */
export interface AtlasQuery {
  /** Lower-cased substring matched against node ID and label. */
  readonly text?: string | null;
  /** Node kinds to keep. */
  readonly kinds?: readonly string[];
  /** Node statuses to keep. */
  readonly statuses?: readonly string[];
  /** Canonical edge kinds to traverse. */
  readonly edgeKinds?: readonly string[];
  /** Start node of a neighbourhood or path query. */
  readonly start?: string | null;
  /** Target node of a path query. */
  readonly target?: string | null;
  /** Traversal direction of a neighbourhood query. */
  readonly direction?: string;
  /** Hop bound. */
  readonly depth?: number;
  /** Items per page. */
  readonly pageSize?: number;
  /** Cursor of the previous page of the same query. */
  readonly cursor?: string | null;
}

/** One page of an answer, equal to the Python owner's result. */
export interface AtlasQueryAnswer {
  /** Digest of the snapshot view the answer was computed on. */
  readonly snapshotDigest: string;
  /** Access scope the query ran under. */
  readonly accessScope: string;
  /** Query shape. */
  readonly shape: "lookup" | "neighbourhood" | "path";
  /** Nodes of this page. */
  readonly nodes: readonly AtlasNode[];
  /** Edges among this page's nodes, in canonical order. */
  readonly edges: readonly AtlasEdge[];
  /** Node-ID sequences of this page, for a path query. */
  readonly paths: readonly (readonly string[])[];
  /** Admitted items of the whole answer. */
  readonly total: number;
  /** Admitted items after this page. */
  readonly omitted: number;
  /** Admitted edges among this page's nodes left out by the edge limit. */
  readonly omittedEdges: number;
  /** Cursor of the next page, or null on the last page. */
  readonly nextCursor: string | null;
  /** What the answer was computed from and what it does not establish. */
  readonly explanation: readonly string[];
}

/** Outcome of a query: the answer, or the refusal with the Python owner's code and field. */
export type AtlasQueryOutcome =
  | {
      /** The query was answered. */
      readonly ok: true;
      /** The answer page. */
      readonly value: AtlasQueryAnswer;
    }
  | {
      /** The query was refused. */
      readonly ok: false;
      /** Stable refusal code. */
      readonly code: string;
      /** Query field the refusal concerns. */
      readonly field: string;
      /** Human-readable reason. */
      readonly message: string;
    };

class QueryRefusal extends Error {
  constructor(
    readonly code: string,
    readonly field: string,
    message: string,
  ) {
    super(message);
  }
}

type Key = string | readonly string[];
type Adjacency = Map<string, [string, AtlasEdge][]>;

function codePoints(value: string): number[] {
  return Array.from(value, (character) => character.codePointAt(0) as number);
}
function compareText(left: string, right: string): number {
  const a = codePoints(left);
  const b = codePoints(right);
  const shared = Math.min(a.length, b.length);
  for (let index = 0; index < shared; index += 1) {
    const x = a[index] as number;
    const y = b[index] as number;
    if (x !== y) return x - y;
  }
  return a.length - b.length;
}
/** Order same-length ID sequences lexicographically; NUL sorts below every admitted ID character. */
function comparePaths(left: readonly string[], right: readonly string[]): number {
  return compareText(left.join("\u0000"), right.join("\u0000"));
}
function sameKey(left: unknown, right: Key): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

interface Normal {
  readonly text: string | null;
  readonly kinds: readonly string[];
  readonly statuses: readonly string[];
  readonly edgeKinds: readonly string[];
  readonly wire: Record<string, unknown>;
}

function normalise(query: AtlasQuery): Normal {
  const checks: [string, readonly string[], readonly string[], string][] = [
    ["kinds", query.kinds ?? [], atlasNodeKinds, "unknown_node_kind"],
    ["statuses", query.statuses ?? [], nodeStatuses, "unknown_status"],
    ["edge_kinds", query.edgeKinds ?? [], atlasEdgeKinds, "unknown_relation"],
  ];
  for (const [field, values, choices, code] of checks) {
    const unknown = [...new Set(values)]
      .filter((value) => !choices.includes(value))
      .sort(compareText);
    if (unknown.length > 0) throw new QueryRefusal(code, field, `'${unknown[0]}' is not admitted`);
  }
  const direction = query.direction ?? "out";
  if (!["out", "in", "both"].includes(direction))
    throw new QueryRefusal("invalid_direction", "direction", `'${direction}'`);
  const depth = query.depth ?? 1;
  if (!Number.isInteger(depth) || depth < 1 || depth > atlasMaxDepth) {
    throw new QueryRefusal("oversized_traversal", "depth", `depth must be 1 to ${atlasMaxDepth}`);
  }
  const pageSize = query.pageSize ?? atlasMaxPageSize;
  if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > atlasMaxPageSize) {
    throw new QueryRefusal(
      "oversized_page",
      "page_size",
      `page size must be 1 to ${atlasMaxPageSize}`,
    );
  }
  const start = query.start ?? null;
  const target = query.target ?? null;
  if (target !== null && start === null)
    throw new QueryRefusal("missing_start", "start", "a path query needs a start node");
  const sortedSet = (values: readonly string[] | undefined): string[] =>
    [...new Set(values ?? [])].sort(compareText);
  const text = query.text === undefined || query.text === null ? null : query.text.toLowerCase();
  const kinds = sortedSet(query.kinds);
  const statuses = sortedSet(query.statuses);
  const edgeKinds = sortedSet(query.edgeKinds);
  return {
    text,
    kinds,
    statuses,
    edgeKinds,
    wire: {
      schema: atlasQuerySchema,
      text,
      kinds,
      statuses,
      edge_kinds: edgeKinds,
      start,
      target,
      direction,
      // Integers are BigInt so the canonical codec tags them as integers, as Python does.
      depth: BigInt(depth),
      page_size: BigInt(pageSize),
    },
  };
}

function matches(node: AtlasNode, normal: Normal): boolean {
  if (
    normal.text !== null &&
    !node.id.toLowerCase().includes(normal.text) &&
    !node.label.toLowerCase().includes(normal.text)
  ) {
    return false;
  }
  if (normal.kinds.length > 0 && !normal.kinds.includes(node.kind)) return false;
  return normal.statuses.length === 0 || normal.statuses.includes(node.status);
}

function adjacency(
  view: AtlasSnapshot,
  edgeKinds: readonly string[],
  direction: string,
): Adjacency {
  const result: Adjacency = new Map(view.nodes.map((node) => [node.id, []]));
  for (const edge of view.edges) {
    if (edgeKinds.length > 0 && !edgeKinds.includes(edge.kind)) continue;
    if (direction !== "in") result.get(edge.source)?.push([edge.target, edge]);
    if (direction !== "out") result.get(edge.target)?.push([edge.source, edge]);
  }
  return result;
}

function distances(graph: Adjacency, start: string, depth: number): Map<string, number> {
  const distance = new Map<string, number>([[start, 0]]);
  const queue = [start];
  for (let head = 0; head < queue.length; head += 1) {
    const current = queue[head] as string;
    const here = distance.get(current) as number;
    if (here === depth) continue;
    for (const [neighbour] of graph.get(current) as [string, AtlasEdge][]) {
      if (!distance.has(neighbour)) {
        distance.set(neighbour, here + 1);
        queue.push(neighbour);
      }
    }
  }
  return distance;
}

function shortestPaths(graph: Adjacency, start: string, target: string, depth: number): string[][] {
  const distance = distances(graph, start, depth);
  const goal = distance.get(target);
  if (goal === undefined) return [];
  const paths = new Map<string, string[]>();
  const extend = (path: string[]): void => {
    const current = path[path.length - 1] as string;
    if (current === target) {
      paths.set(JSON.stringify(path), path);
      return;
    }
    const next = (distance.get(current) as number) + 1;
    for (const [neighbour] of graph.get(current) as [string, AtlasEdge][]) {
      const at = distance.get(neighbour);
      if (at === next && at <= goal) extend([...path, neighbour]);
    }
  };
  extend([start]);
  return [...paths.values()].sort(comparePaths);
}

function encodeCursor(binding: Record<string, string>, last: Key): string {
  const document = {
    access: binding.access,
    last,
    query: binding.query,
    snapshot: binding.snapshot,
  };
  const raw = JSON.stringify(document);
  return btoa(raw).replaceAll("+", "-").replaceAll("/", "_");
}

function decodeCursor(cursor: string | null | undefined, binding: Record<string, string>): unknown {
  if (cursor === undefined || cursor === null) return null;
  let decoded: unknown;
  try {
    decoded = JSON.parse(atob(cursor.replaceAll("-", "+").replaceAll("_", "/")));
  } catch {
    throw new QueryRefusal("invalid_cursor", "cursor", "not an Atlas cursor");
  }
  if (
    typeof decoded !== "object" ||
    decoded === null ||
    Array.isArray(decoded) ||
    !("last" in decoded)
  ) {
    throw new QueryRefusal("invalid_cursor", "cursor", "not an Atlas cursor");
  }
  const { last, ...rest } = decoded as Record<string, unknown>;
  const expected = Object.keys(binding).sort();
  const actual = Object.keys(rest).sort();
  if (!sameKey(actual, expected) || expected.some((key) => rest[key] !== binding[key])) {
    throw new QueryRefusal(
      "cursor_mismatch",
      "cursor",
      "the cursor belongs to another snapshot, query or access scope",
    );
  }
  return last;
}

// Validated snapshots are immutable, so each one's digest and public projection are computed
// once; a repeated query then costs its filtering only.
const digests = new WeakMap<AtlasSnapshot, Promise<string>>();
const publicViews = new WeakMap<AtlasSnapshot, AtlasSnapshot>();

/**
 * Return a validated snapshot's canonical digest, computed once per snapshot.
 *
 * @param snapshot - A validated snapshot, which is never changed afterwards.
 * @returns Lowercase SHA-256 hex of the canonical wire document.
 */
export function rememberedAtlasSnapshotDigest(snapshot: AtlasSnapshot): Promise<string> {
  let digest = digests.get(snapshot);
  if (digest === undefined) {
    digest = atlasSnapshotDigest(snapshot);
    digests.set(snapshot, digest);
  }
  return digest;
}

/** A validated public snapshot holds nothing private, so it is its own public projection. */
function publicView(snapshot: AtlasSnapshot): AtlasSnapshot {
  if (snapshot.accessScope === "public") return snapshot;
  let view = publicViews.get(snapshot);
  if (view === undefined) {
    view = projectAtlasPublic(snapshot);
    publicViews.set(snapshot, view);
  }
  return view;
}

async function answer(
  snapshot: AtlasSnapshot | null,
  query: AtlasQuery,
  access: string,
): Promise<AtlasQueryAnswer> {
  if (snapshot === null)
    throw new QueryRefusal("absent_snapshot", "snapshot", "no snapshot to query");
  if (snapshot.nodes.length > atlasNodeCeiling || snapshot.edges.length > atlasEdgeCeiling) {
    throw new QueryRefusal(
      "snapshot_too_large",
      "snapshot",
      "above the admitted node or edge ceiling",
    );
  }
  if (access !== "public" && (access !== "private" || snapshot.accessScope !== "private"))
    throw new QueryRefusal(
      "access_refused",
      "access",
      `'${access}' is not admitted for this snapshot`,
    );
  const view = access === "public" ? publicView(snapshot) : snapshot;
  const normal = normalise(query);
  const byId = new Map(view.nodes.map((node) => [node.id, node]));
  for (const [field, id] of [
    ["start", query.start ?? null],
    ["target", query.target ?? null],
  ] as const) {
    if (id !== null && !byId.has(id))
      throw new QueryRefusal("unknown_node", field, "the node is not in the admitted snapshot");
  }
  const digest = await rememberedAtlasSnapshotDigest(view);
  const binding = {
    snapshot: digest,
    access,
    query: await canonicalDigest(atlasQuerySchema, normal.wire),
  };
  const last = decodeCursor(query.cursor, binding);
  const depth = query.depth ?? 1;
  const explanation = [
    `snapshot ${digest} of commit ${view.sourceCommit}, access ${access}`,
    `${view.exclusions.length} documented exclusions are not part of the graph`,
  ];
  let shape: AtlasQueryAnswer["shape"];
  let ordered: Key[];
  const start = query.start ?? null;
  const target = query.target ?? null;
  const direction = query.direction ?? "out";
  if (start === null) {
    shape = "lookup";
    ordered = view.nodes.filter((node) => matches(node, normal)).map((node) => node.id);
    explanation.push("nodes matching every given filter, ordered by ID");
  } else if (target === null) {
    shape = "neighbourhood";
    const distance = distances(adjacency(view, normal.edgeKinds, direction), start, depth);
    ordered = [...distance.keys()]
      .sort(
        (left, right) =>
          (distance.get(left) as number) - (distance.get(right) as number) ||
          compareText(left, right),
      )
      .filter((id) => matches(byId.get(id) as AtlasNode, normal));
    explanation.push(`${direction} neighbourhood within ${depth} hops, ordered by distance and ID`);
  } else {
    shape = "path";
    ordered = shortestPaths(adjacency(view, normal.edgeKinds, "out"), start, target, depth);
    explanation.push(
      `every shortest directed path within ${depth} hops; a navigation path, not a proof`,
    );
  }
  explanation.push(
    `edge kinds ${normal.edgeKinds.length > 0 ? normal.edgeKinds.join(", ") : "all admitted kinds"}`,
  );
  let position = 0;
  if (last !== null) {
    const index = ordered.findIndex((key) => sameKey(last, key));
    if (index < 0)
      throw new QueryRefusal(
        "cursor_mismatch",
        "cursor",
        "the cursor's last item is not in this answer",
      );
    position = index + 1;
  }
  const pageSize = query.pageSize ?? atlasMaxPageSize;
  const page = ordered.slice(position, position + pageSize);
  const remaining = ordered.length - position - page.length;
  const paths = shape === "path" ? (page as string[][]) : [];
  const ids = shape === "path" ? [...new Set(paths.flat())].sort(compareText) : (page as string[]);
  const members = new Set(ids);
  const inner = view.edges.filter(
    (edge) =>
      members.has(edge.source) &&
      members.has(edge.target) &&
      (normal.edgeKinds.length === 0 || normal.edgeKinds.includes(edge.kind)),
  );
  const lastItem = page[page.length - 1];
  return {
    snapshotDigest: digest,
    accessScope: access,
    shape,
    nodes: ids.map((id) => byId.get(id) as AtlasNode),
    edges: inner.slice(0, atlasMaxVisibleEdges),
    paths,
    total: ordered.length,
    omitted: remaining,
    omittedEdges: Math.max(0, inner.length - atlasMaxVisibleEdges),
    nextCursor: remaining > 0 && lastItem !== undefined ? encodeCursor(binding, lastItem) : null,
    explanation,
  };
}

/**
 * Answer one page of a bounded query exactly as the Python query owner does.
 *
 * @param snapshot - A validated snapshot, or null.
 * @param query - The query.
 * @param access - `"public"` runs on the public projection; `"private"` needs a private-scope snapshot.
 * @returns The answer page, or the refusal with the Python owner's code and field.
 */
export async function runAtlasQuery(
  snapshot: AtlasSnapshot | null,
  query: AtlasQuery,
  access = "public",
): Promise<AtlasQueryOutcome> {
  try {
    return { ok: true, value: await answer(snapshot, query, access) };
  } catch (error) {
    if (error instanceof QueryRefusal)
      return { ok: false, code: error.code, field: error.field, message: error.message };
    throw error;
  }
}
