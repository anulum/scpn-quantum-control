// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas view model shared by the graph, table and inspector

import {
  type AtlasEdge,
  type AtlasEdgeKind,
  type AtlasNode,
  type AtlasNodeKind,
  atlasNodeKinds,
} from "../../shared/contracts/atlas";
import {
  type AtlasQuery,
  type AtlasQueryAnswer,
  atlasMaxDepth,
} from "../../shared/contracts/atlasQuery";

/** A registered lens. */
export type AtlasLens = "architecture" | "claims" | "evidence";

/** One lens: the label of its link and the node kinds it keeps. */
export interface AtlasLensDefinition {
  /** Label of the lens link. */
  readonly label: string;
  /** Node kinds the lens keeps. */
  readonly kinds: readonly AtlasNodeKind[];
}

/** Lenses are deep-linkable instruments; each keeps a fixed set of node kinds. */
export const atlasLenses: Readonly<Record<AtlasLens, AtlasLensDefinition>> = {
  architecture: {
    label: "Architecture",
    kinds: ["capability", "definition", "test", "documentation"],
  },
  claims: { label: "Claims", kinds: ["claim", "definition", "documentation"] },
  evidence: { label: "Evidence", kinds: ["artefact"] },
};
/** Lens shown when the route names none. */
export const defaultAtlasLens: AtlasLens = "architecture";
/** Nodes shown at once; the query owner's page bound. */
export const atlasVisibleNodes = 500;
/** Rows per page of the node and relation tables, which bounds the rows rendered at once. */
export const atlasTableRows = 100;

/**
 * Return the lens a route instrument names, or the default lens.
 *
 * @param instrument - The route's instrument segment, or null.
 * @returns The registered lens.
 */
export function lensOf(instrument: string | null): AtlasLens {
  return instrument !== null && Object.hasOwn(atlasLenses, instrument)
    ? (instrument as AtlasLens)
    : defaultAtlasLens;
}

/**
 * Build the bounded query a view state asks.
 *
 * @param lens - Active lens.
 * @param text - Search text, or an empty string.
 * @param focus - Focused node for a neighbourhood, or null for the lens lookup.
 * @param depth - Hop bound of the neighbourhood.
 * @returns The query; a focused neighbourhood keeps the lens's node kinds.
 */
export function atlasViewQuery(
  lens: AtlasLens,
  text: string,
  focus: string | null,
  depth: number,
): AtlasQuery {
  return {
    text: text === "" ? null : text,
    kinds: [...atlasLenses[lens].kinds],
    start: focus,
    direction: "both",
    depth,
    pageSize: atlasVisibleNodes,
  };
}

/** View state a deep link restores. */
export interface AtlasFilter {
  /** Search text, or an empty string. */
  readonly text: string;
  /** Focused node of a neighbourhood, or null. */
  readonly focus: string | null;
  /** Hop bound of the neighbourhood. */
  readonly depth: number;
}
/** Filter of a view that a link does not narrow. */
export const defaultAtlasFilter: AtlasFilter = Object.freeze({ text: "", focus: null, depth: 1 });
/** Longest encoded filter a link carries; the workbench router admits no longer context value. */
export const atlasFilterBound = 256;

/** Admitted filter, or the reason a requested one was refused. */
export type AtlasFilterAdmission =
  | {
      /** The filter was admitted. */
      readonly ok: true;
      /** The admitted filter. */
      readonly filter: AtlasFilter;
    }
  | {
      /** The filter was refused; the view opens without it. */
      readonly ok: false;
      /** Fixed explanation of the refusal. */
      readonly reason: string;
    };

const filterFields: readonly string[] = ["q", "focus", "hops"];

/**
 * Admit a requested filter: the fields `q`, `focus` and `hops`, each at most once and non-empty.
 *
 * @param value - The route's encoded filter, or null when the route carries none.
 * @returns The admitted filter, or the refusal reason.
 */
export function parseAtlasFilter(value: string | null): AtlasFilterAdmission {
  if (value === null) return { ok: true, filter: defaultAtlasFilter };
  const seen = new Set<string>();
  let text = "";
  let focus: string | null = null;
  let depth = 1;
  for (const [key, field] of new URLSearchParams(value)) {
    if (!filterFields.includes(key) || seen.has(key))
      return { ok: false, reason: "it has an unknown or repeated field" };
    seen.add(key);
    if (field === "") return { ok: false, reason: `its ${key} field is empty` };
    if (key === "q") text = field;
    else if (key === "focus") focus = field;
    else if (/^[1-9]$/.test(field) && Number(field) <= atlasMaxDepth) depth = Number(field);
    else return { ok: false, reason: `its hops field must be 1 to ${atlasMaxDepth}` };
  }
  return { ok: true, filter: { text, focus, depth } };
}

/**
 * Encode a filter in a stable field order, leaving out fields at their default.
 *
 * @param filter - The view's filter.
 * @returns The encoded filter, or null for the default filter.
 */
export function formatAtlasFilter(filter: AtlasFilter): string | null {
  const fields = new URLSearchParams();
  if (filter.text !== "") fields.set("q", filter.text);
  if (filter.focus !== null) fields.set("focus", filter.focus);
  if (filter.depth !== 1) fields.set("hops", String(filter.depth));
  const encoded = fields.toString();
  return encoded === "" ? null : encoded;
}

/**
 * Build the deep link of a view: its lens, the shown snapshot and its filter.
 *
 * The requested project and revision of the route are kept; a filter longer than
 * {@link atlasFilterBound} cannot be linked.
 *
 * @param lens - The lens to link.
 * @param contextSuffix - The route's encoded context, as the workbench supplies it.
 * @param snapshot - Digest of the snapshot shown.
 * @param filter - The filter to restore.
 * @returns The link, or null when the encoded filter is above the bound.
 */
export function atlasViewHref(
  lens: AtlasLens,
  contextSuffix: string,
  snapshot: string,
  filter: AtlasFilter,
): string | null {
  const requested = new URLSearchParams(contextSuffix.replace(/^\?/, ""));
  const context = new URLSearchParams();
  for (const key of ["project", "revision"]) {
    const value = requested.get(key);
    if (value !== null) context.set(key, value);
  }
  context.set("snapshot", snapshot);
  const encoded = formatAtlasFilter(filter);
  if (encoded !== null && encoded.length > atlasFilterBound) return null;
  if (encoded !== null) context.set("filter", encoded);
  return `#/atlas/${lens}?${context.toString()}`;
}

/** Semantic zoom levels, from kind groups to nodes with their details. */
export const atlasZoomLevels = ["kinds", "nodes", "details"] as const;
/** One semantic zoom level. */
export type AtlasZoom = (typeof atlasZoomLevels)[number];
/** Pages with more nodes than this open on the kind groups. */
export const atlasNodeZoomLimit = 150;

/**
 * Return the zoom level a page opens on when the user has not chosen one.
 *
 * @param answer - The query answer on screen.
 * @returns Kind groups for a page above {@link atlasNodeZoomLimit} nodes, otherwise nodes.
 */
export function automaticAtlasZoom(answer: AtlasQueryAnswer): AtlasZoom {
  return answer.nodes.length > atlasNodeZoomLimit ? "kinds" : "nodes";
}

/** One node-kind group of the aggregated graph. */
export interface AtlasKindGroup {
  /** Node kind. */
  readonly kind: AtlasNodeKind;
  /** Nodes of this kind on the page. */
  readonly count: number;
}

/** Relations of one kind between two node-kind groups. */
export interface AtlasKindLink {
  /** Kind of the source nodes. */
  readonly source: AtlasNodeKind;
  /** Relation kind. */
  readonly kind: AtlasEdgeKind;
  /** Kind of the target nodes. */
  readonly target: AtlasNodeKind;
  /** Relations of this kind on the page. */
  readonly count: number;
}

/** The page aggregated by node kind. */
export interface AtlasAggregate {
  /** Kind groups in the contract's kind order. */
  readonly groups: readonly AtlasKindGroup[];
  /** Aggregated relations ordered by source kind, relation kind and target kind. */
  readonly links: readonly AtlasKindLink[];
}

/**
 * Aggregate the page by node kind; the counts cover the page shown, not the omitted items.
 *
 * @param answer - The query answer on screen.
 * @returns The kind groups and the relations counted between them.
 */
export function aggregateAtlasAnswer(answer: AtlasQueryAnswer): AtlasAggregate {
  const kindOf = new Map(answer.nodes.map((node) => [node.id, node.kind]));
  const groups = atlasNodeKinds
    .map((kind) => ({ kind, count: answer.nodes.filter((node) => node.kind === kind).length }))
    .filter((group) => group.count > 0);
  const counted = new Map<string, AtlasKindLink>();
  for (const edge of answer.edges) {
    const source = kindOf.get(edge.source) as AtlasNodeKind;
    const target = kindOf.get(edge.target) as AtlasNodeKind;
    const key = `${source} ${edge.kind} ${target}`;
    const count = (counted.get(key)?.count ?? 0) + 1;
    counted.set(key, { source, kind: edge.kind, target, count });
  }
  const links = [...counted.keys()].sort().map((key) => counted.get(key) as AtlasKindLink);
  return { groups, links };
}

/** One positioned node of the graph. */
export interface AtlasPlacedNode {
  /** The node. */
  readonly node: AtlasNode;
  /** Horizontal position in view units. */
  readonly x: number;
  /** Vertical position in view units. */
  readonly y: number;
}

/** Layout state, kept apart from the semantic snapshot. */
export interface AtlasLayout {
  /** Placed nodes in answer order. */
  readonly placed: readonly AtlasPlacedNode[];
  /** View-box width. */
  readonly width: number;
  /** View-box height. */
  readonly height: number;
}

const columnWidth = 220;
const rowHeight = 44;

/**
 * Lay the answer out in columns: hop distance from the focus, or node kind for a lookup.
 *
 * @param answer - A query answer.
 * @param focus - The focused node, or null.
 * @param pinned - Positions the user pinned; they override the computed position.
 * @returns Deterministic positions for every answer node.
 */
export function layoutAtlasAnswer(
  answer: AtlasQueryAnswer,
  focus: string | null,
  pinned: ReadonlyMap<string, readonly [number, number]> = new Map(),
): AtlasLayout {
  const column = new Map<string, number>();
  if (focus !== null && answer.nodes.some((node) => node.id === focus)) {
    column.set(focus, 0);
    const frontier = [focus];
    for (let head = 0; head < frontier.length; head += 1) {
      const current = frontier[head] as string;
      for (const edge of answer.edges) {
        const other =
          edge.source === current ? edge.target : edge.target === current ? edge.source : null;
        if (other !== null && !column.has(other)) {
          column.set(other, (column.get(current) as number) + 1);
          frontier.push(other);
        }
      }
    }
  }
  const kinds = [...new Set(answer.nodes.map((node) => node.kind))];
  const rows = new Map<number, number>();
  const placed = answer.nodes.map((node) => {
    const index = column.get(node.id) ?? (focus === null ? kinds.indexOf(node.kind) : kinds.length);
    const row = rows.get(index) ?? 0;
    rows.set(index, row + 1);
    const pin = pinned.get(node.id);
    return { node, x: pin?.[0] ?? 40 + index * columnWidth, y: pin?.[1] ?? 40 + row * rowHeight };
  });
  const width = Math.max(320, ...placed.map((item) => item.x + columnWidth));
  const height = Math.max(160, ...placed.map((item) => item.y + rowHeight));
  return { placed, width, height };
}

/**
 * Return where a navigation key moves the selection.
 *
 * ArrowRight follows the first outgoing edge and ArrowLeft the first incoming edge, by target and
 * source ID; ArrowDown and ArrowUp step through the answer order.
 *
 * @param answer - The query answer on screen.
 * @param selected - Selected node ID, or null.
 * @param key - The pressed key.
 * @returns The next selected node ID, or null when the key moves nowhere.
 */
export function atlasKeyTarget(
  answer: AtlasQueryAnswer,
  selected: string | null,
  key: string,
): string | null {
  const ids = answer.nodes.map((node) => node.id);
  if (ids.length === 0) return null;
  if (selected === null || !ids.includes(selected))
    return key.startsWith("Arrow") ? (ids[0] as string) : null;
  const along = (
    edges: readonly AtlasEdge[],
    end: "source" | "target",
    other: "source" | "target",
  ): string | null =>
    edges
      .filter((edge) => edge[end] === selected)
      .map((edge) => edge[other])
      .sort()[0] ?? null;
  const index = ids.indexOf(selected);
  switch (key) {
    case "ArrowRight":
      return along(answer.edges, "source", "target");
    case "ArrowLeft":
      return along(answer.edges, "target", "source");
    case "ArrowDown":
      return ids[Math.min(index + 1, ids.length - 1)] as string;
    case "ArrowUp":
      return ids[Math.max(index - 1, 0)] as string;
    default:
      return null;
  }
}
