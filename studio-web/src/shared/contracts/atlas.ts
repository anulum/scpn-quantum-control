// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas snapshot ontology and identity contract

import { canonicalDigest } from "./canonical";

/** Snapshot document schema shared with the Python contract owner. */
export const atlasSnapshotSchema = "quantum_atlas_snapshot.v1";
/** The only admitted relation version. */
export const atlasRelationVersion = "atlas_relation.v1";

/** Node kinds, each identified as a definition, revision or instance. */
export const atlasNodeKinds = [
  "definition",
  "instance",
  "revision",
  "parameter",
  "program",
  "backend",
  "run",
  "artefact",
  "claim",
  "test",
  "documentation",
  "capability",
] as const;
/** One Atlas node kind. */
export type AtlasNodeKind = (typeof atlasNodeKinds)[number];
/** How a node is identified: qualified name, content digest or project plus UUID. */
export type AtlasIdentityClass = "definition" | "revision" | "instance";

/** Identity class of every node kind. */
export const atlasIdentityClasses: Readonly<Record<AtlasNodeKind, AtlasIdentityClass>> = {
  definition: "definition",
  parameter: "definition",
  program: "definition",
  backend: "definition",
  claim: "definition",
  test: "definition",
  documentation: "definition",
  capability: "definition",
  revision: "revision",
  artefact: "revision",
  instance: "instance",
  run: "instance",
};

/** The sixteen canonical edge kinds of relation version 1. */
export const atlasEdgeKinds = [
  "implements",
  "consumes",
  "produces",
  "derives_from",
  "constrained_by",
  "lowers_to",
  "supports",
  "requires",
  "verified_by",
  "benchmarked_against",
  "calibrated_by",
  "supersedes",
  "contradicts_under_assumptions",
  "maps_to",
  "contains",
  "documents",
] as const;
/** One canonical Atlas edge kind. */
export type AtlasEdgeKind = (typeof atlasEdgeKinds)[number];

/** Version-1 aliases of the earlier design vocabulary: alias to canonical kind and whether it reverses. */
export const atlasEdgeAliasesV1: Readonly<Record<string, readonly [AtlasEdgeKind, boolean]>> = {
  produced: ["produces", false],
  derived_from: ["derives_from", false],
  verifies: ["verified_by", true],
  depends_on: ["requires", false],
};
/** Spellings that are refused, with the reason. */
export const atlasRefusedEdgeKinds: Readonly<Record<string, string>> = {
  contradicts: "state the contradiction as contradicts_under_assumptions with its assumptions",
};

const accessClasses = ["public", "private"] as const;
const nodeStatuses = ["current", "stale", "incomplete"] as const;
const derivationMethods = [
  "extracted",
  "curated",
  "measured",
  "declared",
  "import_adjacency",
] as const;
const evidenceRequired = new Set<string>([
  "supports",
  "verified_by",
  "contradicts_under_assumptions",
]);
const assumptionsRequired = new Set<string>(["supports", "contradicts_under_assumptions"]);
const definitionLike = ["definition", "parameter", "program", "capability"];
const allKinds: readonly string[] = atlasNodeKinds;

/** Admitted source and target kinds of every canonical edge kind. */
export const atlasEdgeEndpoints: Readonly<
  Record<AtlasEdgeKind, readonly [ReadonlySet<string>, ReadonlySet<string>]>
> = {
  implements: [new Set(["definition", "program"]), new Set(["capability", "definition"])],
  consumes: [
    new Set(["run", "program", "definition", "instance"]),
    new Set(["artefact", "revision", "parameter", "backend"]),
  ],
  produces: [new Set(["run", "program", "instance"]), new Set(["artefact", "revision", "claim"])],
  derives_from: [
    new Set(["artefact", "revision", "claim", "definition", "instance"]),
    new Set(["artefact", "revision", "claim", "definition"]),
  ],
  constrained_by: [
    new Set(["definition", "program", "parameter", "run", "capability"]),
    new Set(["definition", "capability"]),
  ],
  lowers_to: [new Set(["program", "definition"]), new Set(["program", "definition"])],
  supports: [new Set(["artefact", "run", "test"]), new Set(["claim", "capability"])],
  requires: [
    new Set(["definition", "program", "capability", "run", "instance"]),
    new Set(["definition", "capability", "backend", "parameter"]),
  ],
  verified_by: [
    new Set(["claim", "capability", "definition", "program", "artefact"]),
    new Set(["test", "artefact", "run"]),
  ],
  benchmarked_against: [
    new Set(["program", "definition", "run", "artefact"]),
    new Set(["artefact", "program", "definition"]),
  ],
  calibrated_by: [new Set(["backend", "run"]), new Set(["artefact", "revision"])],
  supersedes: [
    new Set(["revision", "artefact", "claim", "definition", "documentation"]),
    new Set(["revision", "artefact", "claim", "definition", "documentation"]),
  ],
  contradicts_under_assumptions: [new Set(["claim"]), new Set(["claim"])],
  maps_to: [new Set(definitionLike), new Set(definitionLike)],
  contains: [
    new Set(["definition", "program", "capability"]),
    new Set([
      "definition",
      "parameter",
      "program",
      "test",
      "documentation",
      "capability",
      "instance",
    ]),
  ],
  documents: [
    new Set(["documentation"]),
    new Set(allKinds.filter((kind) => kind !== "documentation")),
  ],
};

/** PROV-O relation of each provenance edge kind; other kinds are not provenance. */
export const atlasProvRelations: Readonly<Partial<Record<AtlasEdgeKind, string>>> = {
  derives_from: "prov:wasDerivedFrom",
  produces: "prov:wasGeneratedBy",
  consumes: "prov:used",
  calibrated_by: "prov:used",
  supersedes: "prov:wasRevisionOf",
};
/** PROV-O class of node kinds that are not plain entities. */
export const atlasProvClasses: Readonly<Partial<Record<AtlasNodeKind, string>>> = {
  run: "prov:Activity",
  instance: "prov:Activity",
  backend: "prov:SoftwareAgent",
};

/** One validated Atlas node. */
export interface AtlasNode {
  /** Identity string whose shape follows the kind's identity class. */
  readonly id: string;
  /** Node kind. */
  readonly kind: AtlasNodeKind;
  /** Display label. */
  readonly label: string;
  /** Where the node was taken from. */
  readonly sourceRef: string;
  /** Content digest, or null with a stated reason. */
  readonly sourceDigest: string | null;
  /** Validity scope as sorted string pairs, or null with a stated reason. */
  readonly validityDomain: readonly (readonly [string, string])[] | null;
  /** UTC observation time, or null with a stated reason. */
  readonly asOf: string | null;
  /** Access class. */
  readonly access: "public" | "private";
  /** Freshness status. */
  readonly status: "current" | "stale" | "incomplete";
  /** Sorted evidence digests. */
  readonly evidenceRefs: readonly string[];
  /** Sorted reasons for each null optional field. */
  readonly nullReasons: readonly (readonly [string, string])[];
}

/** One validated, canonical Atlas edge. */
export interface AtlasEdge {
  /** Node ID the relation starts at. */
  readonly source: string;
  /** Canonical edge kind. */
  readonly kind: AtlasEdgeKind;
  /** Node ID the relation ends at. */
  readonly target: string;
  /** Always the admitted relation version. */
  readonly relationVersion: string;
  /** How the relation was obtained. */
  readonly derivationMethod: (typeof derivationMethods)[number];
  /** Where the relation was taken from. */
  readonly sourceRef: string;
  /** Evidence digest, or null where none is required. */
  readonly evidenceDigest: string | null;
  /** Stated assumptions in their given order. */
  readonly assumptions: readonly string[];
  /** Units and semantics of a mapping as sorted string pairs, or null. */
  readonly mapping: readonly (readonly [string, string])[] | null;
  /** UTC observation time, or null. */
  readonly asOf: string | null;
  /** Access class. */
  readonly access: "public" | "private";
}

/** Exact identity of a deleted object. */
export interface AtlasTombstone {
  /** The deleted node's ID, unchanged. */
  readonly id: string;
  /** The deleted node's kind, unchanged. */
  readonly kind: AtlasNodeKind;
  /** Content digest of the last live revision, or null. */
  readonly lastSourceDigest: string | null;
  /** UTC time of the deletion. */
  readonly removedAt: string;
  /** Why the object was removed. */
  readonly reason: string;
}

/** A documented exclusion from extraction. */
export interface AtlasExclusion {
  /** What was left out. */
  readonly sourceRef: string;
  /** Why it was left out. */
  readonly reason: string;
}

/** A validated Atlas snapshot in canonical order. */
export interface AtlasSnapshot {
  /** Version of the extractor that built the snapshot. */
  readonly extractorVersion: string;
  /** Full 40-hex commit the snapshot describes. */
  readonly sourceCommit: string;
  /** Digest of the evidence set. */
  readonly evidenceSetHash: string;
  /** Access scope of the whole snapshot. */
  readonly accessScope: "public" | "private";
  /** Nodes sorted by ID. */
  readonly nodes: readonly AtlasNode[];
  /** Edges sorted by source, kind, target and evidence digest. */
  readonly edges: readonly AtlasEdge[];
  /** Tombstones sorted by ID. */
  readonly tombstones: readonly AtlasTombstone[];
  /** Exclusions sorted by source reference and reason. */
  readonly exclusions: readonly AtlasExclusion[];
}

/** Outcome of parsing: the snapshot, or the refusal with the Python contract's code and path. */
export type AtlasParseResult =
  | {
      /** Validation succeeded. */
      readonly ok: true;
      /** The canonical snapshot. */
      readonly value: AtlasSnapshot;
    }
  | {
      /** Validation refused the document. */
      readonly ok: false;
      /** Stable refusal category, equal to the Python contract's code. */
      readonly code: string;
      /** JSON path of the offending value. */
      readonly path: string;
      /** Human-readable reason. */
      readonly message: string;
    };

class AtlasRefusal extends Error {
  constructor(
    readonly code: string,
    readonly path: string,
    message: string,
  ) {
    super(message);
  }
}

const hex64 = /^[0-9a-f]{64}$/;
const commit40 = /^[0-9a-f]{40}$/;
const uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const idPatterns: Readonly<Record<AtlasIdentityClass, RegExp>> = {
  definition: /^(?<kind>[a-z]+):[A-Za-z0-9_.-]+\/[A-Za-z0-9_.:-]+$/,
  revision: /^(?<kind>[a-z]+):sha256:(?<digest>[0-9a-f]{64})$/,
  instance: new RegExp(`^(?<kind>[a-z]+):${uuid}/${uuid}$`),
};
const nodeFields = [
  "id",
  "kind",
  "label",
  "source_ref",
  "source_digest",
  "validity_domain",
  "as_of",
  "access",
  "status",
  "evidence_refs",
  "null_reasons",
];
const nullableNodeFields = ["source_digest", "validity_domain", "as_of"];
const edgeFields = [
  "source",
  "kind",
  "target",
  "relation_version",
  "derivation_method",
  "source_ref",
  "evidence_digest",
  "assumptions",
  "mapping",
  "as_of",
  "access",
];
const snapshotFields = [
  "schema",
  "extractor_version",
  "source_commit",
  "evidence_set_hash",
  "access_scope",
  "nodes",
  "edges",
  "tombstones",
  "exclusions",
];

/** Compare by Unicode code points, the order Python uses for strings. */
function compareText(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0) as number);
  const b = Array.from(right, (character) => character.codePointAt(0) as number);
  const shared = Math.min(a.length, b.length);
  for (let index = 0; index < shared; index += 1) {
    const x = a[index] as number;
    const y = b[index] as number;
    if (x !== y) return x - y;
  }
  return a.length - b.length;
}
function compareTuples(left: readonly string[], right: readonly string[]): number {
  for (let index = 0; index < left.length; index += 1) {
    const difference = compareText(left[index] as string, right[index] as string);
    if (difference !== 0) return difference;
  }
  return 0;
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new AtlasRefusal("invalid_type", path, "expected an object with string keys");
  }
  return value as Record<string, unknown>;
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value === "")
    throw new AtlasRefusal("invalid_type", path, "expected a nonempty string");
  return value;
}
function choice<T extends string>(
  value: unknown,
  path: string,
  choices: readonly T[],
  code: string,
): T {
  const candidate = text(value, path);
  if (!(choices as readonly string[]).includes(candidate)) {
    throw new AtlasRefusal(code, path, `'${candidate}' is not one of ${choices.join(", ")}`);
  }
  return candidate as T;
}
function digest(value: unknown, path: string): string {
  const candidate = text(value, path);
  if (!hex64.test(candidate))
    throw new AtlasRefusal("invalid_digest", path, "expected 64 lowercase hexadecimal characters");
  return candidate;
}
function strings(value: unknown, path: string): string[] {
  if (!Array.isArray(value)) throw new AtlasRefusal("invalid_type", path, "expected an array");
  return value.map((item, index) => text(item, `${path}[${index}]`));
}
function pairs(value: unknown, path: string): (readonly [string, string])[] {
  const object = record(value, path);
  return Object.keys(object)
    .map((key) => [key, text(object[key], `${path}.${key}`)] as const)
    .sort((left, right) => compareTuples(left, right));
}
function closed(object: Record<string, unknown>, fields: readonly string[], path: string): void {
  const unknown = Object.keys(object)
    .filter((key) => !fields.includes(key))
    .sort(compareText);
  if (unknown.length > 0)
    throw new AtlasRefusal(
      "unknown_field",
      `${path}.${unknown[0]}`,
      "field is not part of the contract",
    );
  const missing = fields.filter((key) => !Object.hasOwn(object, key)).sort(compareText);
  if (missing.length > 0)
    throw new AtlasRefusal("missing_field", `${path}.${missing[0]}`, "required field is absent");
}
function identity(kind: AtlasNodeKind, id: string, path: string, message: string): RegExpExecArray {
  const match = idPatterns[atlasIdentityClasses[kind]].exec(id);
  if (match === null || match.groups?.kind !== kind)
    throw new AtlasRefusal("invalid_identity", path, message);
  return match;
}

function parseNode(payload: unknown, path: string): AtlasNode {
  const node = record(payload, path);
  closed(node, nodeFields, path);
  const kind = choice(node.kind, `${path}.kind`, atlasNodeKinds, "unknown_node_kind");
  const id = text(node.id, `${path}.id`);
  const identityClass = atlasIdentityClasses[kind];
  const match = identity(kind, id, `${path}.id`, `not a ${identityClass} identity of kind ${kind}`);
  const nullReasons = pairs(node.null_reasons, `${path}.null_reasons`);
  const reasons = new Map(nullReasons);
  for (const key of reasons.keys()) {
    if (!nullableNodeFields.includes(key) || node[key] !== null) {
      throw new AtlasRefusal(
        "invalid_null_reason",
        `${path}.null_reasons.${key}`,
        "no null to explain",
      );
    }
  }
  for (const key of nullableNodeFields) {
    if (node[key] === null && !reasons.has(key))
      throw new AtlasRefusal(
        "missing_null_reason",
        `${path}.${key}`,
        "a null needs a stated reason",
      );
  }
  const sourceDigest =
    node.source_digest === null ? null : digest(node.source_digest, `${path}.source_digest`);
  if (identityClass === "revision" && match.groups?.digest !== sourceDigest) {
    throw new AtlasRefusal(
      "invalid_identity",
      `${path}.id`,
      "a revision ID binds its content digest",
    );
  }
  const label = text(node.label, `${path}.label`);
  const sourceRef = text(node.source_ref, `${path}.source_ref`);
  const validityDomain =
    node.validity_domain === null ? null : pairs(node.validity_domain, `${path}.validity_domain`);
  const asOf = node.as_of === null ? null : text(node.as_of, `${path}.as_of`);
  const access = choice(node.access, `${path}.access`, accessClasses, "unknown_access");
  const status = choice(node.status, `${path}.status`, nodeStatuses, "unknown_status");
  const evidenceRefs = strings(node.evidence_refs, `${path}.evidence_refs`)
    .map((item, index) => digest(item, `${path}.evidence_refs[${index}]`))
    .sort(compareText);
  return {
    id,
    kind,
    label,
    sourceRef,
    sourceDigest,
    validityDomain,
    asOf,
    access,
    status,
    evidenceRefs,
    nullReasons,
  };
}

function canonicalKind(kind: string, path: string): readonly [AtlasEdgeKind, boolean] {
  if (Object.hasOwn(atlasRefusedEdgeKinds, kind))
    throw new AtlasRefusal("refused_edge_kind", path, atlasRefusedEdgeKinds[kind] as string);
  if (Object.hasOwn(atlasEdgeAliasesV1, kind))
    return atlasEdgeAliasesV1[kind] as readonly [AtlasEdgeKind, boolean];
  if (!(atlasEdgeKinds as readonly string[]).includes(kind))
    throw new AtlasRefusal("unknown_edge_kind", path, `'${kind}' is not an Atlas relation`);
  return [kind as AtlasEdgeKind, false];
}

function parseEdge(payload: unknown, path: string): AtlasEdge {
  const edge = record(payload, path);
  closed(edge, edgeFields, path);
  const relationVersion = text(edge.relation_version, `${path}.relation_version`);
  if (relationVersion !== atlasRelationVersion)
    throw new AtlasRefusal("unknown_relation_version", `${path}.relation_version`, relationVersion);
  const [kind, reverse] = canonicalKind(text(edge.kind, `${path}.kind`), `${path}.kind`);
  let source = text(edge.source, `${path}.source`);
  let target = text(edge.target, `${path}.target`);
  if (reverse) [source, target] = [target, source];
  const derivationMethod = choice(
    edge.derivation_method,
    `${path}.derivation_method`,
    derivationMethods,
    "unknown_derivation_method",
  );
  const sourceRef = text(edge.source_ref, `${path}.source_ref`);
  const evidenceDigest =
    edge.evidence_digest === null ? null : digest(edge.evidence_digest, `${path}.evidence_digest`);
  const assumptions = strings(edge.assumptions, `${path}.assumptions`);
  const mapping = edge.mapping === null ? null : pairs(edge.mapping, `${path}.mapping`);
  const asOf = edge.as_of === null ? null : text(edge.as_of, `${path}.as_of`);
  const access = choice(edge.access, `${path}.access`, accessClasses, "unknown_access");
  return {
    source,
    kind,
    target,
    relationVersion,
    derivationMethod,
    sourceRef,
    evidenceDigest,
    assumptions,
    mapping,
    asOf,
    access,
  };
}

function checkEdge(
  edge: AtlasEdge,
  nodes: ReadonlyMap<string, AtlasNode>,
  tombstoned: ReadonlySet<string>,
  path: string,
): void {
  for (const [end, id] of [
    ["source", edge.source],
    ["target", edge.target],
  ] as const) {
    if (tombstoned.has(id))
      throw new AtlasRefusal("tombstoned_endpoint", `${path}.${end}`, `${id} was deleted`);
    if (!nodes.has(id))
      throw new AtlasRefusal("dangling_endpoint", `${path}.${end}`, `${id} is not a live node`);
  }
  const source = nodes.get(edge.source) as AtlasNode;
  const target = nodes.get(edge.target) as AtlasNode;
  const [sources, targets] = atlasEdgeEndpoints[edge.kind];
  if (!sources.has(source.kind) || !targets.has(target.kind)) {
    throw new AtlasRefusal(
      "illegal_endpoint",
      path,
      `${edge.kind} cannot join ${source.kind} to ${target.kind}`,
    );
  }
  if (evidenceRequired.has(edge.kind) && edge.evidenceDigest === null) {
    throw new AtlasRefusal(
      "missing_evidence",
      `${path}.evidence_digest`,
      `${edge.kind} needs evidence`,
    );
  }
  if (assumptionsRequired.has(edge.kind) && edge.assumptions.length === 0) {
    throw new AtlasRefusal(
      "missing_assumptions",
      `${path}.assumptions`,
      `${edge.kind} needs assumptions`,
    );
  }
  if (edge.kind === "supports" && edge.derivationMethod === "import_adjacency") {
    throw new AtlasRefusal(
      "adjacency_is_not_support",
      `${path}.derivation_method`,
      "import adjacency never establishes support",
    );
  }
  if (edge.kind === "maps_to") {
    const keys = new Set((edge.mapping ?? []).map(([key]) => key));
    if (!keys.has("units") || !keys.has("semantics"))
      throw new AtlasRefusal(
        "missing_mapping",
        `${path}.mapping`,
        "maps_to needs units and semantics",
      );
  }
  if (edge.kind === "contradicts_under_assumptions") {
    const sourceDomain = new Map(source.validityDomain ?? []);
    const targetDomain = new Map(target.validityDomain ?? []);
    for (const field of ["estimand", "units"]) {
      if (!sourceDomain.has(field) || sourceDomain.get(field) !== targetDomain.get(field)) {
        throw new AtlasRefusal(
          "non_comparable",
          path,
          `the claims do not share one ${field}; show them as non-comparable`,
        );
      }
    }
  }
  if ((source.access === "private" || target.access === "private") && edge.access !== "private") {
    throw new AtlasRefusal(
      "access_leak",
      `${path}.access`,
      "an edge touching a private node is private",
    );
  }
}

function parseTombstone(payload: unknown, path: string): AtlasTombstone {
  const tombstone = record(payload, path);
  closed(tombstone, ["id", "kind", "last_source_digest", "removed_at", "reason"], path);
  const kind = choice(tombstone.kind, `${path}.kind`, atlasNodeKinds, "unknown_node_kind");
  const id = text(tombstone.id, `${path}.id`);
  identity(kind, id, `${path}.id`, "tombstone keeps the exact prior identity");
  const lastSourceDigest =
    tombstone.last_source_digest === null
      ? null
      : digest(tombstone.last_source_digest, `${path}.last_source_digest`);
  return {
    id,
    kind,
    lastSourceDigest,
    removedAt: text(tombstone.removed_at, `${path}.removed_at`),
    reason: text(tombstone.reason, `${path}.reason`),
  };
}

function items(document: Record<string, unknown>, key: string): unknown[] {
  const value = document[key];
  if (!Array.isArray(value))
    throw new AtlasRefusal("invalid_type", `$.${key}`, "expected an array");
  return value;
}

function edgeKey(edge: AtlasEdge): string[] {
  return [edge.source, edge.kind, edge.target, edge.evidenceDigest ?? ""];
}

function parseOrThrow(payload: unknown): AtlasSnapshot {
  const document = record(payload, "$");
  closed(document, snapshotFields, "$");
  const schema = text(document.schema, "$.schema");
  if (schema !== atlasSnapshotSchema)
    throw new AtlasRefusal(
      "unknown_schema",
      "$.schema",
      `'${schema}' is not ${atlasSnapshotSchema}`,
    );
  const sourceCommit = text(document.source_commit, "$.source_commit");
  if (!commit40.test(sourceCommit))
    throw new AtlasRefusal("invalid_commit", "$.source_commit", "expected a full 40-hex commit");
  const accessScope = choice(
    document.access_scope,
    "$.access_scope",
    accessClasses,
    "unknown_access",
  );
  const nodes = items(document, "nodes").map((item, index) => parseNode(item, `$.nodes[${index}]`));
  const tombstones = items(document, "tombstones").map((item, index) =>
    parseTombstone(item, `$.tombstones[${index}]`),
  );
  const byId = new Map<string, AtlasNode>();
  nodes.forEach((node, index) => {
    if (byId.has(node.id))
      throw new AtlasRefusal("duplicate_id", `$.nodes[${index}].id`, `${node.id} appears twice`);
    if (accessScope === "public" && node.access === "private") {
      throw new AtlasRefusal(
        "access_leak",
        `$.nodes[${index}].access`,
        "a public snapshot has no private node",
      );
    }
    byId.set(node.id, node);
  });
  const tombstoned = new Set<string>();
  tombstones.forEach((tombstone, index) => {
    if (byId.has(tombstone.id))
      throw new AtlasRefusal(
        "reused_id",
        `$.tombstones[${index}].id`,
        `${tombstone.id} is deleted and live`,
      );
    if (tombstoned.has(tombstone.id))
      throw new AtlasRefusal(
        "duplicate_id",
        `$.tombstones[${index}].id`,
        `${tombstone.id} appears twice`,
      );
    tombstoned.add(tombstone.id);
  });
  const edges = items(document, "edges").map((item, index) => parseEdge(item, `$.edges[${index}]`));
  const seen = new Set<string>();
  edges.forEach((edge, index) => {
    checkEdge(edge, byId, tombstoned, `$.edges[${index}]`);
    if (accessScope === "public" && edge.access === "private") {
      throw new AtlasRefusal(
        "access_leak",
        `$.edges[${index}].access`,
        "a public snapshot has no private edge",
      );
    }
    const key = JSON.stringify(edgeKey(edge));
    if (seen.has(key))
      throw new AtlasRefusal(
        "duplicate_edge",
        `$.edges[${index}]`,
        "the same relation appears twice",
      );
    seen.add(key);
  });
  const extractorVersion = text(document.extractor_version, "$.extractor_version");
  const evidenceSetHash = digest(document.evidence_set_hash, "$.evidence_set_hash");
  const exclusions = items(document, "exclusions").map((item, index) => {
    const path = `$.exclusions[${index}]`;
    const exclusion = record(item, path);
    closed(exclusion, ["source_ref", "reason"], path);
    return {
      sourceRef: text(exclusion.source_ref, `${path}.source_ref`),
      reason: text(exclusion.reason, `${path}.reason`),
    };
  });
  return Object.freeze({
    extractorVersion,
    sourceCommit,
    evidenceSetHash,
    accessScope,
    nodes: [...nodes].sort((left, right) => compareText(left.id, right.id)),
    edges: [...edges].sort((left, right) => compareTuples(edgeKey(left), edgeKey(right))),
    tombstones: [...tombstones].sort((left, right) => compareText(left.id, right.id)),
    exclusions: exclusions.sort((left, right) =>
      compareTuples([left.sourceRef, left.reason], [right.sourceRef, right.reason]),
    ),
  });
}

/**
 * Validate an untrusted Atlas snapshot document.
 *
 * @param payload - A `quantum_atlas_snapshot.v1` document.
 * @returns The canonical snapshot, or the refusal with the same code and path as the Python contract.
 */
export function parseAtlasSnapshot(payload: unknown): AtlasParseResult {
  try {
    return { ok: true, value: parseOrThrow(payload) };
  } catch (error) {
    if (error instanceof AtlasRefusal)
      return { ok: false, code: error.code, path: error.path, message: error.message };
    throw error;
  }
}

function pairsWire(
  value: readonly (readonly [string, string])[] | null,
): Record<string, string> | null {
  return value === null ? null : Object.fromEntries(value);
}

/**
 * Return the canonical wire document of a validated snapshot.
 *
 * @param snapshot - A snapshot returned by {@link parseAtlasSnapshot} or {@link projectAtlasPublic}.
 * @returns The document with canonical edge kinds and canonical order.
 */
export function atlasSnapshotToWire(snapshot: AtlasSnapshot): Record<string, unknown> {
  return {
    schema: atlasSnapshotSchema,
    extractor_version: snapshot.extractorVersion,
    source_commit: snapshot.sourceCommit,
    evidence_set_hash: snapshot.evidenceSetHash,
    access_scope: snapshot.accessScope,
    nodes: snapshot.nodes.map((node) => ({
      id: node.id,
      kind: node.kind,
      label: node.label,
      source_ref: node.sourceRef,
      source_digest: node.sourceDigest,
      validity_domain: pairsWire(node.validityDomain),
      as_of: node.asOf,
      access: node.access,
      status: node.status,
      evidence_refs: [...node.evidenceRefs],
      null_reasons: Object.fromEntries(node.nullReasons),
    })),
    edges: snapshot.edges.map((edge) => ({
      source: edge.source,
      kind: edge.kind,
      target: edge.target,
      relation_version: edge.relationVersion,
      derivation_method: edge.derivationMethod,
      source_ref: edge.sourceRef,
      evidence_digest: edge.evidenceDigest,
      assumptions: [...edge.assumptions],
      mapping: pairsWire(edge.mapping),
      as_of: edge.asOf,
      access: edge.access,
    })),
    tombstones: snapshot.tombstones.map((tombstone) => ({
      id: tombstone.id,
      kind: tombstone.kind,
      last_source_digest: tombstone.lastSourceDigest,
      removed_at: tombstone.removedAt,
      reason: tombstone.reason,
    })),
    exclusions: snapshot.exclusions.map((exclusion) => ({
      source_ref: exclusion.sourceRef,
      reason: exclusion.reason,
    })),
  };
}

/**
 * Hash a validated snapshot exactly as the Python contract does.
 *
 * @param snapshot - A validated snapshot.
 * @returns Lowercase SHA-256 hex of the canonical wire document.
 */
export async function atlasSnapshotDigest(snapshot: AtlasSnapshot): Promise<string> {
  return canonicalDigest(atlasSnapshotSchema, atlasSnapshotToWire(snapshot));
}

/**
 * Return the public projection of a snapshot without disclosing what was removed.
 *
 * @param snapshot - A validated snapshot of either access scope.
 * @returns Public-scope snapshot without private nodes or any edge that is private or touches one.
 */
export function projectAtlasPublic(snapshot: AtlasSnapshot): AtlasSnapshot {
  const publicIds = new Set(
    snapshot.nodes.filter((node) => node.access === "public").map((node) => node.id),
  );
  return Object.freeze({
    ...snapshot,
    accessScope: "public",
    nodes: snapshot.nodes.filter((node) => publicIds.has(node.id)),
    edges: snapshot.edges.filter(
      (edge) =>
        edge.access === "public" && publicIds.has(edge.source) && publicIds.has(edge.target),
    ),
  });
}
