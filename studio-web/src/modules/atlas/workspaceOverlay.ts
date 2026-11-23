// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — private Atlas overlay of the admitted workspace

import {
  type AtlasSnapshot,
  atlasRelationVersion,
  atlasSnapshotDigest,
  atlasSnapshotToWire,
  parseAtlasSnapshot,
} from "../../shared/contracts/atlas";
import { canonicalDigest } from "../../shared/contracts/canonical";
import type { AdmittedWorkspaceArchive } from "../../shared/storage/workspaceArchive";

/** Version of the projection rules; the Python owner declares the same value. */
export const workspaceOverlayVersion = "workspace-overlay.1";
/** Domain of the overlay's evidence-set hash, shared with the extractor. */
export const atlasEvidenceSetSchema = "quantum_atlas_evidence_set.v1";

const hex64 = /^[0-9a-f]{64}$/;
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const parameterKey = /^[A-Za-z0-9_.:-]+$/;
const noTime = "workspace documents carry no observation time";
const noDomain = "the workspace states no validity scope";

/** A workspace view or a previous overlay that cannot be projected. */
export class AtlasWorkspaceOverlayError extends Error {
  /**
   * @param path - Address of the refused field.
   * @param reason - Why it was refused.
   */
  constructor(
    /** Address of the refused field. */ readonly path: string,
    reason: string,
  ) {
    super(`${path}: ${reason}`);
    this.name = "AtlasWorkspaceOverlayError";
  }
}

/** One admitted workspace document as the overlay reads it. */
export interface AtlasWorkspaceDocument {
  /** The document schema. */ readonly schema: string;
  /** The document body. */ readonly body: Readonly<Record<string, unknown>>;
}

/** The admitted workspace facts the overlay reads. */
export interface AtlasWorkspaceView {
  /** The workspace project UUID. */ readonly projectId: string;
  /** Digest of the admitted workspace root document. */ readonly workspaceHash: string;
  /** The root's `updated_at`; it dates tombstones of removed objects. */ readonly updatedAt: string;
  /** Admitted documents by digest. */ readonly documents: ReadonlyMap<
    string,
    AtlasWorkspaceDocument
  >;
}

type Json = Readonly<Record<string, unknown>>;

function record(value: unknown, path: string): Json {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new AtlasWorkspaceOverlayError(path, "must be an object");
  return value as Json;
}
function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value === "")
    throw new AtlasWorkspaceOverlayError(path, "must be non-empty text");
  return value;
}
function digest(value: unknown, path: string): string {
  const value_ = text(value, path);
  if (!hex64.test(value_))
    throw new AtlasWorkspaceOverlayError(path, "must be a lowercase sha256 digest");
  return value_;
}
/** Order strings by Unicode code point, as Python's `sorted` does. */
function compareText(left: string, right: string): number {
  const a = Array.from(left, (character) => character.codePointAt(0) as number);
  const b = Array.from(right, (character) => character.codePointAt(0) as number);
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    const difference = (a[index] as number) - (b[index] as number);
    if (difference !== 0) return difference;
  }
  return a.length - b.length;
}
function compareTuples(left: readonly string[], right: readonly string[]): number {
  return left.reduce((found, item, index) => found || compareText(item, right[index] as string), 0);
}

/**
 * Read a workspace view from its wire form, as the Python owner does.
 *
 * @param payload - `{project_id, workspace_hash, updated_at, documents}`; each document is `{schema, body}` under its digest.
 * @returns The view.
 * @throws AtlasWorkspaceOverlayError when a field is missing, of the wrong type or not a valid identity.
 */
export function parseAtlasWorkspaceView(payload: unknown): AtlasWorkspaceView {
  const view = record(payload, "$");
  const projectId = text(view["project_id"], "$.project_id");
  if (!uuid.test(projectId))
    throw new AtlasWorkspaceOverlayError("$.project_id", "must be a lowercase UUID");
  const documents = new Map<string, AtlasWorkspaceDocument>();
  for (const [key, value] of Object.entries(record(view["documents"], "$.documents"))) {
    const path = `$.documents.${key}`;
    const document = record(value, path);
    documents.set(digest(key, path), {
      schema: text(document["schema"], `${path}.schema`),
      body: record(document["body"], `${path}.body`),
    });
  }
  return {
    projectId,
    workspaceHash: digest(view["workspace_hash"], "$.workspace_hash"),
    updatedAt: text(view["updated_at"], "$.updated_at"),
    documents,
  };
}

/**
 * Take the overlay's view of a fully admitted workspace archive.
 *
 * @param archive - An archive admitted by `admitWorkspaceArchive`.
 * @returns The project, root digest, update time and every admitted document.
 */
export function atlasWorkspaceViewOf(archive: AdmittedWorkspaceArchive): AtlasWorkspaceView {
  return parseAtlasWorkspaceView({
    project_id: archive.preview.projectId,
    workspace_hash: archive.preview.workspaceHash,
    updated_at: archive.manifest.body["updated_at"],
    documents: Object.fromEntries(
      Object.entries(archive.documents).map(([key, value]) => [
        key,
        { schema: value.schema, body: value.body },
      ]),
    ),
  });
}

class Fragment {
  readonly source: string;
  readonly nodes = new Map<string, Record<string, unknown>>();
  readonly edges = new Map<string, Record<string, unknown>>();
  readonly exclusions = new Map<string, readonly [string, string]>();

  constructor(readonly view: AtlasWorkspaceView) {
    this.source = `workspace/${view.projectId}`;
  }

  node(id: string, kind: string, label: string, source: string | null, why: string): void {
    const reasons: Record<string, string> = { as_of: noTime, validity_domain: noDomain };
    if (source === null) reasons["source_digest"] = why;
    this.nodes.set(id, {
      id,
      kind,
      label,
      source_ref: this.source,
      source_digest: source,
      validity_domain: null,
      as_of: null,
      access: "private",
      status: "current",
      evidence_refs: [this.view.workspaceHash],
      null_reasons: reasons,
    });
  }

  edge(source: string, kind: string, target: string): void {
    this.edges.set(JSON.stringify([source, kind, target]), {
      source,
      kind,
      target,
      relation_version: atlasRelationVersion,
      derivation_method: "extracted",
      source_ref: this.source,
      evidence_digest: this.view.workspaceHash,
      assumptions: [],
      mapping: null,
      as_of: null,
      access: "private",
    });
  }

  exclude(document: string, reason: string): void {
    const sourceRef = `${this.source}/${document}`;
    this.exclusions.set(JSON.stringify([sourceRef, reason]), [sourceRef, reason]);
  }
}

function ofSchema(view: AtlasWorkspaceView, schema: string): (readonly [string, Json])[] {
  return [...view.documents]
    .filter(([, document]) => document.schema === schema)
    .sort(([left], [right]) => compareText(left, right))
    .map(([key, document]) => [key, record(document.body, `$.documents.${key}.body`)] as const);
}

function parameters(fragment: Fragment): Map<string, string> {
  const keys = new Map<string, string[]>();
  for (const [key, body] of ofSchema(fragment.view, "parameter_spec.v1")) {
    const name = text(body["key"], `$.documents.${key}.body.key`);
    if (parameterKey.test(name)) keys.set(name, [...(keys.get(name) ?? []), key]);
    else fragment.exclude(key, "the parameter key is not an Atlas identifier");
  }
  const nodes = new Map<string, string>();
  for (const [name, digests] of [...keys].sort(([left], [right]) => compareText(left, right))) {
    const id = `parameter:workspace.${fragment.view.projectId}/${name}`;
    const single = digests.length === 1 ? (digests[0] as string) : null;
    fragment.node(
      id,
      "parameter",
      name,
      single,
      "the workspace holds several specifications of it",
    );
    nodes.set(name, id);
  }
  return nodes;
}

function revisions(fragment: Fragment, parameterNodes: ReadonlyMap<string, string>): void {
  const view = fragment.view;
  const all = ofSchema(view, "experiment_revision.v1");
  const present = new Set(all.map(([key]) => key));
  for (const [key, body] of all) {
    const path = `$.documents.${key}.body`;
    const id = `revision:sha256:${key}`;
    fragment.node(id, "revision", `Experiment revision ${key.slice(0, 12)}`, key, "");
    const parents = body["parent_revision_hashes"];
    if (!Array.isArray(parents))
      throw new AtlasWorkspaceOverlayError(`${path}.parent_revision_hashes`, "must be an array");
    parents.forEach((parent: unknown, index) => {
      const parentDigest = digest(parent, `${path}.parent_revision_hashes[${index}]`);
      if (present.has(parentDigest))
        fragment.edge(id, "derives_from", `revision:sha256:${parentDigest}`);
      else fragment.exclude(key, `parent revision ${parentDigest} is not in the workspace`);
    });
    const program = digest(
      record(body["program_ref"], `${path}.program_ref`)["sha256"],
      `${path}.program_ref.sha256`,
    );
    const programId = `program:workspace.${view.projectId}/${program}`;
    fragment.node(programId, "program", `Program ${program.slice(0, 12)}`, program, "");
    fragment.edge(programId, "consumes", id);
    const bound = Object.keys(record(body["parameters"], `${path}.parameters`)).sort(compareText);
    for (const name of bound) {
      const parameter = parameterNodes.get(name);
      if (parameter !== undefined) fragment.edge(programId, "consumes", parameter);
    }
  }
}

function runs(fragment: Fragment): void {
  for (const [key, body] of ofSchema(fragment.view, "local_run_record.v1")) {
    const path = `$.documents.${key}.body`;
    const runId = text(body["run_id"], `${path}.run_id`);
    const attemptId = text(body["attempt_id"], `${path}.attempt_id`);
    const revision = digest(body["revision_hash"], `${path}.revision_hash`);
    if (!(uuid.test(runId) && uuid.test(attemptId))) {
      fragment.exclude(key, "the run or attempt identifier is not a lowercase UUID");
      continue;
    }
    const id = `run:${runId}/${attemptId}`;
    fragment.node(id, "run", `Run ${runId.slice(0, 8)} attempt ${attemptId.slice(0, 8)}`, key, "");
    if (fragment.nodes.has(`revision:sha256:${revision}`))
      fragment.edge(id, "consumes", `revision:sha256:${revision}`);
    else fragment.exclude(key, `revision ${revision} is not in the workspace`);
  }
}

function tombstones(
  publicSnapshot: AtlasSnapshot,
  previous: AtlasSnapshot | null,
  live: ReadonlySet<string>,
  removedAt: string,
): Record<string, unknown>[] {
  const wire = (snapshot: AtlasSnapshot) =>
    atlasSnapshotToWire(snapshot)["tombstones"] as Record<string, unknown>[];
  const carried = [...wire(publicSnapshot)];
  if (previous === null) return carried;
  if (previous.accessScope !== "private")
    throw new AtlasWorkspaceOverlayError("previous", "must be an earlier private overlay");
  const seen = new Set(carried.map((item) => item["id"] as string));
  for (const tombstone of wire(previous)) {
    const id = tombstone["id"] as string;
    if (!seen.has(id) && !live.has(id)) {
      seen.add(id);
      carried.push(tombstone);
    }
  }
  for (const node of previous.nodes) {
    if (node.access === "private" && !live.has(node.id) && !seen.has(node.id))
      carried.push({
        id: node.id,
        kind: node.kind,
        last_source_digest: node.sourceDigest,
        removed_at: removedAt,
        reason: "no longer in the admitted workspace",
      });
  }
  return carried;
}

/**
 * Join a public snapshot and one admitted workspace into a private snapshot.
 *
 * @param publicSnapshot - The public Atlas snapshot, normally the committed pin.
 * @param view - The admitted workspace.
 * @param previous - An earlier overlay; its private nodes that are gone become tombstones dated by `view.updatedAt`.
 * @returns A validated private snapshot holding every public node and edge unchanged.
 * @throws AtlasWorkspaceOverlayError when the inputs cannot be projected or a workspace identity collides with a public one.
 */
export async function overlayAtlasWorkspace(
  publicSnapshot: AtlasSnapshot,
  view: AtlasWorkspaceView,
  previous: AtlasSnapshot | null = null,
): Promise<AtlasSnapshot> {
  if (publicSnapshot.accessScope !== "public")
    throw new AtlasWorkspaceOverlayError("public", "must be a public snapshot");
  const fragment = new Fragment(view);
  revisions(fragment, parameters(fragment));
  runs(fragment);
  const wire = atlasSnapshotToWire(publicSnapshot);
  const nodes = new Map(
    (wire["nodes"] as Record<string, unknown>[]).map((node) => [node["id"] as string, node]),
  );
  for (const [id, node] of fragment.nodes) {
    if (nodes.has(id)) throw new AtlasWorkspaceOverlayError(id, "collides with a public node");
    nodes.set(id, node);
  }
  const edgeKey = (edge: Record<string, unknown>) =>
    [edge["source"], edge["kind"], edge["target"], edge["evidence_digest"]] as string[];
  const edges = [...(wire["edges"] as Record<string, unknown>[]), ...fragment.edges.values()].sort(
    (left, right) => compareTuples(edgeKey(left), edgeKey(right)),
  );
  const exclusions = new Map(fragment.exclusions);
  for (const item of wire["exclusions"] as Record<string, unknown>[]) {
    const pair = [item["source_ref"] as string, item["reason"] as string] as const;
    exclusions.set(JSON.stringify(pair), pair);
  }
  const evidence = [
    ["atlas-pin", await atlasSnapshotDigest(publicSnapshot)],
    [fragment.source, view.workspaceHash],
  ].sort(compareTuples);
  const parsed = parseAtlasSnapshot({
    schema: wire["schema"],
    extractor_version: `${publicSnapshot.extractorVersion}+${workspaceOverlayVersion}`,
    source_commit: publicSnapshot.sourceCommit,
    evidence_set_hash: await canonicalDigest(atlasEvidenceSetSchema, evidence),
    access_scope: "private",
    nodes: [...nodes.keys()].sort(compareText).map((id) => nodes.get(id)),
    edges,
    tombstones: tombstones(publicSnapshot, previous, new Set(nodes.keys()), view.updatedAt),
    exclusions: [...exclusions.values()]
      .sort(compareTuples)
      .map(([sourceRef, reason]) => ({ source_ref: sourceRef, reason })),
  });
  if (!parsed.ok)
    throw new AtlasWorkspaceOverlayError(
      parsed.path,
      `the overlay breaks the contract: ${parsed.code}`,
    );
  return parsed.value;
}
