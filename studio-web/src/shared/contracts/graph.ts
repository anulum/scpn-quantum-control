// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace reference admission

import { documentDigest, parseDocument, parseParameterSpec, parseWorkspaceManifest, validateParameterBinding, workspaceSchemas } from "./workspace";
import type { ParameterSpec, ParseResult, WorkspaceDocument, WorkspaceManifest } from "./workspace";

/** Identity verified with an explicitly registered original producer codec. */
export interface RawIdentity {
  /** Actual verified schema. */ readonly schema: string;
  /** Owning producer role, such as problem, program or plan. */ readonly kind: string;
  /** Original digest, never a workspace re-encoding. */ readonly digest: string;
}
/** Source bytes copied before asynchronous verification starts. */
export interface RawArtifact {
  /** Claimed schema selects a verifier; it is not evidence of support. */ readonly schema: string;
  /** Original producer content. */ readonly content: Uint8Array;
}
/** Trusted offline verifier supplied in code, never loaded from imported data. */
export type RawCodec = (content: Uint8Array) => Promise<RawIdentity>;
/** Structural integrity receipt that grants no worker or provider authority. */
export interface WorkspaceAdmission {
  /** Admitted workspace project UUID. */ readonly projectId: string;
  /** Exact admitted root identity, including its extensions and reference lists. */ readonly workspaceHash: string;
  /** Sorted exact document identities. */ readonly documentHashes: readonly string[];
  /** Sorted original identities actually verified. */ readonly rawHashes: readonly string[];
}

function take<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
  return result.value;
}
function acyclic(edges: ReadonlyMap<string, readonly string[]>, path: string): void {
  const degrees = new Map([...edges].map(([key, parents]) => [key, new Set(parents).size]));
  const children = new Map([...edges.keys()].map(key => [key, [] as string[]]));
  for (const [key, parents] of edges) {
    for (const parent of new Set(parents)) {
      const descendants = children.get(parent);
      if (!descendants) throw new Error(`${path}: dangling dependency ${parent}`);
      descendants.push(key);
    }
  }
  const ready = [...degrees].filter(([, degree]) => degree === 0).map(([key]) => key);
  let visited = 0;
  while (ready.length > 0) {
    const key = ready.pop()!;
    visited++;
    for (const child of children.get(key)!) {
      const degree = degrees.get(child)! - 1;
      degrees.set(child, degree);
      if (degree === 0) ready.push(child);
    }
  }
  if (visited !== edges.size) throw new Error(`${path}: dependency cycle`);
}

/** Admit complete reference indexes without fetching, writing or launching a worker. */
export async function admitWorkspace(
  manifest: WorkspaceManifest,
  documents: ReadonlyMap<string, WorkspaceDocument>,
  rawRecords: ReadonlyMap<string, RawArtifact>,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  parameterSpecs: ReadonlyMap<string, ParameterSpec>,
  parameterUnits: ReadonlyMap<string, string>,
): Promise<ParseResult<WorkspaceAdmission>> {
  try {
    const root = take(parseWorkspaceManifest(manifest));
    const projectId = root.body["project_id"] as string;
    // Capture all caller-owned containers before the first asynchronous digest/verifier.
    const index = new Map([...documents].map(([digest, value]) => [digest, take(parseDocument(value))]));
    const rawSnapshot = new Map([...rawRecords].map(([digest, value]) => [digest, { schema: value.schema, content: new Uint8Array(value.content) }]));
    const specs = new Map([...parameterSpecs].map(([key, value]) => [key, take(parseParameterSpec(value))]));
    const units = new Map(parameterUnits);
    const codecs = new Map(rawCodecs);
    const rootDigest = await documentDigest(root);
    for (const [digest, record] of index) {
      if (await documentDigest(record) !== digest) throw new Error("$.documents: digest mismatch");
      if (record.schema === "quantum_workspace.v1" && digest !== rootDigest) throw new Error("$.documents: unexpected workspace root");
      if (record.schema === "experiment_revision.v1" && record.body["project_id"] !== projectId) throw new Error("$.documents: cross-project revision");
      if (rawSnapshot.has(digest)) throw new Error("$.documents: ambiguous raw/document identity");
    }
    if (specs.size !== units.size || [...specs.keys()].some(key => !units.has(key))) throw new Error("$.parameter_units: keys must match specifications");
    const specDigests = new Map<string, string>();
    for (const [key, spec] of specs) {
      const digest = await documentDigest(spec);
      if (key !== spec.body["key"] || index.get(digest)?.schema !== "parameter_spec.v1") throw new Error("$.parameter_specs: key or indexed identity mismatch");
      specDigests.set(key, digest);
    }
    acyclic(new Map([...specs].map(([key, spec]) => [key, spec.body["dependency_keys"] as readonly string[]])), "$.parameter_specs");
    const verified = new Map<string, RawIdentity>();
    const rawIdentity = async (digest: string): Promise<RawIdentity> => {
      const cached = verified.get(digest);
      if (cached) return cached;
      const artifact = rawSnapshot.get(digest);
      if (!artifact) throw new Error("$.references: dangling raw reference");
      const codec = codecs.get(artifact.schema);
      if (!codec) throw new Error("$.references: unsupported raw producer/schema");
      const identity = await codec(new Uint8Array(artifact.content));
      if (identity.schema !== artifact.schema || identity.digest !== digest || typeof identity.kind !== "string" || identity.kind === "") throw new Error("$.references: raw producer identity mismatch");
      const snapshot = Object.freeze({ schema: identity.schema, kind: identity.kind, digest: identity.digest });
      verified.set(digest, snapshot);
      return snapshot;
    };
    const resolve = async (ref: Readonly<Record<string, unknown>>, expected?: string): Promise<void> => {
      const digest = ref["sha256"] as string;
      const document = index.get(digest);
      if (!document && workspaceSchemas.some(schema => schema === ref["schema"])) throw new Error("$.references: workspace reference requires indexed document");
      const identity = document ? { schema: document.schema, kind: document.schema } : await rawIdentity(digest);
      if (identity.schema !== ref["schema"] || (expected !== undefined && identity.kind !== expected)) throw new Error("$.references: schema or kind mismatch");
    };
    for (const ref of root.body["revision_refs"] as readonly Readonly<Record<string, unknown>>[]) await resolve(ref, "experiment_revision.v1");
    for (const ref of root.body["artefact_refs"] as readonly Readonly<Record<string, unknown>>[]) await resolve(ref);
    if (root.body["draft_ref"] !== null) await resolve(root.body["draft_ref"] as Readonly<Record<string, unknown>>, "experiment_revision.v1");
    const parents = new Map<string, readonly string[]>();
    for (const [digest, record] of index) {
      const body = record.body;
      if (record.schema === "experiment_revision.v1") {
        parents.set(digest, body["parent_revision_hashes"] as readonly string[]);
        await resolve(body["problem_ref"] as Readonly<Record<string, unknown>>, "problem");
        await resolve(body["program_ref"] as Readonly<Record<string, unknown>>, "program");
        await resolve(body["semantic_settings_ref"] as Readonly<Record<string, unknown>>, "resolved_settings.v1");
        const inputs = body["input_refs"] as readonly Readonly<Record<string, unknown>>[];
        for (const ref of inputs) await resolve(ref);
        const inputHashes = new Set(inputs.map(ref => ref["sha256"]));
        for (const [key, values] of Object.entries(body["parameters"] as Readonly<Record<string, unknown>>)) {
          const spec = specs.get(key);
          if (!spec || !inputHashes.has(specDigests.get(key))) throw new Error("$.parameters: missing immutable specification reference");
          take(validateParameterBinding(spec, values, units.get(key)!));
        }
      } else if (record.schema === "resolved_settings.v1") {
        await resolve(body["policy_ref"] as Readonly<Record<string, unknown>>, "policy");
        await resolve(body["environment_ref"] as Readonly<Record<string, unknown>>, "environment");
      } else if (record.schema === "local_run_record.v1") {
        if (index.get(body["revision_hash"] as string)?.schema !== "experiment_revision.v1") throw new Error("$.run.revision_hash: missing revision");
        if ((await rawIdentity(body["plan_hash"] as string)).kind !== "plan") throw new Error("$.run.plan_hash: wrong raw kind");
        for (const ref of body["output_refs"] as readonly Readonly<Record<string, unknown>>[]) await resolve(ref);
      }
    }
    acyclic(parents, "$.revisions");
    return { ok: true, value: Object.freeze({ projectId, workspaceHash: rootDigest, documentHashes: Object.freeze([...index.keys()].sort()), rawHashes: Object.freeze([...verified.keys()].sort()) }) };
  } catch (error: unknown) {
    return { ok: false, code: "invalid_graph", path: "$", message: error instanceof Error ? error.message : "Workspace admission refused" };
  }
}
