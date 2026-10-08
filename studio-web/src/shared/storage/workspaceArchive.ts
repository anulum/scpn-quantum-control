// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — portable workspace archive

import { admitWorkspace } from "../contracts/graph";
import type { RawArtifact, RawCodec } from "../contracts/graph";
import {
  canonicalDigest,
  readJson,
  writeJson,
  parseDocument,
  parseParameterSpec,
  parseWorkspaceManifest,
} from "../contracts";
import type {
  ExperimentRevision,
  ParameterSpec,
  ParseResult,
  WorkspaceDocument,
  WorkspaceManifest,
} from "../contracts";

/** Maximum encoded archive bytes; a product bound, not available host RAM. */
export const maxArchiveBytes = 64 * 1024 * 1024;
/** Maximum decoded member bytes retained by the portable archive. */
export const maxExpandedBytes = 128 * 1024 * 1024;
/** Maximum member count, including the root manifest. */
export const maxArchiveMembers = 1000;
/** Canonical string domain for an exact archive snapshot, including its source formatting. */
export const archiveSnapshotDomain = "quantum_workspace_archive_source.v1";

/** Complete offline admission preview. It grants no numerical or provider authority. */
export interface WorkspaceArchivePreview {
  /** Exact portable container version; unknown versions are refused. */
  readonly schema: "quantum_workspace_archive.v1";
  /** Validated project identity. */
  readonly projectId: string;
  /** Digest of the exact source text under archiveSnapshotDomain; original document hashes stay separate. */
  readonly archiveDigest: string;
  /** Original workspace document identity. */
  readonly workspaceHash: string;
  /** Sorted exact immutable document identities. */
  readonly documentHashes: readonly string[];
  /** Sorted identities validated by trusted original producer codecs. */
  readonly rawHashes: readonly string[];
  /** Safe unique relative member names in input order. */
  readonly memberNames: readonly string[];
  /** Exact lossless source text, retained for portable backup and atomic storage. */
  readonly json: string;
}

/** Exact immutable container member retained by the original archive decoder. */
export interface WorkspaceArchiveMember {
  /** Validated unique relative member name. */ readonly name: string;
  /** Original document or producer-verified raw bytes. */ readonly kind: "document" | "raw";
  /** Original versioned producer/document schema. */ readonly schema: string;
  /** Verified original identity, independent of the member name. */ readonly sha256: string;
  /** Original lossless document text or canonical raw hex, without rewriting. */ readonly content: string;
}

/** Typed editing inputs released only after complete original graph admission. */
export interface AdmittedWorkspaceArchive {
  /** Original offline preview, including the exact archive text and digest. */ readonly preview: WorkspaceArchivePreview;
  /** Structurally validated and graph-admitted original root. */ readonly manifest: WorkspaceManifest;
  /** Original container members in their input order. */ readonly members: readonly WorkspaceArchiveMember[];
  /** Explicit admitted source units, without dimensional conversion. */ readonly parameterUnits: Readonly<
    Record<string, string>
  >;
  /** Complete original immutable document index after graph/digest admission, including recorded runs and settings. */ readonly documents: Readonly<
    Record<string, WorkspaceDocument>
  >;
  /** Original immutable revisions indexed by their verified document digests. */ readonly revisions: Readonly<
    Record<string, ExperimentRevision>
  >;
  /** Original immutable parameter declarations indexed by their unique keys. */ readonly parameterSpecs: Readonly<
    Record<string, ParameterSpec>
  >;
}

function take<T>(value: ParseResult<T>): T {
  if (!value.ok) throw new Error(`${value.path}: ${value.message}`);
  return value.value;
}
function record(value: unknown, keys: readonly string[], name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`${name}: object required`);
  const result = value as Record<string, unknown>;
  if (Object.keys(result).length !== keys.length || keys.some((key) => !Object.hasOwn(result, key)))
    throw new Error(`${name} fields are incomplete or unsupported`);
  return result;
}
function text(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0)
    throw new Error(`${name}: nonempty string required`);
  return value;
}
function safeName(value: unknown): string {
  const name = text(value, "member name");
  if (
    name.length > 4096 ||
    /[\\:%?#@]/.test(name) ||
    Array.from(name).some(
      (character) => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127,
    ) ||
    name.split("/").some((part) => ["", ".", ".."].includes(part))
  )
    throw new Error("unsafe member name");
  return name;
}
function digest(value: unknown): string {
  const result = text(value, "member digest");
  if (result.length !== 64 || !/^[0-9a-f]{64}$/.test(result))
    throw new Error("lowercase SHA-256 member digest required");
  return result;
}

/** Preview bounded JSON and admit every reference; callers may lower the expanded byte budget. */
export async function admitWorkspaceArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec> = new Map(),
  expandedByteLimit = maxExpandedBytes,
): Promise<AdmittedWorkspaceArchive> {
  if (
    !Number.isSafeInteger(expandedByteLimit) ||
    expandedByteLimit <= 0 ||
    expandedByteLimit > maxExpandedBytes
  )
    throw new Error("expanded byte budget must be a positive integer within the product bound");
  if (
    typeof json !== "string" ||
    json.length > maxArchiveBytes ||
    new TextEncoder().encode(json).byteLength > maxArchiveBytes
  )
    throw new Error("encoded archive limit exceeded");
  // The original lossless reader owns duplicate keys, exact scalars and depth<=64.
  const archive = record(
    readJson(json),
    ["schema", "manifest", "members", "parameter_units"],
    "archive",
  );
  if (archive["schema"] !== "quantum_workspace_archive.v1")
    throw new Error(
      "Unsupported archive schema; explicit supported migration is required before import",
    );
  const manifest = take(parseWorkspaceManifest(archive["manifest"]));
  const members = archive["members"];
  if (!Array.isArray(members) || members.length + 1 > maxArchiveMembers)
    throw new Error("archive member limit exceeded");
  const names = new Set<string>();
  const hashes = new Set<string>();
  const documents = new Map<string, WorkspaceDocument>();
  const raw = new Map<string, RawArtifact>();
  const originalCodecs = new Map(rawCodecs);
  let expanded = new TextEncoder().encode(writeJson(manifest)).byteLength;
  if (expanded > expandedByteLimit) throw new Error("expanded archive limit exceeded");
  // Complete shape/path/identity inventory precedes decoding and async producer calls.
  const entries = members.map((value) => {
    const member = record(value, ["name", "kind", "schema", "sha256", "content"], "member");
    const name = safeName(member["name"]);
    if (names.has(name)) throw new Error("duplicate archive member name");
    names.add(name);
    const hash = digest(member["sha256"]);
    if (hashes.has(hash)) throw new Error("duplicate archive identity");
    hashes.add(hash);
    const schema = text(member["schema"], "member schema");
    const content = member["content"];
    if (typeof content !== "string") throw new Error("member content: string required");
    const kind = member["kind"];
    if (kind !== "document" && kind !== "raw")
      throw new Error("unsupported archive member kind; executable members and links are refused");
    if (kind === "raw" && (content.length % 2 !== 0 || /[^0-9a-f]/.test(content)))
      throw new Error("canonical lowercase hex bytes required");
    expanded += kind === "raw" ? content.length / 2 : new TextEncoder().encode(content).byteLength;
    if (expanded > expandedByteLimit) throw new Error("expanded archive limit exceeded");
    return Object.freeze({ name, sha256: hash, schema, kind, content });
  });
  for (const entry of entries) {
    if (entry.kind === "document") {
      const document = take(parseDocument(readJson(entry.content)));
      if (document.schema !== entry.schema) throw new Error("member document schema mismatch");
      documents.set(entry.sha256, document);
    } else {
      const nativeDecoder = (
        Uint8Array as Uint8ArrayConstructor & {
          fromHex?: (hex: string) => Uint8Array<ArrayBuffer>;
        }
      ).fromHex;
      const content =
        typeof nativeDecoder === "function"
          ? nativeDecoder(entry.content)
          : new Uint8Array(entry.content.length / 2);
      if (typeof nativeDecoder !== "function")
        for (let index = 0; index < content.length; index++)
          content[index] = parseInt(entry.content.slice(2 * index, 2 * index + 2), 16);
      raw.set(entry.sha256, { schema: entry.schema, content });
      // Verify even unreferenced raw members; no orphan executable/unverified data slips in.
      const codec = originalCodecs.get(entry.schema);
      if (!codec) throw new Error("unsupported raw producer/schema");
      const identity = await codec(new Uint8Array(content));
      if (
        identity.digest !== entry.sha256 ||
        identity.schema !== entry.schema ||
        typeof identity.kind !== "string" ||
        identity.kind.length === 0
      )
        throw new Error("raw producer identity mismatch");
    }
  }
  const specs = new Map<string, ParameterSpec>();
  for (const document of documents.values()) {
    if (document.schema !== "parameter_spec.v1") continue;
    const spec = take(parseParameterSpec(document));
    const key = spec.body["key"] as string;
    if (specs.has(key)) throw new Error("duplicate parameter specification key");
    specs.set(key, spec);
  }
  const unitPayload = archive["parameter_units"];
  if (typeof unitPayload !== "object" || unitPayload === null || Array.isArray(unitPayload))
    throw new Error("parameter units object required");
  const units = new Map(
    Object.entries(unitPayload).map(([key, value]) => [key, text(value, "parameter unit")]),
  );
  const admission = take(
    await admitWorkspace(manifest, documents, raw, originalCodecs, specs, units),
  );
  const preview: WorkspaceArchivePreview = Object.freeze({
    schema: "quantum_workspace_archive.v1",
    projectId: admission.projectId,
    archiveDigest: await canonicalDigest(archiveSnapshotDomain, json),
    workspaceHash: admission.workspaceHash,
    documentHashes: admission.documentHashes,
    rawHashes: Object.freeze([...raw.keys()].sort()),
    memberNames: Object.freeze([...names]),
    json,
  });
  const revisions = [...documents].filter(
    (entry): entry is [string, ExperimentRevision] => entry[1].schema === "experiment_revision.v1",
  );
  return Object.freeze({
    preview,
    manifest,
    members: Object.freeze(entries),
    documents: Object.freeze(Object.fromEntries(documents)),
    parameterUnits: Object.freeze(Object.fromEntries(units)),
    revisions: Object.freeze(Object.fromEntries(revisions)),
    parameterSpecs: Object.freeze(Object.fromEntries(specs)),
  });
}

/** Preview the original bounded archive without exposing editing inputs or execution authority. */
export async function previewWorkspaceArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec> = new Map(),
  expandedByteLimit = maxExpandedBytes,
): Promise<WorkspaceArchivePreview> {
  return (await admitWorkspaceArchive(json, rawCodecs, expandedByteLimit)).preview;
}

/** Build a lossless portable archive; original binary bytes remain byte-identical hex. */
export async function createWorkspaceArchive(
  manifest: WorkspaceManifest,
  documents: ReadonlyMap<string, WorkspaceDocument>,
  raw: ReadonlyMap<string, RawArtifact>,
  parameterUnits: ReadonlyMap<string, string>,
  rawCodecs: ReadonlyMap<string, RawCodec> = new Map(),
): Promise<WorkspaceArchivePreview> {
  if (documents.size + raw.size + 1 > maxArchiveMembers)
    throw new Error("archive member limit exceeded");
  const members = [
    ...[...documents].map(([hash, document]) => ({
      name: `documents/${hash}.json`,
      kind: "document",
      schema: document.schema,
      sha256: hash,
      content: writeJson(document),
    })),
    ...[...raw].map(([hash, artifact]) => ({
      name: `raw/${hash}.bin`,
      kind: "raw",
      schema: artifact.schema,
      sha256: hash,
      content: Array.from(artifact.content, (byte) => byte.toString(16).padStart(2, "0")).join(""),
    })),
  ];
  const json = writeJson({
    schema: "quantum_workspace_archive.v1",
    manifest,
    members,
    parameter_units: Object.fromEntries(parameterUnits),
  });
  return previewWorkspaceArchive(json, rawCodecs);
}
