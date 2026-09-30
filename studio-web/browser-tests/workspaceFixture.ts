// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native browser workspace conformance

import { canonicalDigest, documentDigest, documentToWire, parseDocument, parseParameterSpec, parseWorkspaceManifest, readJson } from "../src/shared/contracts";
import type { ParameterSpec, ParseResult, RawArtifact, RawCodec, RawIdentity, WorkspaceDocument, WorkspaceManifest } from "../src/shared/contracts";
import { createWorkspaceArchive } from "../src/shared/storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../src/shared/storage/workspaceArchive";

/** Explicit synthetic metadata oracle; never imported or registered by the production panel. */
export async function conformanceCodec(content: Uint8Array): Promise<RawIdentity> {
  const value = readJson(new TextDecoder("utf-8", { fatal: true }).decode(content)) as { schema: string; body: { role: string; synthetic: boolean } };
  if (value.schema !== "review_fixture.v1" || value.body.synthetic !== true || !["problem", "program", "policy", "environment", "plan"].includes(value.body.role)) throw new Error("Synthetic review conformance producer required");
  return { schema: value.schema, kind: value.body.role, digest: await canonicalDigest(value.schema, value) };
}
/** Trusted, code-supplied conformance registry only, with no provider or solver support claim. */
export const conformanceCodecs: ReadonlyMap<string, RawCodec> = new Map([["review_fixture.v1", conformanceCodec]]);

function take<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(`${result.path}: ${result.message}`);
  return result.value;
}

/** Build full original workspace corpus indexes using native WebCrypto and source-owned contracts. */
export async function conformanceArchive(corpusText: string, edited: boolean): Promise<WorkspaceArchivePreview> {
  const corpus = readJson(corpusText) as { schema: string; fixtures: Record<string, unknown> };
  if (corpus.schema !== "studio_workspace_document_conformance.v1") throw new Error("Original workspace conformance corpus required");
  const fixtures = corpus.fixtures;
  const documents = new Map<string, WorkspaceDocument>();
  for (const name of ["settings", "parameter", "revision_root", "revision_child", "run"]) {
    const document = take(parseDocument(fixtures[name]));
    documents.set(await documentDigest(document), document);
  }
  const raw = new Map<string, RawArtifact>();
  const { writeJson } = await import("../src/shared/contracts");
  for (const name of ["problem", "program", "policy", "environment", "plan"]) {
    const content = new TextEncoder().encode(writeJson(fixtures[name]));
    const identity = await conformanceCodec(content);
    raw.set(identity.digest, { schema: identity.schema, content });
  }
  let manifest: WorkspaceManifest = take(parseWorkspaceManifest(fixtures["workspace"]));
  if (edited) {
    const original = take(parseDocument(fixtures["revision_child"]));
    const priorHash = await documentDigest(original);
    const wire = documentToWire(original);
    (wire["body"] as Record<string, unknown>)["parent_revision_hashes"] = [priorHash];
    (wire["extensions"] as Record<string, unknown>)["draft_note"] = "Edited synthetic metadata; no execution or scientific evidence";
    const revision = take(parseDocument(wire));
    const hash = await documentDigest(revision);
    documents.set(hash, revision);
    const root = documentToWire(manifest);
    const body = root["body"] as Record<string, unknown>;
    const reference = { schema: revision.schema, sha256: hash, media_type: "application/json" };
    body["revision_refs"] = [...body["revision_refs"] as readonly unknown[], reference];
    body["draft_ref"] = reference;
    manifest = take(parseWorkspaceManifest(root));
  }
  // Independent units are fixed by the original explicit synthetic parameter fixture.
  const parameter: ParameterSpec = take(parseParameterSpec(fixtures["parameter"]));
  return createWorkspaceArchive(manifest, documents, raw, new Map([[parameter.body["key"] as string, parameter.body["unit"] as string]]), conformanceCodecs);
}
