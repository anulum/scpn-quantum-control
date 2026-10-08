// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound immutable parameter revision

import { documentDigest, parseWorkspaceManifest, writeJson } from "../../shared/contracts";
import type { ExperimentRevision, RawCodec } from "../../shared/contracts";
import {
  admitWorkspaceArchive,
  previewWorkspaceArchive,
} from "../../shared/storage/workspaceArchive";
import type {
  AdmittedWorkspaceArchive,
  WorkspaceArchiveMember,
  WorkspaceArchivePreview,
} from "../../shared/storage/workspaceArchive";
import type { ParameterDraftSource, ParameterSnapshot } from "./parameterDraft";
import { createParameterDraft, validateParameterSnapshot } from "./parameterDraft";

/** Create a v1 child revision; source records, settings, results and prior references remain untouched. */
export async function createParameterRevision(
  source: ParameterDraftSource,
  candidate: ParameterSnapshot,
): Promise<ExperimentRevision> {
  const captured = createParameterDraft(source).source;
  const snapshot = validateParameterSnapshot(captured, candidate);
  const original = captured.revision;
  const parent = await documentDigest(original);
  const revision: ExperimentRevision = Object.freeze({
    schema: original.schema,
    body: Object.freeze({
      ...original.body,
      parent_revision_hashes: Object.freeze([parent]),
      parameters: snapshot.parameters,
    }),
    extensions: Object.freeze({
      ...original.extensions,
      parameter_editor: Object.freeze({ version: 1n, trainable_masks: snapshot.trainableMasks }),
    }),
  });
  await documentDigest(revision);
  return revision;
}

function sourceFromAdmission(
  archive: AdmittedWorkspaceArchive,
  revisionHash: string,
): ParameterDraftSource;
function sourceFromAdmission(
  archive: AdmittedWorkspaceArchive,
  revisionHash?: string,
): ParameterDraftSource | null;
function sourceFromAdmission(
  archive: AdmittedWorkspaceArchive,
  revisionHash?: string,
): ParameterDraftSource | null {
  const refs = archive.manifest.body["revision_refs"] as readonly { readonly sha256: string }[];
  const draft = archive.manifest.body["draft_ref"] as { readonly sha256: string } | null;
  const hash = revisionHash ?? draft?.sha256 ?? refs.at(-1)?.sha256;
  if (hash === undefined) return null;
  if (!refs.some((ref) => ref.sha256 === hash) && draft?.sha256 !== hash)
    throw new Error("Parameter revision must be listed in the admitted project");
  const document = archive.revisions[hash] as ExperimentRevision;
  const keys = Object.keys(document.body["parameters"] as Readonly<Record<string, unknown>>);
  const specs = Object.values(archive.parameterSpecs).filter((spec) =>
    keys.includes(spec.body["key"] as string),
  );
  const units = Object.fromEntries(keys.map((key) => [key, archive.parameterUnits[key] as string]));
  return Object.freeze({
    revision: document,
    specs: Object.freeze(specs),
    units: Object.freeze(units),
  });
}

/** Re-admit a portable archive and select its exact source revision without fetching or executing. */
export function parameterSourceFromArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  revisionHash: string,
): Promise<ParameterDraftSource>;
/** Select the draft or latest revision; an empty project returns null. */
export function parameterSourceFromArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  revisionHash?: string,
): Promise<ParameterDraftSource | null>;
/** Admit the original archive before resolving its selected parameter source. */
export async function parameterSourceFromArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  revisionHash?: string,
): Promise<ParameterDraftSource | null> {
  return sourceFromAdmission(await admitWorkspaceArchive(json, rawCodecs), revisionHash);
}

/** Append a fully admitted child to the exact original archive, retaining every prior member byte. */
export async function appendParameterRevision(
  json: string,
  source: ParameterDraftSource,
  snapshot: ParameterSnapshot,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  updatedAt: string,
): Promise<{
  /** Fully admitted candidate, to be saved by the original workspace store. */ readonly archive: WorkspaceArchivePreview;
  /** Exact original v1 document digest of the appended child revision. */ readonly revisionHash: string;
}> {
  const original = await admitWorkspaceArchive(json, rawCodecs);
  const parentHash = await documentDigest(source.revision);
  const admitted = sourceFromAdmission(original, parentHash);
  const revision = await createParameterRevision(admitted, snapshot);
  const revisionHash = await documentDigest(revision);
  if (original.members.some((member) => member.sha256 === revisionHash))
    throw new Error(
      "That immutable parameter revision already exists; make another edit before saving",
    );
  const ref = { schema: revision.schema, sha256: revisionHash, media_type: "application/json" };
  const manifest = parseWorkspaceManifest({
    ...original.manifest,
    body: {
      ...original.manifest.body,
      revision_refs: [...(original.manifest.body["revision_refs"] as readonly unknown[]), ref],
      draft_ref: ref,
      updated_at: updatedAt,
    },
  });
  if (!manifest.ok) throw new Error(`${manifest.path}: ${manifest.message}`);
  const member: WorkspaceArchiveMember = {
    name: `documents/${revisionHash}.json`,
    kind: "document",
    schema: revision.schema,
    sha256: revisionHash,
    content: writeJson(revision),
  };
  const archive = await previewWorkspaceArchive(
    writeJson({
      schema: original.preview.schema,
      manifest: manifest.value,
      members: [...original.members, member],
      parameter_units: original.parameterUnits,
    }),
    rawCodecs,
  );
  return Object.freeze({ archive, revisionHash });
}
