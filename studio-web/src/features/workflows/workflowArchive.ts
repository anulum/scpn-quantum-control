// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workspace workflow checkpoint retention

import { canonicalDigest, writeJson } from "../../shared/contracts";
import type { RawCodec } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";
import {
  admitWorkspaceArchive,
  previewWorkspaceArchive,
} from "../../shared/storage/workspaceArchive";
import type {
  AdmittedWorkspaceArchive,
  WorkspaceArchivePreview,
} from "../../shared/storage/workspaceArchive";
import { parseWorkflowJournal, workflowJournalDocument } from "./workflowJournal";
import type { WorkflowJournal } from "./workflowJournal";
import { parseWorkflow, WorkflowRefusal, workflowDocument } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";

/** Bounded retained graph definitions; replacing a graph never drops its prior partial journal. */
export const maxArchivedWorkflows = 32;
/** Original graph and journal associated with one explicitly selected immutable source. */
export interface ArchivedWorkflow {
  /** Exact whole definition identity. */ readonly hash: string;
  /** Original immutable experiment revision, still present in the admitted archive. */ readonly base_revision_hash: string;
  /** Admitted original graph; no backend authority is inferred. */ readonly definition: WorkflowDefinition;
  /** Original complete and partial attempts, or null before any run. */ readonly journal: WorkflowJournal | null;
}
/** Whole source admitted before workflow metadata is released to a consumer. */
export interface WorkflowArchive {
  /** Original archive admission; raw outputs remain stored once in its members. */ readonly source: AdmittedWorkspaceArchive;
  /** Explicit selected graph, or null when none has been composed. */ readonly selected:
    | string
    | null;
  /** Prior graph and partial result history retained in original order. */ readonly workflows: readonly ArchivedWorkflow[];
}

function object(value: unknown, fields: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new WorkflowRefusal("workflow archive metadata object required");
  const result = Object.fromEntries(dataEntries(value));
  if (
    Object.keys(result).length !== fields.length ||
    fields.some((key) => !Object.hasOwn(result, key))
  )
    throw new WorkflowRefusal("workflow archive fields are incomplete or unsupported");
  return result;
}
function identity(value: unknown): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value))
    throw new WorkflowRefusal("exact original workflow/revision identity required");
  return value;
}

/** Admit the complete original archive and all retained graph/checkpoint metadata without executing or saving. */
export async function readWorkflowArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
): Promise<WorkflowArchive> {
  const source = await admitWorkspaceArchive(json, rawCodecs);
  const extension = source.manifest.extensions["experiment_workflows"];
  if (extension === undefined)
    return Object.freeze({ source, selected: null, workflows: Object.freeze([]) });
  const metadata = object(extension, ["version", "selected", "items"]);
  if (
    metadata["version"] !== 1n ||
    !Array.isArray(metadata["items"]) ||
    metadata["items"].length < 1 ||
    metadata["items"].length > maxArchivedWorkflows
  )
    throw new WorkflowRefusal("unsupported workflow archive version or history bound");
  const workflows: ArchivedWorkflow[] = [];
  const seen = new Set<string>();
  for (const item of metadata["items"]) {
    const row = object(item, ["hash", "base_revision_hash", "definition", "journal"]);
    const hash = identity(row["hash"]),
      base_revision_hash = identity(row["base_revision_hash"]);
    const definition = parseWorkflow(row["definition"]);
    if (
      seen.has(hash) ||
      hash !==
        (await canonicalDigest("studio.workflow-definition.v1", workflowDocument(definition)))
    )
      throw new WorkflowRefusal("duplicate or changed original archived workflow");
    if (source.revisions[base_revision_hash] === undefined)
      throw new WorkflowRefusal("original workflow baseline revision is absent");
    const journal =
      row["journal"] === null ? null : await parseWorkflowJournal(row["journal"], definition);
    seen.add(hash);
    workflows.push(Object.freeze({ hash, base_revision_hash, definition, journal }));
  }
  const selected = identity(metadata["selected"]);
  if (!seen.has(selected)) throw new WorkflowRefusal("selected original workflow is absent");
  return Object.freeze({ source, selected, workflows: Object.freeze(workflows) });
}

async function retainHistory(previous: WorkflowJournal, next: WorkflowJournal): Promise<void> {
  if (
    previous.source_fingerprint !== next.source_fingerprint ||
    previous.runtime_fingerprint !== next.runtime_fingerprint ||
    next.entries.length < previous.entries.length ||
    next.evaluations < previous.evaluations
  )
    throw new WorkflowRefusal("original checkpoint source, runtime or attempt history changed");
  for (const [index, before] of previous.entries.entries()) {
    const after = next.entries[index] as WorkflowJournal["entries"][number];
    const digest = await canonicalDigest("studio.workflow-attempt.v1", before);
    if (digest === (await canonicalDigest("studio.workflow-attempt.v1", after))) continue;
    if (before.status !== "running" || after.status === "running")
      throw new WorkflowRefusal("original completed or partial attempt cannot be rewritten");
    const original = {
      ...before,
      status: after.status,
      output: after.output,
      output_digest: after.output_digest,
      reason: after.reason,
    };
    if (
      (await canonicalDigest("studio.workflow-attempt.v1", original)) !==
      (await canonicalDigest("studio.workflow-attempt.v1", after))
    )
      throw new WorkflowRefusal("original reserved attempt identity changed");
  }
}

/** Preview an additive original workspace save; original transactions still guard the exact prior source text. */
export async function archiveWorkflow(
  json: string,
  baseRevision: string,
  definition: WorkflowDefinition,
  journal: WorkflowJournal | null,
  rawCodecs: ReadonlyMap<string, RawCodec>,
): Promise<WorkspaceArchivePreview> {
  const current = await readWorkflowArchive(json, rawCodecs);
  const base_revision_hash = identity(baseRevision);
  if (current.source.revisions[base_revision_hash] === undefined)
    throw new WorkflowRefusal("original workflow baseline revision is absent");
  const admitted = parseWorkflow(workflowDocument(definition));
  const hash = await canonicalDigest("studio.workflow-definition.v1", workflowDocument(admitted));
  const next =
    journal === null
      ? null
      : await parseWorkflowJournal(workflowJournalDocument(journal), admitted);
  const prior = current.workflows.find((item) => item.hash === hash);
  if (prior !== undefined) {
    if (prior.base_revision_hash !== base_revision_hash)
      throw new WorkflowRefusal("original graph cannot be rebound to another baseline revision");
    if (prior.journal !== null) {
      if (next === null)
        throw new WorkflowRefusal("original partial or completed journal cannot be discarded");
      await retainHistory(prior.journal, next);
    }
  } else if (current.workflows.length >= maxArchivedWorkflows)
    throw new WorkflowRefusal(
      "workflow history bound reached; export original evidence before composing another graph",
    );
  const item: ArchivedWorkflow = { hash, base_revision_hash, definition: admitted, journal: next };
  const retained =
    prior === undefined
      ? [...current.workflows, item]
      : current.workflows.map((entry) => (entry.hash === hash ? item : entry));
  const original = current.source;
  const manifest = {
    ...original.manifest,
    extensions: {
      ...original.manifest.extensions,
      experiment_workflows: {
        version: 1n,
        selected: hash,
        items: retained.map((entry) => ({
          hash: entry.hash,
          base_revision_hash: entry.base_revision_hash,
          definition: workflowDocument(entry.definition),
          journal: entry.journal === null ? null : workflowJournalDocument(entry.journal),
        })),
      },
    },
  };
  return previewWorkspaceArchive(
    writeJson({
      schema: original.preview.schema,
      manifest,
      members: original.members,
      parameter_units: original.parameterUnits,
    }),
    rawCodecs,
  );
}

/** Select an admitted immutable revision through the original manifest without dropping evidence or workflow history. */
export async function selectWorkflowRevision(
  json: string,
  revisionHash: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
  recordHash?: string,
): Promise<WorkspaceArchivePreview> {
  const original = await admitWorkspaceArchive(json, rawCodecs);
  const revision = original.revisions[identity(revisionHash)];
  if (revision === undefined) throw new WorkflowRefusal("original workflow revision is absent");
  let references = original.manifest.body["artefact_refs"] as readonly {
    readonly sha256: string;
  }[];
  if (recordHash !== undefined) {
    const hash = identity(recordHash);
    if (
      !references.some((ref) => ref.sha256 === hash) ||
      original.documents[hash]?.schema !== "local_run_record.v1"
    )
      throw new WorkflowRefusal("original indexed workflow run is absent");
    references = [
      ...references.filter((ref) => ref.sha256 !== hash),
      ...references.filter((ref) => ref.sha256 === hash),
    ];
  }
  const manifest = {
    ...original.manifest,
    body: {
      ...original.manifest.body,
      draft_ref: { schema: revision.schema, sha256: revisionHash, media_type: "application/json" },
      artefact_refs: references,
    },
  };
  return previewWorkspaceArchive(
    writeJson({
      schema: original.preview.schema,
      manifest,
      members: original.members,
      parameter_units: original.parameterUnits,
    }),
    rawCodecs,
  );
}
