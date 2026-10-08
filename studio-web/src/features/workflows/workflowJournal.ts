// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workflow attempt checkpoint admission

import { canonicalDigest, readJson, writeJson } from "../../shared/contracts";
import { dataEntries } from "../../shared/contracts/canonical";
import { topologicalOrder, WorkflowRefusal, workflowDocument } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";
import { buildWorkflowCells } from "./workflowSweep";

/** Portable original attempt history; importing it never grants execution authority. */
export const workflowJournalSchema = "experiment_workflow_journal.v1";
/** Maximum encoded checkpoint size before admission. */
export const maxWorkflowJournalBytes = 64 * 1024 * 1024;
/** Explicit original outcome; incomplete attempts never become cached successes. */
export type WorkflowAttemptStatus =
  | "running"
  | "complete"
  | "failed"
  | "gated"
  | "blocked"
  | "cancelled"
  | "interrupted";
/** One original stage attempt with unchanged producer output or partial diagnostics. */
export interface WorkflowAttempt {
  /** Exact admitted coordinate identity. */ readonly cell_id: string;
  /** Original stage identity. */ readonly stage_id: string;
  /** Adapter-derived source/runtime/input/plan/dependency identity. */ readonly fingerprint: string;
  /** Original parent output digests; null records an incomplete parent. */ readonly dependencies: Readonly<
    Record<string, string | null>
  >;
  /** Explicit terminal or unfinished state. */ readonly status: WorkflowAttemptStatus;
  /** Unchanged original output; null means none was received. */ readonly output: unknown;
  /** Canonical original output identity; null only when output is absent. */ readonly output_digest:
    | string
    | null;
  /** Bounded explicit incomplete reason; null on completion. */ readonly reason: string | null;
  /** Whether this attempt consumed one stage-evaluation unit. */ readonly evaluated: boolean;
}
/** Immutable bounded checkpoint; adapters must independently verify producer evidence before reuse. */
export interface WorkflowJournal {
  /** Exact whole graph identity. */ readonly workflow_digest: string;
  /** Actual original source identity, independently obtained by the host. */ readonly source_fingerprint: string;
  /** Actual runtime/build identity, independently obtained by the host. */ readonly runtime_fingerprint: string;
  /** Chronological history retaining every partial and failed attempt. */ readonly entries: readonly WorkflowAttempt[];
  /** Whole graph completion; partial and cancelled remain explicit. */ readonly state:
    | "partial"
    | "cancelled"
    | "complete";
  /** Actual reserved stage evaluations, including failures. */ readonly evaluations: bigint;
  /** Opaque preserved metadata, with no authority inferred. */ readonly extensions: Readonly<
    Record<string, unknown>
  >;
}

const statuses: readonly string[] = [
  "running",
  "complete",
  "failed",
  "gated",
  "blocked",
  "cancelled",
  "interrupted",
];
function digest(value: unknown, name: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value))
    throw new WorkflowRefusal(`${name}: exact SHA-256 identity required`);
  return value;
}
function object(value: unknown, name: string, fields?: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new WorkflowRefusal(`${name}: complete supported object required`);
  const result = Object.fromEntries(dataEntries(value));
  if (
    fields !== undefined &&
    (Object.keys(result).length !== fields.length ||
      fields.some((key) => !Object.hasOwn(result, key)))
  )
    throw new WorkflowRefusal(`${name}: complete supported object required`);
  return result;
}
function freeze(value: unknown): unknown {
  if (Array.isArray(value)) return Object.freeze(value.map(freeze));
  if (typeof value === "object" && value !== null)
    return Object.freeze(
      Object.fromEntries(dataEntries(value).map(([key, item]) => [key, freeze(item)])),
    );
  return value;
}

/** Return independent lossless wire data without dropping original attempts or opaque extensions. */
export function workflowJournalDocument(journal: WorkflowJournal): Record<string, unknown> {
  return object(
    readJson(
      writeJson({
        schema: workflowJournalSchema,
        body: {
          workflow_digest: journal.workflow_digest,
          source_fingerprint: journal.source_fingerprint,
          runtime_fingerprint: journal.runtime_fingerprint,
          entries: journal.entries,
          state: journal.state,
          evaluations: journal.evaluations,
        },
        extensions: journal.extensions,
      }),
    ),
    "journal",
  );
}

/** Admit graph, output, dependency and evaluation identities; structural admission is not producer attestation. */
export async function parseWorkflowJournal(
  payload: unknown,
  definition: WorkflowDefinition,
): Promise<WorkflowJournal> {
  const text = writeJson(payload);
  if (new TextEncoder().encode(text).length > maxWorkflowJournalBytes)
    throw new WorkflowRefusal("workflow journal exceeds byte bound");
  const document = object(readJson(text), "journal", ["schema", "body", "extensions"]);
  if (document["schema"] !== workflowJournalSchema)
    throw new WorkflowRefusal("unsupported workflow journal schema");
  const body = object(document["body"], "journal body", [
    "workflow_digest",
    "source_fingerprint",
    "runtime_fingerprint",
    "entries",
    "state",
    "evaluations",
  ]);
  const cells = await buildWorkflowCells(definition);
  const workflow_digest = await canonicalDigest(
    "studio.workflow-definition.v1",
    workflowDocument(definition),
  );
  if (digest(body["workflow_digest"], "workflow") !== workflow_digest)
    throw new WorkflowRefusal("journal belongs to another workflow definition");
  const source_fingerprint = digest(body["source_fingerprint"], "original source");
  const runtime_fingerprint = digest(body["runtime_fingerprint"], "actual runtime");
  const stages = new Map(definition.stages.map((stage) => [stage.id, stage]));
  const cellIds = new Set(cells.map((cell) => cell.id));
  const rawEntries = body["entries"];
  if (!Array.isArray(rawEntries) || rawEntries.length > 8192)
    throw new WorkflowRefusal("bounded original journal entries required");
  const entries: WorkflowAttempt[] = [];
  const latest = new Map<string, WorkflowAttempt>();
  let evaluations = 0n;
  for (const raw of rawEntries) {
    const row = object(raw, "attempt", [
      "cell_id",
      "stage_id",
      "fingerprint",
      "dependencies",
      "status",
      "output",
      "output_digest",
      "reason",
      "evaluated",
    ]);
    const cid = row["cell_id"],
      sid = row["stage_id"];
    if (typeof cid !== "string" || !cellIds.has(cid) || typeof sid !== "string" || !stages.has(sid))
      throw new WorkflowRefusal("attempt coordinate or stage is absent");
    const stage = stages.get(sid) as WorkflowDefinition["stages"][number];
    const parents = new Set([
      ...stage.depends_on,
      ...stage.inputs.map((input) => input.source_stage),
    ]);
    const rawDependencies = object(row["dependencies"], "attempt dependencies");
    if (
      Object.keys(rawDependencies).length !== parents.size ||
      Object.keys(rawDependencies).some((parent) => !parents.has(parent))
    )
      throw new WorkflowRefusal("attempt dependencies differ from original graph");
    const dependencies = Object.fromEntries(
      Object.entries(rawDependencies).map(([parent, value]) => [
        parent,
        value === null ? null : digest(value, "parent output"),
      ]),
    );
    const status = row["status"],
      reason = row["reason"];
    if (typeof status !== "string" || !statuses.includes(status))
      throw new WorkflowRefusal("unsupported original attempt status");
    if (status === "complete") {
      if (reason !== null || row["output"] === null)
        throw new WorkflowRefusal(
          "completed attempt requires original output without failure reason",
        );
      for (const [parent, parentDigest] of Object.entries(dependencies)) {
        const original = latest.get(`${cid}:${parent}`);
        if (
          original === undefined ||
          original.status !== "complete" ||
          parentDigest !== original.output_digest
        )
          throw new WorkflowRefusal("completed attempt lacks its original completed dependency");
      }
    } else if (
      typeof reason !== "string" ||
      Array.from(reason).length < 1 ||
      Array.from(reason).length > 2048
    )
      throw new WorkflowRefusal("incomplete attempt requires a bounded original reason");
    const output = row["output"],
      outputDigest = row["output_digest"];
    if (output === null) {
      if (outputDigest !== null)
        throw new WorkflowRefusal("absent output cannot carry an output digest");
    } else if (
      digest(outputDigest, "original output") !==
      (await canonicalDigest("studio.workflow-output.v1", output))
    )
      throw new WorkflowRefusal("original attempt output digest differs");
    const evaluated = row["evaluated"];
    if (
      typeof evaluated !== "boolean" ||
      (status === "complete" && !evaluated) ||
      (status === "blocked" && evaluated)
    )
      throw new WorkflowRefusal("attempt evaluation accounting differs");
    const previous = latest.get(`${cid}:${sid}`);
    if (previous?.status === "complete" && previous.fingerprint === row["fingerprint"])
      throw new WorkflowRefusal("completed matching stage cannot be duplicated");
    if (evaluated) evaluations += 1n;
    const attempt: WorkflowAttempt = Object.freeze({
      cell_id: cid,
      stage_id: sid,
      fingerprint: digest(row["fingerprint"], "stage input"),
      dependencies: Object.freeze(dependencies),
      status: status as WorkflowAttemptStatus,
      output: freeze(output),
      output_digest: outputDigest === null ? null : digest(outputDigest, "original output"),
      reason: reason === null ? null : (reason as string),
      evaluated,
    });
    entries.push(attempt);
    latest.set(`${cid}:${sid}`, attempt);
  }
  if (
    typeof body["evaluations"] !== "bigint" ||
    body["evaluations"] !== evaluations ||
    evaluations > definition.sweep.evaluation_budget
  )
    throw new WorkflowRefusal("journal stage evaluation budget or accounting differs");
  const state = body["state"];
  if (state !== "partial" && state !== "cancelled" && state !== "complete")
    throw new WorkflowRefusal("unsupported journal completion state");
  if (state === "complete") {
    for (const cell of cells)
      for (const sid of topologicalOrder(definition)) {
        const entry = latest.get(`${cell.id}:${sid}`);
        if (entry === undefined || entry.status !== "complete")
          throw new WorkflowRefusal("journal has incomplete original coordinates");
        for (const [parent, parentDigest] of Object.entries(entry.dependencies)) {
          const original = latest.get(`${cell.id}:${parent}`) as WorkflowAttempt;
          if (original.status !== "complete" || parentDigest !== original.output_digest)
            throw new WorkflowRefusal("completed journal dependency has changed");
        }
      }
  }
  const extensions = object(document["extensions"], "journal extensions");
  return Object.freeze({
    workflow_digest,
    source_fingerprint,
    runtime_fingerprint,
    entries: Object.freeze(entries),
    state,
    evaluations,
    extensions: Object.freeze(
      Object.fromEntries(Object.entries(extensions).map(([key, value]) => [key, freeze(value)])),
    ),
  });
}

/** Create an empty partial checkpoint bound to actual source/build identities supplied by the owning host. */
export async function createWorkflowJournal(
  definition: WorkflowDefinition,
  source_fingerprint: string,
  runtime_fingerprint: string,
): Promise<WorkflowJournal> {
  return parseWorkflowJournal(
    {
      schema: workflowJournalSchema,
      body: {
        workflow_digest: await canonicalDigest(
          "studio.workflow-definition.v1",
          workflowDocument(definition),
        ),
        source_fingerprint,
        runtime_fingerprint,
        entries: [],
        state: "partial",
        evaluations: 0n,
      },
      extensions: {},
    },
    definition,
  );
}
