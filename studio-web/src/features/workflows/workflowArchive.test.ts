// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workflow archive public tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { instantiateKuramoto } from "../../panel/kuramoto";
import {
  canonicalBytes,
  canonicalDigest,
  documentDigest,
  readJson,
  writeJson,
} from "../../shared/contracts";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { createParameterDraft, parameterDraftReducer } from "../parameters/parameterDraft";
import {
  appendParameterRevision,
  parameterSourceFromArchive,
} from "../parameters/parameterRevision";
import {
  archiveWorkflow,
  maxArchivedWorkflows,
  readWorkflowArchive,
  selectWorkflowRevision,
} from "./workflowArchive";
import { parseWorkflowJournal, workflowJournalDocument } from "./workflowJournal";
import { parseWorkflow, workflowDocument } from "./workflowModel";

beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  vi.unstubAllGlobals();
});
const wasm = new Uint8Array(
  readFileSync(
    process.env["STUDIO_EXPERIMENT_WASM_PATH"] ??
      "../scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
  ),
);

async function originals() {
  const kernel = await instantiateKuramoto(wasm);
  const archive = await createLocalExperiment(
    kernel,
    { mode: "mean-field", omega: [0.2, 0.2], theta0: [0, 0.8], coupling: 1.4, dt: 0.01, steps: 4 },
    "original workflow archive storage fixture",
    localExperimentCodecs,
    "native storage test",
  );
  const source = await readLocalExperiment(archive.json, localExperimentCodecs);
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  const definition = parseWorkflow(corpus["workflow"]);
  const journal = await parseWorkflowJournal(corpus["original_checkpoint"], definition);
  return { archive, source, definition, journal };
}

it("retains original member bytes, revisions and opaque extensions in an additive graph save", async () => {
  const { archive, source, definition } = await originals();
  const plain = readJson(archive.json) as Record<string, unknown>;
  const manifest = plain["manifest"] as Record<string, unknown>;
  const extensions = manifest["extensions"] as Record<string, unknown>;
  extensions["owner_note"] = { large: 9007199254740993n, zero: -0 };
  const prior = writeJson(plain);
  const saved = await archiveWorkflow(
    prior,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  const recovered = await readWorkflowArchive(saved.json, localExperimentCodecs);
  expect(recovered.workflows).toHaveLength(1);
  expect(recovered.selected).toBe(recovered.workflows[0]?.hash);
  expect(recovered.workflows[0]?.journal).toBeNull();
  expect(recovered.source.members).toEqual(source.archive.members);
  expect(
    await documentDigest(recovered.source.revisions[source.revisionHash] as typeof source.revision),
  ).toBe(source.revisionHash);
  expect(canonicalBytes("opaque.v1", recovered.source.manifest.extensions["owner_note"])).toEqual(
    canonicalBytes("opaque.v1", extensions["owner_note"]),
  );
  expect((await readWorkflowArchive(prior, localExperimentCodecs)).selected).toBeNull();
});

it("keeps the actual incomplete Python compiler journal when composing a different graph", async () => {
  const { archive, source, definition, journal } = await originals();
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    journal,
    localExperimentCodecs,
  );
  const next = { ...definition, workflow_id: "another-compile-sweep" };
  const changed = await archiveWorkflow(
    saved.json,
    source.revisionHash,
    next,
    null,
    localExperimentCodecs,
  );
  const recovered = await readWorkflowArchive(changed.json, localExperimentCodecs);
  expect(recovered.workflows).toHaveLength(2);
  expect(recovered.workflows[0]?.journal?.state).toBe("partial");
  expect(
    canonicalBytes(
      "original-journal.v1",
      workflowJournalDocument(recovered.workflows[0]?.journal as typeof journal),
    ),
  ).toEqual(canonicalBytes("original-journal.v1", workflowJournalDocument(journal)));
  expect(recovered.workflows[1]?.journal).toBeNull();
  expect(recovered.selected).toBe(recovered.workflows[1]?.hash);
  expect(recovered.source.members).toEqual(source.archive.members);
});

it("refuses to discard or rewrite an original completed attempt", async () => {
  const { archive, source, definition, journal } = await originals();
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    journal,
    localExperimentCodecs,
  );
  const before = saved.json;
  await expect(
    archiveWorkflow(saved.json, source.revisionHash, definition, null, localExperimentCodecs),
  ).rejects.toThrow("discarded");
  const wire = workflowJournalDocument(journal);
  const body = wire["body"] as Record<string, unknown>;
  const row = (body["entries"] as [Record<string, unknown>])[0];
  row["status"] = "failed";
  row["reason"] = "changed original outcome";
  const changed = await parseWorkflowJournal(wire, definition);
  await expect(
    archiveWorkflow(saved.json, source.revisionHash, definition, changed, localExperimentCodecs),
  ).rejects.toThrow("rewritten");
  expect(saved.json).toBe(before);
});

it("refuses missing baseline and malformed workflow metadata without changing the prior archive", async () => {
  const { archive, source, definition } = await originals();
  await expect(
    archiveWorkflow(archive.json, "a".repeat(64), definition, null, localExperimentCodecs),
  ).rejects.toThrow("baseline revision");
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  for (const fault of ["version", "selected", "duplicate", "definition", "baseline"]) {
    const wire = readJson(saved.json) as Record<string, unknown>;
    const manifest = wire["manifest"] as Record<string, unknown>;
    const extension = (manifest["extensions"] as Record<string, unknown>)[
      "experiment_workflows"
    ] as Record<string, unknown>;
    const items = extension["items"] as [Record<string, unknown>];
    if (fault === "version") extension["version"] = 2n;
    else if (fault === "selected") extension["selected"] = "a".repeat(64);
    else if (fault === "duplicate") items.push(items[0]);
    else if (fault === "definition") items[0]["hash"] = "a".repeat(64);
    else items[0]["base_revision_hash"] = "a".repeat(64);
    await expect(readWorkflowArchive(writeJson(wire), localExperimentCodecs)).rejects.toThrow();
  }
  expect((await readWorkflowArchive(saved.json, localExperimentCodecs)).workflows).toHaveLength(1);
});

it("refuses malformed caller workflow metadata and identities without changing prior bytes", async () => {
  const { archive, source, definition } = await originals();
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  for (const fault of [
    "null",
    "array",
    "missing",
    "extra",
    "row-extra",
    "hash-type",
    "hash-spelling",
    "selected-type",
    "empty",
    "history-count",
  ]) {
    const wire = readJson(saved.json) as Record<string, unknown>;
    const manifest = wire["manifest"] as Record<string, unknown>;
    const extensions = manifest["extensions"] as Record<string, unknown>;
    const metadata = extensions["experiment_workflows"] as Record<string, unknown>;
    const items = metadata["items"] as Record<string, unknown>[];
    if (fault === "null") extensions["experiment_workflows"] = null;
    else if (fault === "array") extensions["experiment_workflows"] = [];
    else if (fault === "missing") delete metadata["selected"];
    else if (fault === "extra") metadata["unknown"] = true;
    else if (fault === "row-extra") (items[0] as Record<string, unknown>)["unknown"] = true;
    else if (fault === "hash-type") (items[0] as Record<string, unknown>)["hash"] = 1n;
    else if (fault === "hash-spelling")
      (items[0] as Record<string, unknown>)["hash"] = "A".repeat(64);
    else if (fault === "selected-type") metadata["selected"] = null;
    else if (fault === "empty") metadata["items"] = [];
    else metadata["items"] = Array.from({ length: maxArchivedWorkflows + 1 }, () => items[0]);
    await expect(readWorkflowArchive(writeJson(wire), localExperimentCodecs)).rejects.toThrow();
  }
  expect(saved.json).toBe(
    (await readWorkflowArchive(saved.json, localExperimentCodecs)).source.preview.json,
  );
});

it("keeps every retained graph when selecting an unchanged earlier graph before its first run", async () => {
  const { archive, source, definition } = await originals();
  const first = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  const second = await archiveWorkflow(
    first.json,
    source.revisionHash,
    { ...definition, workflow_id: "another-original-graph" },
    null,
    localExperimentCodecs,
  );
  const before = await readWorkflowArchive(second.json, localExperimentCodecs);
  const selected = await archiveWorkflow(
    second.json,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  const after = await readWorkflowArchive(selected.json, localExperimentCodecs);
  expect(after.workflows).toEqual(before.workflows);
  expect(after.selected).toBe(before.workflows[0]?.hash);
  expect(after.source.members).toEqual(source.archive.members);
});

it("retains source, runtime and evaluation history instead of accepting shortened or relabelled checkpoints", async () => {
  const { archive, source, definition, journal } = await originals();
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    journal,
    localExperimentCodecs,
  );
  for (const fault of ["source_fingerprint", "runtime_fingerprint", "shortened", "evaluation"]) {
    const wire = workflowJournalDocument(journal);
    const body = wire["body"] as Record<string, unknown>;
    if (fault === "source_fingerprint" || fault === "runtime_fingerprint")
      body[fault] = "a".repeat(64);
    else if (fault === "shortened") {
      body["entries"] = [];
      body["evaluations"] = 0n;
    } else {
      const row = (body["entries"] as Record<string, unknown>[])[0] as Record<string, unknown>;
      Object.assign(row, {
        status: "blocked",
        evaluated: false,
        output: null,
        output_digest: null,
        reason: "caller removed evaluation",
      });
      body["evaluations"] = 0n;
    }
    const changed = await parseWorkflowJournal(wire, definition);
    await expect(
      archiveWorkflow(saved.json, source.revisionHash, definition, changed, localExperimentCodecs),
    ).rejects.toThrow("original checkpoint source, runtime or attempt history changed");
  }
  expect(
    (await readWorkflowArchive(saved.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(journal);
});

it("settles a reserved original attempt once and refuses changed reservation identity", async () => {
  const { archive, source, definition, journal } = await originals();
  const wire = workflowJournalDocument(journal);
  const row = (
    (wire["body"] as Record<string, unknown>)["entries"] as Record<string, unknown>[]
  )[0] as Record<string, unknown>;
  Object.assign(row, {
    status: "running",
    output: null,
    output_digest: null,
    reason: "original reserved attempt",
  });
  const reserved = await parseWorkflowJournal(wire, definition);
  const pending = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    reserved,
    localExperimentCodecs,
  );
  const completed = await archiveWorkflow(
    pending.json,
    source.revisionHash,
    definition,
    journal,
    localExperimentCodecs,
  );
  expect(
    (await readWorkflowArchive(completed.json, localExperimentCodecs)).workflows[0]?.journal,
  ).toEqual(journal);
  for (const fault of ["fingerprint", "still-running"]) {
    const changed = workflowJournalDocument(fault === "fingerprint" ? journal : reserved);
    const candidate = (
      (changed["body"] as Record<string, unknown>)["entries"] as Record<string, unknown>[]
    )[0] as Record<string, unknown>;
    if (fault === "fingerprint") candidate["fingerprint"] = "b".repeat(64);
    else candidate["reason"] = "caller changed reservation";
    const next = await parseWorkflowJournal(changed, definition);
    await expect(
      archiveWorkflow(pending.json, source.revisionHash, definition, next, localExperimentCodecs),
    ).rejects.toThrow(
      fault === "fingerprint" ? "reserved attempt identity changed" : "cannot be rewritten",
    );
  }
});

it("refuses a graph rebound to a genuine new parameter revision", async () => {
  const { archive, source, definition } = await originals();
  const saved = await archiveWorkflow(
    archive.json,
    source.revisionHash,
    definition,
    null,
    localExperimentCodecs,
  );
  const parameters = await parameterSourceFromArchive(saved.json, localExperimentCodecs);
  if (parameters === null) throw new Error("Original parameter source is required");
  const initial = createParameterDraft(parameters);
  const edited = parameterDraftReducer(initial, {
    type: "value",
    key: "coupling",
    index: 0,
    text: "1.5",
    unit: initial.snapshot.units["coupling"] as string,
  });
  const child = await appendParameterRevision(
    saved.json,
    parameters,
    edited.snapshot,
    localExperimentCodecs,
    new Date(Date.now() + 1000).toISOString(),
  );
  await expect(
    archiveWorkflow(
      child.archive.json,
      child.revisionHash,
      definition,
      null,
      localExperimentCodecs,
    ),
  ).rejects.toThrow("rebound to another baseline revision");
  expect(
    (await readWorkflowArchive(child.archive.json, localExperimentCodecs)).workflows[0]
      ?.base_revision_hash,
  ).toBe(source.revisionHash);
  expect(
    (await readLocalExperiment(child.archive.json, localExperimentCodecs)).request.coupling,
  ).toBe(1.5);
});

it("refuses the next graph at the history bound while retaining every imported definition", async () => {
  const { archive, source, definition } = await originals();
  const wire = readJson(archive.json) as Record<string, unknown>;
  const manifest = wire["manifest"] as Record<string, unknown>;
  const entries = await Promise.all(
    Array.from({ length: maxArchivedWorkflows }, async (_, index) => {
      const document = workflowDocument({ ...definition, workflow_id: `retained-import-${index}` });
      return {
        hash: await canonicalDigest("studio.workflow-definition.v1", document),
        base_revision_hash: source.revisionHash,
        definition: document,
        journal: null,
      };
    }),
  );
  (manifest["extensions"] as Record<string, unknown>)["experiment_workflows"] = {
    version: 1n,
    selected: entries[0]?.hash,
    items: entries,
  };
  const prior = writeJson(wire);
  expect((await readWorkflowArchive(prior, localExperimentCodecs)).workflows).toHaveLength(
    maxArchivedWorkflows,
  );
  await expect(
    archiveWorkflow(prior, source.revisionHash, definition, null, localExperimentCodecs),
  ).rejects.toThrow("workflow history bound reached");
  expect(
    (await readWorkflowArchive(prior, localExperimentCodecs)).workflows.map((item) => item.hash),
  ).toEqual(entries.map((item) => item.hash));
});

it("refuses absent revisions and source artefacts presented as original run records", async () => {
  const { archive, source } = await originals();
  await expect(
    selectWorkflowRevision(archive.json, "a".repeat(64), localExperimentCodecs),
  ).rejects.toThrow("original workflow revision is absent");
  await expect(
    selectWorkflowRevision(
      archive.json,
      source.revisionHash,
      localExperimentCodecs,
      "a".repeat(64),
    ),
  ).rejects.toThrow("original indexed workflow run is absent");
  const wire = readJson(archive.json) as Record<string, unknown>;
  const manifest = wire["manifest"] as Record<string, unknown>;
  const body = manifest["body"] as Record<string, unknown>;
  body["artefact_refs"] = [
    {
      schema: source.environmentMember.schema,
      sha256: source.environmentMember.sha256,
      media_type: "application/json",
    },
  ];
  await expect(
    selectWorkflowRevision(
      writeJson(wire),
      source.revisionHash,
      localExperimentCodecs,
      source.environmentMember.sha256,
    ),
  ).rejects.toThrow("original indexed workflow run is absent");
  expect((await readLocalExperiment(archive.json, localExperimentCodecs)).revisionHash).toBe(
    source.revisionHash,
  );
});
