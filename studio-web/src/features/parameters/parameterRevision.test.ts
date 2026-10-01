// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — immutable parameter revision archive tests

// @vitest-environment node
import { expect, it } from "vitest";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";
import { documentDigest, parseExperimentRevision, parseWorkspaceManifest, readJson, writeJson } from "../../shared/contracts";
import { createWorkspaceArchive, previewWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import { createParameterDraft, parameterDraftReducer } from "./parameterDraft";
import { appendParameterRevision, createParameterRevision, parameterSourceFromArchive } from "./parameterRevision";

it("appends a real v1 child while retaining every old raw/document member and run reference", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("The original admitted revision was not selected");
  const sourceBytes = writeJson(source.revision);
  const oldHash = await documentDigest(source.revision);
  const initial = createParameterDraft(source);
  const changed = parameterDraftReducer(initial, { type: "value", key: "theta", index: 0, text: "-3", unit: "rad" });
  const child = await appendParameterRevision(prior.json, source, changed.snapshot, conformanceCodecs, "2026-10-01T12:00:00Z");
  const before = readJson(prior.json) as { members: unknown[] };
  const after = readJson(child.archive.json) as { members: unknown[]; manifest: { body: { draft_ref: { sha256: string } } } };
  expect(after.members.slice(0, before.members.length)).toEqual(before.members);
  expect(after.members).toHaveLength(before.members.length + 1);
  expect(after.manifest.body.draft_ref.sha256).toBe(child.revisionHash);
  expect(child.archive.rawHashes).toEqual(prior.rawHashes);
  expect(prior.documentHashes.every(hash => child.archive.documentHashes.includes(hash))).toBe(true);
  expect(writeJson(source.revision)).toBe(sourceBytes);
  const restored = await parameterSourceFromArchive(child.archive.json, conformanceCodecs);
  if (!restored) throw new Error("New child missing after complete archive admission");
  expect(restored.revision.body["parent_revision_hashes"]).toEqual([oldHash]);
  expect(restored.revision.body["problem_ref"]).toEqual(source.revision.body["problem_ref"]);
  expect(restored.revision.body["program_ref"]).toEqual(source.revision.body["program_ref"]);
  expect(restored.revision.body["semantic_settings_ref"]).toEqual(source.revision.body["semantic_settings_ref"]);
  expect(createParameterDraft(restored).snapshot.parameters["theta"]!.values[0]).toBe("c008000000000000");
  expect(await previewWorkspaceArchive(child.archive.json, conformanceCodecs)).toEqual(child.archive);
});

it("persists the exact trainable subset across child creation and source reload", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("Missing original revision");
  const state = parameterDraftReducer(createParameterDraft(source), { type: "mask", key: "theta", index: 1, enabled: false });
  const revision = await createParameterRevision(source, state.snapshot);
  expect(createParameterDraft({ ...source, revision }).snapshot.trainableMasks["theta"]).toEqual([true, false]);
});

it("refuses a changed source unit before adding any revision or rewriting original results", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const before = prior.json;
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("Missing original revision");
  const state = createParameterDraft(source);
  await expect(appendParameterRevision(prior.json, source, { ...state.snapshot, units: { theta: "s" } }, conformanceCodecs, "2026-10-01T12:00:00Z")).rejects.toThrow("unit mismatch");
  expect(prior.json).toBe(before);
  expect((await previewWorkspaceArchive(prior.json, conformanceCodecs)).archiveDigest).toBe(prior.archiveDigest);
});

it("refuses an unlisted parent and future archive major without creating a child", async () => {
  const prior = await conformanceArchive(corpusText, false);
  await expect(parameterSourceFromArchive(prior.json, conformanceCodecs, "0".repeat(64))).rejects.toThrow("listed");
  const future = (readJson(prior.json) as Record<string, unknown>);
  future["schema"] = "quantum_workspace_archive.v2";
  await expect(parameterSourceFromArchive(writeJson(future), conformanceCodecs)).rejects.toThrow("Unsupported archive schema");
});

it("selects the last listed revision when the admitted project has no draft pointer", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const original = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!original) throw new Error("Missing original revision");
  const wire = readJson(prior.json) as { manifest: { body: Record<string, unknown> } };
  wire.manifest.body["draft_ref"] = null;
  const restored = await parameterSourceFromArchive(writeJson(wire), conformanceCodecs);
  expect(restored?.revision).toEqual(original.revision);
});

it("selects a graph-admitted draft that is absent from the saved revision list", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const wire = readJson(prior.json) as { manifest: { body: Record<string, unknown> } };
  const refs = wire.manifest.body["revision_refs"] as readonly unknown[];
  wire.manifest.body["draft_ref"] = refs.at(-1);
  wire.manifest.body["revision_refs"] = [];
  const restored = await parameterSourceFromArchive(writeJson(wire), conformanceCodecs);
  expect(restored?.revision.schema).toBe("experiment_revision.v1");
  await expect(parameterSourceFromArchive(writeJson(wire), conformanceCodecs, "0".repeat(64))).rejects.toThrow("listed");
});

it("keeps unrelated admitted declarations out of a revision with no editable parameters", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("Missing original revision");
  const parsed = parseExperimentRevision({ ...source.revision, body: { ...source.revision.body, parameters: {} } });
  if (!parsed.ok) throw new Error(parsed.message);
  const revision = parsed.value;
  const hash = await documentDigest(revision);
  const wire = readJson(prior.json) as { manifest: { body: Record<string, unknown> }; members: unknown[] };
  wire.members.push({ name: `documents/${hash}.json`, kind: "document", schema: revision.schema, sha256: hash, content: writeJson(revision) });
  wire.manifest.body["draft_ref"] = { schema: revision.schema, sha256: hash, media_type: "application/json" };
  const restored = await parameterSourceFromArchive(writeJson(wire), conformanceCodecs);
  expect(restored?.revision).toEqual(revision);
  expect(restored?.specs).toEqual([]);
  expect(restored?.units).toEqual({});
});

it("captures immutable source and candidate values before the asynchronous digest boundary", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("Missing original revision");
  const input = structuredClone(source);
  const candidate = structuredClone(createParameterDraft(source).snapshot);
  const child = createParameterRevision(input, candidate);
  Reflect.set(input.units, "theta", "s");
  Reflect.set(input.revision.body, "parameters", {});
  Reflect.set(candidate.parameters["theta"]!.values, "0", "4008000000000000");
  const revision = await child;
  expect(revision.body["parent_revision_hashes"]).toEqual([await documentDigest(source.revision)]);
  expect(createParameterDraft({ ...source, revision }).snapshot.parameters["theta"]!.values[0]).toBe("8000000000000000");
  expect(Object.isFrozen(revision.body)).toBe(true);
  expect(Object.isFrozen(revision.extensions["parameter_editor"])).toBe(true);
});

it("keeps an empty project explicitly without manufacturing an experiment revision", async () => {
  const parsed = parseWorkspaceManifest({ schema: "quantum_workspace.v1", body: {
    project_id: "00000000-0000-4000-8000-000000000001", revision_refs: [], draft_ref: null,
    created_at: "2026-09-29T00:00:00Z", updated_at: "2026-09-29T00:00:00Z", artefact_refs: [],
  }, extensions: {} });
  if (!parsed.ok) throw new Error(parsed.message);
  const empty = await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map());
  expect(await parameterSourceFromArchive(empty.json, new Map())).toBeNull();
});

it("refuses a timestamp before creation and duplicate immutable child without changing the prior archive", async () => {
  const prior = await conformanceArchive(corpusText, false);
  const source = await parameterSourceFromArchive(prior.json, conformanceCodecs);
  if (!source) throw new Error("Missing original revision");
  const state = createParameterDraft(source);
  await expect(appendParameterRevision(prior.json, source, state.snapshot, conformanceCodecs, "2020-01-01T00:00:00Z")).rejects.toThrow("precedes creation");
  const first = await appendParameterRevision(prior.json, source, state.snapshot, conformanceCodecs, "2026-10-01T12:00:00Z");
  await expect(appendParameterRevision(first.archive.json, source, state.snapshot, conformanceCodecs, "2026-10-01T12:01:00Z")).rejects.toThrow("already exists");
});
