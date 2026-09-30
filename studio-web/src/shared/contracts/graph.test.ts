// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace graph admission tests

// @vitest-environment node
import { describe, expect, it } from "vitest";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { canonicalDigest } from "./canonical";
import { admitWorkspace } from "./graph";
import type { RawArtifact, RawCodec, RawIdentity } from "./graph";
import { readJson, writeJson } from "./jsonTransport";
import { documentDigest, documentToWire, parseDocument, parseParameterSpec, parseWorkspaceManifest } from "./workspace";
import type { ParameterSpec, ParseResult, WorkspaceDocument, WorkspaceManifest } from "./workspace";

const fixtures = (readJson(corpusText) as { fixtures: Record<string, unknown> }).fixtures;
function take<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(result.path + ": " + result.message);
  return result.value;
}
async function fixtureCodec(content: Uint8Array): Promise<RawIdentity> {
  const payload = readJson(new TextDecoder().decode(content)) as { schema: string; body: { role: string; synthetic: boolean } };
  if (payload.schema !== "review_fixture.v1" || payload.body.synthetic !== true) throw new Error("synthetic conformance producer required");
  return { schema: payload.schema, kind: payload.body.role, digest: await canonicalDigest(payload.schema, payload) };
}
type Bundle = [WorkspaceManifest, Map<string, WorkspaceDocument>, Map<string, RawArtifact>, Map<string, RawCodec>, Map<string, ParameterSpec>, Map<string, string>];
async function bundle(): Promise<Bundle> {
  const documents = new Map<string, WorkspaceDocument>();
  for (const name of ["settings", "parameter", "revision_root", "revision_child", "run"]) {
    const document = take(parseDocument(fixtures[name]));
    documents.set(await documentDigest(document), document);
  }
  const raw = new Map<string, RawArtifact>();
  for (const name of ["problem", "program", "policy", "environment", "plan"]) {
    const content = new TextEncoder().encode(writeJson(fixtures[name]));
    const identity = await fixtureCodec(content);
    raw.set(identity.digest, { schema: identity.schema, content });
  }
  return [take(parseWorkspaceManifest(fixtures["workspace"])), documents, raw,
    new Map([["review_fixture.v1", fixtureCodec]]),
    new Map([["theta", take(parseParameterSpec(fixtures["parameter"]))]]), new Map([["theta", "rad"]])];
}

describe("complete workspace graph admission", () => {
  it("admits public document objects with explicit producer identities", async () => {
    const inputs = await bundle();
    const result = take(await admitWorkspace(...inputs));
    expect(result.projectId).toBe(inputs[0].body["project_id"]);
    expect(result.workspaceHash).toBe(await documentDigest(inputs[0]));
    expect(result.documentHashes).toEqual([...inputs[1].keys()].sort());
    expect(result.rawHashes).toEqual([...inputs[2].keys()].sort());
  });
  it("requires a registered producer and never invents raw support", async () => {
    const inputs = await bundle();
    inputs[3].clear();
    expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("unsupported raw producer") });
  });
  it("rejects dishonest index keys and altered raw content", async () => {
    const inputs = await bundle();
    const keys = [...inputs[1].keys()];
    inputs[1].set(keys[0]!, inputs[1].get(keys[1]!)!);
    expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("digest mismatch") });
    const altered = await bundle();
    const [key, raw] = [...altered[2]][0]!;
    const payload = readJson(new TextDecoder().decode(raw.content)) as { body: Record<string, unknown> };
    payload.body["purpose"] = "changed";
    altered[2].set(key, { ...raw, content: new TextEncoder().encode(writeJson(payload)) });
    expect(await admitWorkspace(...altered)).toMatchObject({ ok: false, message: expect.stringContaining("raw producer identity mismatch") });
  });
  it("captures maps and bytes before awaiting the first digest", async () => {
    const inputs = await bundle();
    const pending = admitWorkspace(...inputs);
    inputs[5].set("theta", "Hz");
    for (const raw of inputs[2].values()) raw.content.fill(0);
    inputs[1].clear();
    inputs[3].clear();
    expect((await pending).ok).toBe(true);
  });
  it("refuses missing units, sources and renamed specs", async () => {
    const units = await bundle();
    units[5].clear();
    expect(await admitWorkspace(...units)).toMatchObject({ ok: false, message: expect.stringContaining("keys must match") });
    const raw = await bundle();
    raw[2].clear();
    expect(await admitWorkspace(...raw)).toMatchObject({ ok: false, message: expect.stringContaining("dangling raw") });
    const specs = await bundle();
    specs[4].set("renamed", specs[4].get("theta")!);
    specs[4].delete("theta");
    specs[5].set("renamed", "rad");
    specs[5].delete("theta");
    expect(await admitWorkspace(...specs)).toMatchObject({ ok: false, message: expect.stringContaining("key or indexed identity") });
  });
  it("refuses a real parameter dependency cycle before a producer callback", async () => {
    const inputs = await bundle();
    const payload = structuredClone(fixtures["parameter"]) as { body: Record<string, unknown> };
    payload.body["dependency_keys"] = ["theta"];
    const spec = take(parseParameterSpec(payload));
    inputs[4].set("theta", spec);
    inputs[1].set(await documentDigest(spec), spec);
    inputs[3].clear();
    expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("dependency cycle") });
  });
});

async function revisionBundle(payload: unknown): Promise<Bundle> {
  const inputs = await bundle();
  const revision = take(parseDocument(payload));
  for (const [digest, record] of inputs[1]) {
    if (["experiment_revision.v1", "local_run_record.v1"].includes(record.schema)) inputs[1].delete(digest);
  }
  const digest = await documentDigest(revision);
  inputs[1].set(digest, revision);
  const wire = documentToWire(inputs[0]);
  (wire["body"] as Record<string, unknown>)["revision_refs"] = [{ schema: revision.schema, sha256: digest, media_type: "application/json" }];
  inputs[0] = take(parseWorkspaceManifest(wire));
  return inputs;
}
it.each([
  ["wrong_reference_schema", "schema or kind mismatch"],
  ["cross_project_parent", "cross-project revision"],
  ["revision_child", "dangling dependency"],
])("rejects the shared graph case %s", async (name, reason) => {
  expect(await admitWorkspace(...await revisionBundle(fixtures[name!]))).toMatchObject({ ok: false, message: expect.stringContaining(reason!) });
});
it("requires the revision's immutable spec reference", async () => {
  const payload = structuredClone(fixtures["revision_root"]) as { body: Record<string, unknown> };
  payload.body["input_refs"] = [];
  expect(await admitWorkspace(...await revisionBundle(payload))).toMatchObject({ ok: false, message: expect.stringContaining("immutable specification reference") });
});
it("keeps raw roles and run revision identities distinct", async () => {
  const payload = structuredClone(fixtures["revision_root"]) as { body: Record<string, unknown> };
  payload.body["problem_ref"] = payload.body["program_ref"];
  expect(await admitWorkspace(...await revisionBundle(payload))).toMatchObject({ ok: false, message: expect.stringContaining("schema or kind mismatch") });
  const inputs = await bundle();
  const runWire = structuredClone(fixtures["run"]) as { body: Record<string, unknown> };
  runWire.body["revision_hash"] = await documentDigest(inputs[4].get("theta")!);
  for (const [digest, record] of inputs[1]) if (record.schema === "local_run_record.v1") inputs[1].delete(digest);
  const run = take(parseDocument(runWire));
  inputs[1].set(await documentDigest(run), run);
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("missing revision") });
});
it("requires a plan producer for a local run", async () => {
  const inputs = await bundle();
  const runWire = structuredClone(fixtures["run"]) as { body: Record<string, unknown> };
  const problem = await canonicalDigest("review_fixture.v1", fixtures["problem"]);
  runWire.body["plan_hash"] = problem;
  for (const [digest, record] of inputs[1]) if (record.schema === "local_run_record.v1") inputs[1].delete(digest);
  const run = take(parseDocument(runWire));
  inputs[1].set(await documentDigest(run), run);
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("wrong raw kind") });
});
it("admits empty projects and the matching root index without a producer", async () => {
  const wire = documentToWire(take(parseWorkspaceManifest(fixtures["workspace"])));
  Object.assign(wire["body"] as object, { revision_refs: [], artefact_refs: [], draft_ref: null });
  const root = take(parseWorkspaceManifest(wire));
  const receipt = take(await admitWorkspace(root, new Map([[await documentDigest(root), root]]), new Map(), new Map(), new Map(), new Map()));
  expect(receipt.rawHashes).toEqual([]);
  expect(receipt.documentHashes).toEqual([await documentDigest(root)]);
});
it("refuses missing parameter dependencies and ambiguous raw/document identities", async () => {
  const inputs = await bundle();
  const payload = documentToWire(inputs[4].get("theta")!);
  (payload["body"] as Record<string, unknown>)["dependency_keys"] = ["missing"];
  const spec = take(parseParameterSpec(payload));
  inputs[4].set("theta", spec);
  inputs[1].set(await documentDigest(spec), spec);
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("dangling dependency") });
  const collision = await bundle();
  collision[2].set([...collision[1].keys()][0]!, [...collision[2].values()][0]!);
  expect(await admitWorkspace(...collision)).toMatchObject({ ok: false, message: expect.stringContaining("ambiguous raw/document") });
});

it("does not admit workspace documents through a raw verifier", async () => {
  const inputs = await bundle();
  for (const [digest, document] of inputs[1]) {
    if (document.schema === "experiment_revision.v1") {
      inputs[2].set(digest, { schema: document.schema, content: new TextEncoder().encode(writeJson(documentToWire(document))) });
      inputs[1].delete(digest);
    } else if (document.schema === "local_run_record.v1") inputs[1].delete(digest);
  }
  inputs[3].set("experiment_revision.v1", async content => {
    const document = take(parseDocument(readJson(new TextDecoder().decode(content))));
    return { schema: document.schema, kind: document.schema, digest: await documentDigest(document) };
  });
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("workspace reference requires indexed document") });
});


it("revalidates malformed caller-owned document objects before admission", async () => {
  const inputs = await bundle();
  const invalid = documentToWire(inputs[0]);
  delete invalid["extensions"];
  inputs[0] = invalid as unknown as WorkspaceManifest;
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("missing or unknown field") });
});

it("admits a selected two-parent merge without rewriting either parent", async () => {
  const inputs = await bundle();
  const parent = [...inputs[1]].find(([, document]) => document.schema === "experiment_revision.v1" && (document.body["parent_revision_hashes"] as readonly string[]).length === 0)!;
  const siblingWire = documentToWire(parent[1]);
  (siblingWire["extensions"] as Record<string, unknown>)["branch"] = "independent";
  const sibling = take(parseDocument(siblingWire));
  const siblingHash = await documentDigest(sibling);
  const mergeWire = documentToWire(take(parseDocument(fixtures["revision_child"])));
  (mergeWire["body"] as Record<string, unknown>)["parent_revision_hashes"] = [parent[0], siblingHash];
  const merged = take(parseDocument(mergeWire));
  const mergedHash = await documentDigest(merged);
  for (const [digest, document] of inputs[1]) if ((document.schema === "experiment_revision.v1" || document.schema === "local_run_record.v1") && digest !== parent[0]) inputs[1].delete(digest);
  inputs[1].set(siblingHash, sibling); inputs[1].set(mergedHash, merged);
  const references = [parent[0], siblingHash, mergedHash].map(sha256 => ({ schema: "experiment_revision.v1", sha256, media_type: "application/json" }));
  const manifestWire = documentToWire(inputs[0]);
  Object.assign(manifestWire["body"] as object, { revision_refs: references, draft_ref: references[2] });
  inputs[0] = take(parseWorkspaceManifest(manifestWire));
  const admission = take(await admitWorkspace(...inputs));
  expect(admission.documentHashes).toEqual([...inputs[1].keys()].sort());
  expect(await documentDigest(parent[1])).toBe(parent[0]);
  expect(await documentDigest(sibling)).toBe(siblingHash);
});

it("refuses an additional independently valid workspace root", async () => {
  const inputs = await bundle();
  const wire = documentToWire(inputs[0]);
  (wire["extensions"] as Record<string, unknown>)["title"] = "Another root";
  const root = take(parseWorkspaceManifest(wire));
  inputs[1].set(await documentDigest(root), root);
  expect(await admitWorkspace(...inputs)).toMatchObject({ ok: false, message: expect.stringContaining("unexpected workspace root") });
});

it("validates output references against original raw producer bytes", async () => {
  const inputs = await bundle();
  const [rawHash, raw] = [...inputs[2]][0]!;
  const run = [...inputs[1]].find(([, document]) => document.schema === "local_run_record.v1")!;
  const wire = documentToWire(run[1]);
  (wire["body"] as Record<string, unknown>)["output_refs"] = [{ schema: raw.schema, sha256: rawHash, media_type: "application/json" }];
  const changed = take(parseDocument(wire));
  inputs[1].delete(run[0]); inputs[1].set(await documentDigest(changed), changed);
  expect(take(await admitWorkspace(...inputs)).rawHashes).toContain(rawHash);
});

it("refuses caller-owned raw records whose inspection throws a non-error value", async () => {
  const inputs = await bundle();
  const [digest, raw] = [...inputs[2]][0]!;
  const hostile = { get schema(): string { throw "untrusted raw record inspection"; }, content: raw.content };
  inputs[2].set(digest, hostile);
  expect(await admitWorkspace(...inputs)).toEqual({ ok: false, code: "invalid_graph", path: "$", message: "Workspace admission refused" });
  expect(inputs[2].get(digest)).toBe(hostile);
  expect(hostile.content).toBe(raw.content);
});
