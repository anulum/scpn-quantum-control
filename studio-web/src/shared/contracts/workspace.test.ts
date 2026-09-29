// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — workspace document tests

// @vitest-environment node
import { describe, expect, it } from "vitest";
import structuralText from "../../../../tests/data/studio_workspace/structural.json?raw";
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { readJson, writeJson } from "./jsonTransport";
import { documentDigest, documentToWire, parseDocument, parseDocumentJson,
  parseExperimentRevision, parseLocalRunRecord, parseParameterSpec,
  parseResolvedSettings, parseWorkspaceManifest, validateParameterBinding } from "./index";
import type { ParseResult, WorkspaceDocument } from "./workspace";

const corpus = readJson(corpusText) as {
  fixtures: Record<string, { schema: string; body: Record<string, unknown>; extensions: Record<string, unknown> }>;
  cases: { id: string; fixture?: string; expectation: string }[];
};
function fixture(name: string) {
  const payload = corpus.fixtures[name];
  if (!payload) throw new Error("fixture missing: " + name);
  return structuredClone(payload);
}
function accepted<T>(result: ParseResult<T>): T {
  if (!result.ok) throw new Error(result.path + ": " + result.message);
  return result.value;
}

describe("workspace document public contracts", () => {
  it.each(corpus.cases.filter(row => row.fixture !== undefined))("honours the $id structural boundary", async row => {
    const payload = fixture(row.fixture!);
    const before = writeJson(payload);
    const parsed = parseDocument(payload);
    if (row.expectation === "reject") expect(parsed.ok).toBe(false);
    else {
      const document = accepted(parsed);
      expect(documentToWire(document)).toEqual(payload);
      const restored = accepted(parseDocumentJson(writeJson(documentToWire(document))));
      expect(await documentDigest(restored)).toBe(await documentDigest(document));
    }
    expect(writeJson(payload)).toBe(before);
  });
  it("exposes every named parser and refuses a different document kind", () => {
    const pairs: [string, (raw: unknown) => ParseResult<WorkspaceDocument>][] = [
      ["workspace", parseWorkspaceManifest], ["revision_root", parseExperimentRevision],
      ["parameter", parseParameterSpec], ["settings", parseResolvedSettings], ["run", parseLocalRunRecord],
    ];
    for (const [name, parse] of pairs) {
      expect(parse(fixture(name)).ok).toBe(true);
      expect(parse({ schema: "unknown.v1", body: {}, extensions: {} }).ok).toBe(false);
      expect(parse(fixture(name === "workspace" ? "run" : "workspace")).ok).toBe(false);
    }
  });
  it("takes a deep snapshot and exports independent mutable wire data", async () => {
    const payload = fixture("revision_root");
    const document = accepted(parseExperimentRevision(payload));
    const before = await documentDigest(document);
    payload.extensions["later"] = [1n];
    const wire = documentToWire(document);
    (wire["extensions"] as Record<string, unknown>)["later"] = [2n];
    expect(await documentDigest(document)).toBe(before);
    expect(Object.isFrozen(document.body)).toBe(true);
    expect(Object.isFrozen(document.extensions)).toBe(true);
    expect(document.extensions["later"]).toBeUndefined();
    expect(() => { (document.body as Record<string, unknown>)["project_id"] = "changed"; }).toThrow();
  });
  it("binds explicit units, exact scalar values and bounded domains", () => {
    const payload = fixture("parameter");
    payload.body["domain"] = { kind: "closed_interval", lower: "0000000000000000", upper: "3ff0000000000000" };
    const spec = accepted(parseParameterSpec(payload));
    const values = { dtype: "float64", shape: [2n], values: ["8000000000000000", "3ff0000000000000"] };
    expect(validateParameterBinding(spec, values, "rad").ok).toBe(true);
    expect(validateParameterBinding(spec, values, "Hz").ok).toBe(false);
    expect(validateParameterBinding(spec, { ...values, shape: [1n] }, "rad").ok).toBe(false);
    expect(validateParameterBinding(spec, { ...values, values: ["4000000000000000", "3ff0000000000000"] }, "rad").ok).toBe(false);
    payload.body["dtype"] = "int64";
    payload.body["shape"] = [1n];
    payload.body["unit"] = "1";
    payload.body["domain"] = { kind: "enumerated", values: ["9007199254740993"] };
    const integer = accepted(parseParameterSpec(payload));
    expect(validateParameterBinding(integer, { dtype: "int64", shape: [1n], values: ["9007199254740993"] }, "1").ok).toBe(true);
    expect(validateParameterBinding(integer, { dtype: "int64", shape: [1n], values: ["1"] }, "1").ok).toBe(false);
  });
  it("refuses malformed text and unknown envelopes without a success-shaped value", () => {
    expect(parseDocumentJson('{"schema":')).toMatchObject({ ok: false, code: "invalid_document" });
    for (const value of [null, [], {}, { schema: 1 }, { schema: "quantum_workspace.v2", body: {}, extensions: {} }]) {
      expect(parseDocument(value).ok).toBe(false);
    }
  });
});

const structural = readJson(structuralText) as { cases: { id: string; fixture: string; path: string[]; value: unknown; accept: boolean }[] };
it.each(structural.cases)("shares structural admission for $id", async row => {
  const payload = fixture(row.fixture);
  let target = payload as Record<string, unknown>;
  for (const key of row.path.slice(0, -1)) target = target[key] as Record<string, unknown>;
  target[row.path.at(-1)!] = row.value;
  const result = parseDocument(payload);
  expect(result.ok).toBe(row.accept);
  if (result.ok) {
    const restored = accepted(parseDocumentJson(writeJson(documentToWire(result.value))));
    expect(await documentDigest(restored)).toBe(await documentDigest(result.value));
    expect(documentToWire(result.value)["extensions"]).toEqual(payload.extensions);
  }
});
it("requires every field in each supported schema", () => {
  for (const name of ["workspace", "revision_root", "parameter", "settings", "run"]) {
    for (const field of Object.keys(fixture(name).body)) {
      const payload = fixture(name);
      delete payload.body[field];
      expect(parseDocument(payload)).toMatchObject({ ok: false, code: "invalid_document" });
    }
  }
});

it("preserves parent identity when an exported child is edited", async () => {
  const parent = accepted(parseExperimentRevision(fixture("revision_root")));
  const child = accepted(parseExperimentRevision(fixture("revision_child")));
  const before = [writeJson(documentToWire(parent)), await documentDigest(child)];
  const wire = documentToWire(child);
  (wire["extensions"] as Record<string, unknown>)["measurement_order"] = ["q1", "q0"];
  const edited = accepted(parseExperimentRevision(wire));
  expect(await documentDigest(edited)).not.toBe(await documentDigest(child));
  expect(edited.body["parent_revision_hashes"]).toEqual([await documentDigest(parent)]);
  expect([writeJson(documentToWire(parent)), await documentDigest(child)]).toEqual(before);
});
