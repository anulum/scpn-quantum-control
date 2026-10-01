// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned settings behaviour tests

// @vitest-environment node
import { expect, it } from "vitest";
import fixtureText from "../../../../tests/data/studio_workspace/settings.json?raw";
import { canonicalDigest } from "./canonical";
import { readJson, writeJson } from "./jsonTransport";
import { documentDigest, parseResolvedSettings, parseWorkspaceManifest } from "./workspace";
import { settingsFromArchive } from "./settings";
import { createWorkspaceArchive } from "../storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../storage/workspaceArchive";

/** Literal values produced by the real Python resolver, with explicit test-only raw owners. */
interface Fixture {
  readonly document: unknown;
  readonly manifest: unknown;
  readonly raw: readonly { readonly digest: string; readonly record: {
    readonly schema: string; readonly body: { readonly role: string; readonly synthetic: boolean };
  } }[];
}
const fixture = readJson(fixtureText) as Fixture;

/** Admit a complete archive through the original contract and raw-codec boundary. */
async function admitted(): Promise<WorkspaceArchivePreview> {
  const manifest = parseWorkspaceManifest(fixture.manifest);
  const document = parseResolvedSettings(fixture.document);
  if (!manifest.ok || !document.ok) throw new Error("Shared fixture refused");
  const raw = new Map(fixture.raw.map(({ digest, record }) => [
    digest, { schema: record.schema, content: new TextEncoder().encode(writeJson(record)) },
  ]));
  return createWorkspaceArchive(manifest.value,
    new Map([[await documentDigest(document.value), document.value]]), raw, new Map(),
    new Map([["review_fixture.v1", async (content: Uint8Array) => {
      const record = readJson(new TextDecoder().decode(content)) as Fixture["raw"][number]["record"];
      if (record.schema !== "review_fixture.v1" || record.body.synthetic !== true) throw new Error("Synthetic owner required");
      return { schema: record.schema, kind: record.body.role, digest: await canonicalDigest(record.schema, record) };
    }]]));
}

it("projects real admitted source values, exact identity and immutable provenance", async () => {
  const preview = await admitted();
  const result = settingsFromArchive(preview);
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.message);
  const record = result.value[0]!;
  expect(result.value).toHaveLength(1);
  expect(record.document.body["effective"]).toEqual({
    shots: 7n, precision: "float64", theme: "dark", seed: 9007199254740993n,
  });
  expect(record.document.body["origins"]).toEqual({
    shots: "run", precision: "project", theme: "project", seed: "defaults",
  });
  expect(preview.documentHashes).toContain(record.digest);
  expect(record.digest).toBe(await documentDigest(record.document));
  expect(Object.isFrozen(result.value)).toBe(true);
  expect(Object.isFrozen(record)).toBe(true);
  expect(Object.isFrozen(record.document.body["effective"])).toBe(true);
  expect(preview.json).toContain("9007199254740993");
});

it("returns an empty projection when an admitted project has no settings", async () => {
  const parsed = parseWorkspaceManifest(fixture.manifest);
  if (!parsed.ok) throw new Error(parsed.message);
  const preview = await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map());
  expect(settingsFromArchive(preview)).toEqual({ ok: true, value: [] });
});

it.each(["{", "null", "42", "{}", '{"members":null}', '{"members":{}}'])(
  "refuses damaged archive shape %s with an authored stable category", async json => {
    const preview = await admitted();
    expect(settingsFromArchive({ ...preview, json })).toEqual({
      ok: false, code: "settings_inspection_refused", path: "$.members",
      message: "Settings records could not be inspected.",
    });
    expect(settingsFromArchive(preview).ok).toBe(true);
  },
);

it.each([
  { kind: "raw" }, { kind: undefined }, { content: 3 }, { content: undefined },
  { sha256: "A".repeat(64) }, { sha256: undefined }, { sha256: 42 },
  { content: "{}" }, { content: "{" },
])("refuses damaged settings member %j without retaining partial output", async patch => {
  const preview = await admitted();
  const archive = readJson(preview.json) as { members: Record<string, unknown>[] };
  const member = archive.members.find(value => value["schema"] === "resolved_settings.v1")!;
  Object.assign(member, patch);
  for (const [key, value] of Object.entries(member)) if (value === undefined) delete member[key];
  const result = settingsFromArchive({ ...preview, json: writeJson(archive) });
  expect(result).toMatchObject({ ok: false, code: "settings_inspection_refused" });
  expect(settingsFromArchive(preview).ok).toBe(true);
});

it("skips unrelated member kinds without altering source-owned records", async () => {
  const preview = await admitted();
  const archive = readJson(preview.json) as { members: unknown[] };
  archive.members.unshift(null, 1, {}, { schema: "other.v1" });
  const result = settingsFromArchive({ ...preview, json: writeJson(archive) });
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.message);
  expect(result.value).toHaveLength(1);
});
