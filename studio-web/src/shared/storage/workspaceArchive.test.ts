// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — portable workspace archive

// @vitest-environment node
import { expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import {
  documentDigest,
  parseDocument,
  parseWorkspaceManifest,
  readJson,
  writeJson,
} from "../contracts";
import {
  admitWorkspaceArchive,
  createWorkspaceArchive,
  maxArchiveMembers,
  maxArchiveBytes,
  maxExpandedBytes,
  previewWorkspaceArchive,
} from "./workspaceArchive";

it("refuses an actual over-limit archive before parsing and retains the original admitted source", async () => {
  const original = await conformanceArchive(corpusText, false);
  const before = await admitWorkspaceArchive(original.json, conformanceCodecs);
  await expect(previewWorkspaceArchive(" ".repeat(maxArchiveBytes + 1))).rejects.toThrow(
    "encoded archive limit exceeded",
  );
  expect(before.preview.json).toBe(original.json);
});

it("refuses a changed identity from a real producer verifier instead of admitting its raw members", async () => {
  const original = await conformanceArchive(corpusText, false);
  const codecs = new Map(conformanceCodecs);
  const verify = codecs.get("review_fixture.v1");
  if (verify === undefined) throw new Error("Original conformance byte verifier required");
  codecs.set("review_fixture.v1", async (bytes) => ({
    ...(await verify(bytes)),
    digest: "a".repeat(64),
  }));
  await expect(admitWorkspaceArchive(original.json, codecs)).rejects.toThrow(
    "raw producer identity mismatch",
  );
  expect((await admitWorkspaceArchive(original.json, conformanceCodecs)).preview.json).toBe(
    original.json,
  );
});
import corpusText from "../../../../tests/data/studio_workspace/documents.json?raw";
import { conformanceArchive, conformanceCodecs } from "../../../browser-tests/workspaceFixture";

it("exposes the original admitted revision and specifications while retaining every member byte", async () => {
  const original = await conformanceArchive(corpusText, false);
  const admitted = await admitWorkspaceArchive(original.json, conformanceCodecs);
  const wire = readJson(original.json) as {
    members: unknown[];
    parameter_units: Record<string, string>;
  };
  expect(admitted.preview).toEqual(original);
  expect(admitted.members).toEqual(wire.members);
  expect(admitted.parameterUnits).toEqual(wire.parameter_units);
  for (const originalMember of admitted.members.filter((member) => member.kind === "document")) {
    const document = admitted.documents[originalMember.sha256];
    if (document === undefined) throw new Error("Original graph-admitted document required");
    expect(await documentDigest(document)).toBe(originalMember.sha256);
    expect(writeJson(document)).toBe(originalMember.content);
    expect(Object.isFrozen(document.body)).toBe(true);
  }
  expect(Object.isFrozen(admitted.documents)).toBe(true);
  const refs = admitted.manifest.body["revision_refs"] as readonly { sha256: string }[];
  for (const ref of refs) {
    expect(admitted.revisions[ref.sha256]?.schema).toBe("experiment_revision.v1");
    expect(
      await documentDigest(admitted.revisions[ref.sha256] as Parameters<typeof documentDigest>[0]),
    ).toBe(ref.sha256);
  }
  expect(admitted.parameterSpecs["theta"]?.schema).toBe("parameter_spec.v1");
  for (const value of [
    admitted,
    admitted.members,
    ...admitted.members,
    admitted.parameterUnits,
    admitted.revisions,
    ...Object.values(admitted.revisions),
    admitted.parameterSpecs,
    ...Object.values(admitted.parameterSpecs),
  ]) {
    expect(Object.isFrozen(value)).toBe(true);
  }
});

it("retains full graph refusal at the typed admission boundary", async () => {
  const original = await conformanceArchive(corpusText, false);
  const wire = readJson(original.json) as { members: unknown[] };
  wire.members.pop();
  await expect(admitWorkspaceArchive(writeJson(wire), conformanceCodecs)).rejects.toThrow();
  expect((await previewWorkspaceArchive(original.json, conformanceCodecs)).archiveDigest).toBe(
    original.archiveDigest,
  );
});

const root = {
  schema: "quantum_workspace.v1",
  extensions: { exact: 9007199254740993n, negative: -0 },
  body: {
    project_id: "00000000-0000-4000-8000-000000000001",
    revision_refs: [],
    draft_ref: null,
    created_at: "2026-09-30T00:00:00Z",
    updated_at: "2026-09-30T00:00:00Z",
    artefact_refs: [],
  },
};
function archive(patch: Record<string, unknown> = {}): string {
  return writeJson({
    schema: "quantum_workspace_archive.v1",
    manifest: root,
    members: [],
    parameter_units: {},
    ...patch,
  });
}

it("previews an exact empty project without inventing a revision or evidence", async () => {
  const preview = await previewWorkspaceArchive(archive());
  const parsed = parseWorkspaceManifest(root);
  if (!parsed.ok) throw new Error(parsed.message);
  expect(preview.workspaceHash).toBe(await documentDigest(parsed.value));
  expect(preview.projectId).toBe(root.body.project_id);
  expect(preview.documentHashes).toEqual([]);
  expect(preview.rawHashes).toEqual([]);
  expect(preview.json).toBe(archive());
  expect(Object.isFrozen(preview)).toBe(true);
  expect(Object.isFrozen(preview.memberNames)).toBe(true);
});

it("distinguishes exact formatting snapshots while preserving original document identity", async () => {
  const exact = archive();
  const formatted = `\n${exact}\n`;
  const first = await previewWorkspaceArchive(exact);
  const second = await previewWorkspaceArchive(formatted);
  expect(second.json).toBe(formatted);
  expect(second.archiveDigest).not.toBe(first.archiveDigest);
  expect(second.workspaceHash).toBe(first.workspaceHash);
  expect(second.documentHashes).toEqual(first.documentHashes);
  expect(second.rawHashes).toEqual(first.rawHashes);
});

it.each([
  "../outside",
  "/absolute",
  "a/../b",
  "a//b",
  "a\\b",
  "https://external",
  "a%2fb",
  "a\u0000b",
])("refuses unsafe member path %s before decoding or admission", async (name) => {
  await expect(
    previewWorkspaceArchive(
      archive({
        members: [
          { name, kind: "raw", schema: "original.v1", sha256: "0".repeat(64), content: "00" },
        ],
      }),
    ),
  ).rejects.toThrow("member name");
});

it("refuses unsupported major versions, decorated fields and duplicate JSON keys", async () => {
  await expect(
    previewWorkspaceArchive(archive({ schema: "quantum_workspace_archive.v2" })),
  ).rejects.toThrow("Unsupported archive schema");
  await expect(previewWorkspaceArchive(archive({ executable: "javascript" }))).rejects.toThrow(
    "archive fields",
  );
  await expect(previewWorkspaceArchive('{"schema":"one","schema":"two"}')).rejects.toThrow();
});

it("refuses unknown original producers instead of rehashing raw bytes as workspace data", async () => {
  const member = {
    name: "raw/original.bin",
    kind: "raw",
    schema: "original.v1",
    sha256: "0".repeat(64),
    content: "0001ff",
  };
  // A listed, unreferenced raw member is also required to have a trusted original verifier.
  await expect(previewWorkspaceArchive(archive({ members: [member] }))).rejects.toThrow(
    "unsupported raw producer",
  );
});

it("refuses corrupt document identity and repeated member names or identities", async () => {
  const member = {
    name: "documents/root.json",
    kind: "document",
    schema: root.schema,
    sha256: "0".repeat(64),
    content: writeJson(root),
  };
  await expect(previewWorkspaceArchive(archive({ members: [member] }))).rejects.toThrow(
    "digest mismatch",
  );
  await expect(previewWorkspaceArchive(archive({ members: [member, member] }))).rejects.toThrow(
    "duplicate archive member",
  );
  await expect(
    previewWorkspaceArchive(
      archive({ members: [member, { ...member, name: "documents/other.json" }] }),
    ),
  ).rejects.toThrow("duplicate archive identity");
});

it("refuses incomplete graphs and malformed raw bytes", async () => {
  await expect(
    previewWorkspaceArchive(
      archive({
        manifest: {
          ...root,
          body: {
            ...root.body,
            revision_refs: [
              {
                schema: "experiment_revision.v1",
                sha256: "0".repeat(64),
                media_type: "application/json",
              },
            ],
          },
        },
      }),
    ),
  ).rejects.toThrow("indexed document");
  const raw = {
    name: "raw/a.bin",
    kind: "raw",
    schema: "original.v1",
    sha256: "0".repeat(64),
    content: "0G",
  };
  await expect(previewWorkspaceArchive(archive({ members: [raw] }))).rejects.toThrow("hex bytes");
});

it("refuses member-count expansion before attempting any member decode", async () => {
  await expect(
    previewWorkspaceArchive(archive({ members: Array.from({ length: 1000 }, () => null) })),
  ).rejects.toThrow("member limit");
});

it("exports and previews lossless workspace scalars without changing the original document hash", async () => {
  const parsed = parseWorkspaceManifest(root);
  if (!parsed.ok) throw new Error(parsed.message);
  const exported = await createWorkspaceArchive(parsed.value, new Map(), new Map(), new Map());
  const reimport = await previewWorkspaceArchive(exported.json);
  expect(reimport.archiveDigest).toBe(exported.archiveDigest);
  expect(reimport.workspaceHash).toBe(await documentDigest(parsed.value));
  expect(exported.json).toContain("9007199254740993");
  expect(exported.json).toContain("-0.0");
});

it("refuses newline-suffixed hex rather than decoding a different byte sequence", async () => {
  const member = {
    name: "raw/input.bin",
    kind: "raw",
    schema: "original.v1",
    sha256: "0".repeat(64),
    content: "000\n",
  };
  await expect(previewWorkspaceArchive(archive({ members: [member] }))).rejects.toThrow(
    "hex bytes",
  );
});

it.each(["null", "[]", "42", '"archive"'])("refuses a non-object archive %s", async (json) => {
  await expect(previewWorkspaceArchive(json)).rejects.toThrow("archive: object required");
});

it.each([
  [{ name: "" }, "member name: nonempty string required"],
  [{ schema: null }, "member schema: nonempty string required"],
  [{ sha256: "A".repeat(64) }, "lowercase SHA-256 member digest required"],
  [{ sha256: "a".repeat(63) }, "lowercase SHA-256 member digest required"],
  [{ content: false }, "member content: string required"],
  [{ kind: "link" }, "unsupported archive member kind"],
] as const)("refuses malformed member metadata %j before codec calls", async (change, reason) => {
  const member = {
    name: "raw/input.bin",
    kind: "raw",
    schema: "original.v1",
    sha256: "0".repeat(64),
    content: "00",
    ...change,
  };
  await expect(previewWorkspaceArchive(archive({ members: [member] }))).rejects.toThrow(reason);
});

it("refuses a document declared under a different container schema", async () => {
  const parsed = parseWorkspaceManifest(root);
  if (!parsed.ok) throw new Error(parsed.message);
  const member = {
    name: "documents/root.json",
    kind: "document",
    schema: "experiment_revision.v1",
    sha256: await documentDigest(parsed.value),
    content: writeJson(root),
  };
  await expect(previewWorkspaceArchive(archive({ members: [member] }))).rejects.toThrow(
    "member document schema mismatch",
  );
});

it.each([null, [], 1])("refuses a non-object parameter-unit inventory %j", async (units) => {
  await expect(previewWorkspaceArchive(archive({ parameter_units: units }))).rejects.toThrow(
    "parameter units object required",
  );
});

it("refuses distinct original parameter documents that bind the same key", async () => {
  const corpus = readJson(
    await readFile(
      new URL("../../../../tests/data/studio_workspace/documents.json", import.meta.url),
      "utf8",
    ),
  ) as { fixtures: Record<string, unknown> };
  const original = corpus.fixtures["parameter"] as Record<string, unknown>;
  const changed = { ...original, extensions: { duplicate_key_test: true } };
  const members = [];
  for (const [index, wire] of [original, changed].entries()) {
    const parsed = parseDocument(wire);
    if (!parsed.ok) throw new Error(parsed.message);
    members.push({
      name: `documents/parameter-${index}.json`,
      kind: "document",
      schema: parsed.value.schema,
      sha256: await documentDigest(parsed.value),
      content: writeJson(wire),
    });
  }
  expect(members[0]?.sha256).not.toBe(members[1]?.sha256);
  await expect(previewWorkspaceArchive(archive({ members }))).rejects.toThrow(
    "duplicate parameter specification key",
  );
});

it("refuses an oversized export inventory before serialising its documents", async () => {
  const parsed = parseWorkspaceManifest(root);
  if (!parsed.ok) throw new Error(parsed.message);
  const documents = new Map(
    Array.from({ length: maxArchiveMembers }, (_, index) => [String(index), parsed.value] as const),
  );
  await expect(
    createWorkspaceArchive(parsed.value, documents, new Map(), new Map()),
  ).rejects.toThrow("archive member limit exceeded");
});

it.each([0, -1, 1.5, NaN, Infinity, maxExpandedBytes + 1])(
  "refuses invalid or widened expanded byte budget %s",
  async (limit) => {
    await expect(previewWorkspaceArchive(archive(), new Map(), limit)).rejects.toThrow(
      "expanded byte budget",
    );
  },
);

it("admits an exact reduced root budget and refuses an actual one-byte excess", async () => {
  const limit = new TextEncoder().encode(writeJson(root)).byteLength;
  const admitted = await previewWorkspaceArchive(archive(), new Map(), limit);
  expect(admitted.json).toBe(archive());
  await expect(previewWorkspaceArchive(archive(), new Map(), limit - 1)).rejects.toThrow(
    "expanded archive limit exceeded",
  );
});

it("accounts for actual document bytes before admission under a reduced caller budget", async () => {
  const corpus = readJson(
    await readFile(
      new URL("../../../../tests/data/studio_workspace/documents.json", import.meta.url),
      "utf8",
    ),
  ) as { fixtures: Record<string, unknown> };
  const parsed = parseDocument(corpus.fixtures["parameter"]);
  if (!parsed.ok) throw new Error(parsed.message);
  const content = writeJson(parsed.value);
  const member = {
    name: "documents/parameter.json",
    kind: "document",
    schema: parsed.value.schema,
    sha256: await documentDigest(parsed.value),
    content,
  };
  const text = archive({
    members: [member],
    parameter_units: { [parsed.value.body["key"] as string]: parsed.value.body["unit"] },
  });
  const limit =
    new TextEncoder().encode(writeJson(root)).byteLength +
    new TextEncoder().encode(content).byteLength;
  const admitted = await previewWorkspaceArchive(text, new Map(), limit);
  expect(admitted.documentHashes).toEqual([member.sha256]);
  await expect(previewWorkspaceArchive(text, new Map(), limit - 1)).rejects.toThrow(
    "expanded archive limit exceeded",
  );
});
