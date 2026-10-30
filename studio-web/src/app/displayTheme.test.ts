// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — display theme read from real admitted settings

// @vitest-environment node
import { describe, expect, it } from "vitest";
import fixtureText from "../../../tests/data/studio_workspace/settings.json?raw";
import { canonicalDigest } from "../shared/contracts/canonical";
import { readJson, writeJson } from "../shared/contracts/jsonTransport";
import {
  documentDigest,
  parseResolvedSettings,
  parseWorkspaceManifest,
} from "../shared/contracts/workspace";
import { createWorkspaceArchive } from "../shared/storage/workspaceArchive";
import type { WorkspaceArchivePreview } from "../shared/storage/workspaceArchive";
import { displayTheme } from "./displayTheme";

/** Literal values produced by the real Python resolver, with test-only raw owners. */
interface Fixture {
  readonly document: { readonly body: Readonly<Record<string, unknown>> };
  readonly manifest: unknown;
  readonly raw: readonly {
    readonly digest: string;
    readonly record: {
      readonly schema: string;
      readonly body: { readonly role: string; readonly synthetic: boolean };
    };
  }[];
}
const fixture = readJson(fixtureText) as Fixture;

/**
 * Admit the real settings archive, optionally with another recorded theme.
 *
 * @param theme - Theme to record instead of the resolver's value, or `undefined` to keep it.
 * @returns The admitted archive preview.
 */
async function admitted(theme?: string): Promise<WorkspaceArchivePreview> {
  const body = fixture.document.body;
  const effective = body["effective"] as Readonly<Record<string, unknown>>;
  const payload =
    theme === undefined
      ? fixture.document
      : { ...fixture.document, body: { ...body, effective: { ...effective, theme } } };
  const manifest = parseWorkspaceManifest(fixture.manifest);
  const document = parseResolvedSettings(payload);
  if (!manifest.ok || !document.ok) throw new Error("Shared fixture refused");
  const raw = new Map(
    fixture.raw.map(({ digest, record }) => [
      digest,
      { schema: record.schema, content: new TextEncoder().encode(writeJson(record)) },
    ]),
  );
  return createWorkspaceArchive(
    manifest.value,
    new Map([[await documentDigest(document.value), document.value]]),
    raw,
    new Map(),
    new Map([
      [
        "review_fixture.v1",
        async (content: Uint8Array) => {
          const record = readJson(
            new TextDecoder().decode(content),
          ) as Fixture["raw"][number]["record"];
          if (record.schema !== "review_fixture.v1" || record.body.synthetic !== true)
            throw new Error("Synthetic owner required");
          return {
            schema: record.schema,
            kind: record.body.role,
            digest: await canonicalDigest(record.schema, record),
          };
        },
      ],
    ]),
  );
}

describe("display theme", () => {
  it("follows the theme recorded by the real resolver", async () => {
    expect(displayTheme(await admitted())).toBe("dark");
    expect(displayTheme(await admitted("light"))).toBe("light");
  });

  it("leaves an unknown value or no archive to the system preference", async () => {
    expect(displayTheme(await admitted("solarized"))).toBeNull();
    expect(displayTheme(null)).toBeNull();
  });

  it("refuses a malformed archive without choosing a set", () => {
    const preview = { json: "{}" } as WorkspaceArchivePreview;
    expect(displayTheme(preview)).toBeNull();
  });
});
