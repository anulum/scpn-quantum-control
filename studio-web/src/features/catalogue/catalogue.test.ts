// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — catalogue.test

import { describe, expect, it } from "vitest";
import raw from "../../../../docs/_generated/studio_manifest.json";
import { parseManifest } from "../../panel/data";
import { catalogueMatchesSource, parseCatalogue } from "./catalogue";

function documentWith(body: unknown): unknown { return { architecture_map: { catalogue: body } }; }
const body = raw.architecture_map.catalogue;
describe("source-backed catalogue parser", () => {
  it("preserves deterministic rows and accepts the committed source", () => {
    const parsed = parseCatalogue(raw);
    const manifest = parseManifest(raw);
    if (!parsed.ok || !manifest.ok) throw new Error("committed projection refused");
    expect(parsed.value.rows.map(row => row.verb)).toEqual(["analyse", "benchmark", "compile", "differentiate", "execute", "mitigate", "replay", "simulate", "validate"]);
    expect(catalogueMatchesSource(parsed.value, manifest.value)).toBe(true);
    expect(parseCatalogue(documentWith({ ...body, rows: [...body.rows].reverse() }))).toEqual(parsed);
    for (const changed of [
      { ...manifest.value, studio: "other" },
      { ...manifest.value, contentDigest: "sha256:" + "0".repeat(64) },
      { ...manifest.value, verbs: [] },
      { ...manifest.value, verbs: manifest.value.verbs.map(v => ({ ...v, backends: [] })) },
      { ...manifest.value, verbs: manifest.value.verbs.map(v => ({ ...v, produces: [] })) },
    ]) expect(catalogueMatchesSource(parsed.value, changed)).toBe(false);
  });
  it.each([null, [], {}, { architecture_map: [] }, { architecture_map: { catalogue: null } }])("refuses missing extension %j", value => {
    expect(parseCatalogue(value).ok).toBe(false);
  });
  it.each([
    { schema: "future" }, { identity: null }, { identity: "bad" }, { source_digest: "bad" },
    { source_studio: null }, { source_version: null }, { rows: null },
  ])("refuses invalid identity %j", change => {
    expect(parseCatalogue(documentWith({ ...body, ...change })).ok).toBe(false);
  });
  it.each([
    null, [], { verb: null }, { api: 1 }, { label: null }, { reason: null }, { settings: null },
    { backends: null }, { backends: [1] }, { backends: [""] }, { evidence: [null] },
    { route: "https://example.com" }, { route: "#/missing" }, { route: 1 },
    { runtime: "pretend-runtime" }, { route: undefined },
  ])("refuses malformed row %j", change => {
    const row = change === null || Array.isArray(change) ? change : { ...body.rows[0], ...change };
    expect(parseCatalogue(documentWith({ ...body, rows: [row] })).ok).toBe(false);
  });
  it("refuses duplicate verbs", () => {
    expect(parseCatalogue(documentWith({ ...body, rows: [body.rows[0], body.rows[0]] })).ok).toBe(false);
  });
});
