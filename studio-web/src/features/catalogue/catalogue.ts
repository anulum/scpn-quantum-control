// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — catalogue

import type { Loaded, StudioManifestView } from "../../panel/data";

/** A source-owned verb projection; routes are bounded instruments, not general dispatch. */
export interface CatalogueRow {
  /** Exact manifest verb. */ readonly verb: string;
  /** Public local CLI entry. */ readonly api: string;
  /** Runtime needed by the displayed instrument or local handler. */ readonly runtime: string;
  /** Declared library backends, not an installation claim. */ readonly backends: readonly string[];
  /** Evidence schemas emitted by the owning verb. */ readonly evidence: readonly string[];
  /** Actual mounted instrument fragment, or no browser route. */ readonly route: string | null;
  /** Instrument name with its narrower scope. */ readonly label: string;
  /** Scope limit or reason no route exists. */ readonly reason: string;
  /** Source-owned settings description. */ readonly settings: string;
}
/** Deterministic projection bound to one declared source version and surface. */
export interface Catalogue {
  /** Content-addressed projection identity. */ readonly identity: string;
  /** Studio owning the declaration. */ readonly studio: string;
  /** Exact source distribution version. */ readonly version: string;
  /** Source manifest surface digest. */ readonly sourceDigest: string;
  /** Sorted unique verb rows. */ readonly rows: readonly CatalogueRow[];
}
/** Observed kernel load state; absence of a probe remains unknown. */
export interface RuntimeAvailability {
  /** True only after verifying the bounded committed input through the real kernel. */ readonly available: boolean;
  /** Current reason, including missing optional runtime. */ readonly reason: string;
}
/** Exact mounted browser instruments; arbitrary URLs are never accepted. */
export const catalogueRoutes: Readonly<Record<string, string>> = {
  compile: "#/build/compile-recompute",
  differentiate: "#/results/program-ad-replay",
};
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function strings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(item => typeof item === "string" && item.length > 0);
}
function digest(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}
/** Parse only the additive catalogue extension; reject malformed rows and unknown routes. */
export function parseCatalogue(raw: unknown): Loaded<Catalogue> {
  if (!record(raw) || !record(raw["architecture_map"]) || !record(raw["architecture_map"]["catalogue"])) {
    return { ok: false, reason: "Catalogue extension missing" };
  }
  const body = raw["architecture_map"]["catalogue"];
  if (body["schema"] !== "studio-capability-catalogue.v1" || !digest(body["identity"]) ||
      !digest(body["source_digest"]) || typeof body["source_studio"] !== "string" ||
      typeof body["source_version"] !== "string" || !Array.isArray(body["rows"])) {
    return { ok: false, reason: "Catalogue identity or schema malformed" };
  }
  const rows: CatalogueRow[] = [];
  for (const value of body["rows"]) {
    if (!record(value) || typeof value["verb"] !== "string" || typeof value["api"] !== "string" ||
        typeof value["label"] !== "string" || typeof value["reason"] !== "string" ||
        typeof value["settings"] !== "string" || !strings(value["backends"]) || !strings(value["evidence"])) {
      return { ok: false, reason: "Catalogue row malformed" };
    }
    const route = value["route"];
    const runtime = value["runtime"];
    if ((route !== null && (route !== catalogueRoutes[value["verb"]] || typeof route !== "string")) ||
        (runtime !== "local-python" && runtime !== "browser-wasm") ||
        runtime !== (route === null ? "local-python" : "browser-wasm")) {
      return { ok: false, reason: "Catalogue route or runtime unsupported" };
    }
    rows.push({ verb: value["verb"], api: value["api"], label: value["label"], reason: value["reason"],
      settings: value["settings"], backends: value["backends"], evidence: value["evidence"], route, runtime });
  }
  rows.sort((a, b) => a.verb < b.verb ? -1 : a.verb > b.verb ? 1 : 0);
  if (new Set(rows.map(row => row.verb)).size !== rows.length) {
    return { ok: false, reason: "Catalogue carries duplicate verbs" };
  }
  return { ok: true, value: { identity: body["identity"], studio: body["source_studio"],
    version: body["source_version"], sourceDigest: body["source_digest"], rows } };
}
/** Compare the displayed projection with the caller's actual source, never version alone. */
export function catalogueMatchesSource(catalogue: Catalogue, source: StudioManifestView): boolean {
  return catalogue.studio === source.studio && catalogue.version === source.studioVersion &&
    catalogue.sourceDigest === source.contentDigest && catalogue.rows.length === source.verbs.length &&
    catalogue.rows.every(row => source.verbs.some(verb => verb.verb === row.verb &&
      JSON.stringify(verb.backends) === JSON.stringify(row.backends) &&
      JSON.stringify(verb.produces) === JSON.stringify(row.evidence)));
}
