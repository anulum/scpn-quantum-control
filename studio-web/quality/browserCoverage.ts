// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native browser coverage conversion

import { readFile } from "node:fs/promises";
import { convert } from "ast-v8-to-istanbul";
import type { CoverageMapData } from "istanbul-lib-coverage";
import { parseAstAsync } from "vitest/node";

import { browserOwners, coverageObject, panelBrowserOwner, qualifyBrowserRecord } from "./browserRecord";

/** Convert a successful real browser journey, retaining all original native zero counters. */
export async function readBrowserCoverage(filename: string, root: string): Promise<CoverageMapData[]> {
  const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
  if ((evidence["scenario"] !== "workspace_recovery" && evidence["scenario"] !== "workspace_panel_refusal") || evidence["passed"] !== true || evidence["coverage_percentage"] !== "not_calculated" || !Array.isArray(evidence["native_v8_coverage"]) || evidence["native_v8_coverage"].length === 0) throw new Error("Successful native workspace journey evidence required");
  if (typeof evidence["source_url"] !== "string") throw new Error("Native coverage source origin missing");
  const covered = new Set<string>();
  const maps: CoverageMapData[] = [];
  for (const record of evidence["native_v8_coverage"]) {
    const qualified = await qualifyBrowserRecord(record, root, evidence["source_url"]);
    const converted = await convert({ code: qualified.code, ast: await parseAstAsync(qualified.code), wrapperLength: 0, coverage: qualified.coverage, sourceMap: qualified.sourceMap });
    const entries = Object.entries(converted);
    if (entries.length !== 1 || entries[0]?.[0] !== qualified.owner || entries[0][1].path !== qualified.owner || Object.keys(entries[0][1].statementMap).length === 0) throw new Error("Converted counters escaped or omitted the original owner");
    covered.add(qualified.owner.slice(root.length).replaceAll("\\", "/"));
    maps.push(converted);
  }
  const required = evidence["scenario"] === "workspace_panel_refusal" ? new Set([...browserOwners, panelBrowserOwner]) : browserOwners;
  if ([...required].some(owner => !covered.has(owner))) throw new Error("Native workspace coverage has missing production owners");
  return maps;
}
