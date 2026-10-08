// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact original workflow sweep tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { readJson } from "../../shared/contracts";
import { parseWorkflow } from "./workflowModel";
import { buildWorkflowCells } from "./workflowSweep";

beforeAll(() =>
  Object.defineProperty(globalThis, "crypto", { value: webcrypto, configurable: true }),
);
function fixture(): Record<string, unknown> {
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  return corpus["workflow"] as Record<string, unknown>;
}

describe("bounded original sweep plan", () => {
  it("produces six independent literal Cartesian coordinates without executing", async () => {
    const definition = parseWorkflow(fixture());
    const cells = await buildWorkflowCells(definition);
    expect(cells.map((c) => c.coordinate)).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 1],
      [1, 2],
    ]);
    expect(cells.map((c) => c.index)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(new Set(cells.map((c) => c.id)).size).toBe(6);
    expect(cells.map((c) => c.overrides["trace"]?.["optimisation_level"])).toEqual([
      0n,
      1n,
      2n,
      0n,
      1n,
      2n,
    ]);
    expect((await buildWorkflowCells(definition)).map((c) => c.id)).toEqual(cells.map((c) => c.id));
    const corpus = readJson(
      readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
    ) as Record<string, unknown>;
    expect(cells.at(0)?.workflow_digest).toBe(corpus["expected_workflow_digest"]);
    expect(cells.map((c) => c.id)).toEqual(corpus["expected_cell_ids"]);
  });
  it("keeps maximum uint64 seed identities and explicit numerical bindings", async () => {
    const original = fixture();
    const sweep = (original["body"] as Record<string, unknown>)["sweep"] as Record<string, unknown>;
    Object.assign(sweep, {
      seeds: ["0", "18446744073709551615"],
      seed_binding: { stage_id: "source", parameter: "seed" },
      evaluation_budget: 24n,
    });
    const cells = await buildWorkflowCells(parseWorkflow(original));
    expect(cells.map((c) => c.seed)).toEqual([
      ...Array(6).fill("0"),
      ...Array(6).fill("18446744073709551615"),
    ]);
    expect(cells.at(6)?.overrides["source"]?.["seed"]).toBe(18446744073709551615n);
    expect(new Set(cells.map((c) => c.id)).size).toBe(12);
  });
  it("refuses an insufficient evaluation budget before coordinates exist", () => {
    const original = fixture();
    const sweep = (original["body"] as Record<string, unknown>)["sweep"] as Record<string, unknown>;
    sweep["evaluation_budget"] = 11n;
    expect(() => parseWorkflow(original)).toThrow("budget");
  });
});
