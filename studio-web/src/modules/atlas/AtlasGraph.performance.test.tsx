// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — graph tiers, query latency, cancellation and allocation budgets

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import casesText from "../../../../data/studio/resource_budget_verdict_cases_20261123.json?raw";
import { InThreadAtlasWorker } from "../../../test-support/atlasWorker";
import {
  atlasTiers,
  tierNodeId,
  tierPin,
  tierSnapshot,
  tierSnapshotDocument,
} from "../../../test-support/atlasTiers";
import { bindKuramoto, readBounds } from "../../shared/browser/kuramoto";
import type { KuramotoExports } from "../../shared/browser/kuramoto";
import { atlasSnapshotDigest, parseAtlasSnapshot } from "../../shared/contracts/atlas";
import {
  atlasEdgeCeiling,
  atlasMaxVisibleEdges,
  atlasNodeCeiling,
  runAtlasQuery,
} from "../../shared/contracts/atlasQuery";
import {
  type Budget,
  committedBudget,
  committedBudgets,
  judgeBudget,
  parseBudgets,
} from "../../shared/performance/budgetVerdict";
import {
  admitKuramotoResources,
  browserResourcePolicy,
} from "../../shared/resources/kuramotoResources";
import { openAtlasSession } from "./atlasSession";
import { atlasViewQuery, atlasVisibleNodes } from "./atlasViewModel";

/** One shared verdict case, which the Python rule runs too. */
interface VerdictCase {
  /** The budget payload. */ readonly budget: unknown;
  /** The measurement, or null. */ readonly measured: number | null;
  /** The expected verdict. */ readonly status: string;
}
const cases = JSON.parse(casesText) as { readonly cases: readonly VerdictCase[] };

/** Snapshot digests of the frozen tiers; a changed generator changes them. */
const frozenTierDigests = [
  "179d85ee4f296f636223c60ccb27e8915be46fa712b77795a4067967778f6a95",
  "5fec283fb0ee18576daf602b5878eef9db435bd124de2aebe909ef03df1a71a7",
  "2d93d95b156099d611e2cc20320709b45a007b1a5fc25f6cfa47694fd922380a",
  "3b0e272eb561060e9eab78506364c433b81b099e7a1f802314f13f393a9991a6",
];
const largest = atlasTiers[atlasTiers.length - 1] ?? atlasTiers[0];

/** Nearest-rank 95th percentile. */
function p95(samples: readonly number[]): number {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.ceil(0.95 * sorted.length) - 1] ?? Number.NaN;
}

let kernel: KuramotoExports;
beforeAll(async () => {
  const bytes = readFileSync(
    resolve(
      "..",
      "scpn_quantum_engine/studio_wasm_kernel/target/wasm32-unknown-unknown/release/scpn_quantum_studio_wasm_kernel.wasm",
    ),
  );
  kernel = (await WebAssembly.instantiate(bytes, {})).instance
    .exports as unknown as KuramotoExports;
});

describe("Workbench resource budgets", () => {
  it("test_workbench_resource_budgets_01: the 100 / 1 000 / 10 000-node tiers up to 100 000 relations are frozen and shown within the visible bound", async () => {
    expect(atlasTiers.map((tier) => [tier.nodes, tier.relations])).toEqual([
      [100, 300],
      [1_000, 3_000],
      [10_000, 30_000],
      [10_000, 100_000],
    ]);
    expect([atlasNodeCeiling, atlasEdgeCeiling]).toEqual([largest?.nodes, largest?.relations]);
    const digests: string[] = [];
    for (const tier of atlasTiers) {
      const snapshot = tierSnapshot(tier);
      digests.push(await atlasSnapshotDigest(snapshot));
      const outcome = await runAtlasQuery(snapshot, atlasViewQuery("architecture", "", null, 1));
      if (!outcome.ok) throw new Error(outcome.message);
      expect(outcome.value.total).toBe(tier.nodes);
      expect(outcome.value.nodes.length).toBeLessThanOrEqual(atlasVisibleNodes);
      expect(outcome.value.edges.length).toBeLessThanOrEqual(atlasMaxVisibleEdges);
      expect(
        judgeBudget(committedBudget("visible_subgraph_nodes"), outcome.value.nodes.length).status,
      ).toBe("pass");
      expect(
        judgeBudget(committedBudget("visible_subgraph_edges"), outcome.value.edges.length).status,
      ).toBe("pass");
    }
    expect(digests).toEqual(frozenTierDigests);
    const above = parseAtlasSnapshot(tierSnapshotDocument({ nodes: 10_001, relations: 0 }));
    if (!above.ok) throw new Error(above.code);
    const refused = await runAtlasQuery(above.value, atlasViewQuery("architecture", "", null, 1));
    expect(refused.ok ? null : refused.code).toBe("snapshot_too_large");
    expect(() => tierSnapshotDocument({ nodes: 10, relations: 15 })).toThrow();
  }, 240_000);

  it("test_workbench_resource_budgets_02: a bounded neighbourhood query at the largest tier keeps its p95 under the declared target", async () => {
    const snapshot = tierSnapshot(largest ?? { nodes: 1, relations: 0 });
    const query = atlasViewQuery("architecture", "", tierNodeId(0), 2);
    const warm = await runAtlasQuery(snapshot, query);
    if (!warm.ok) throw new Error(warm.message);
    expect(warm.value.shape).toBe("neighbourhood");
    const samples: number[] = [];
    for (let run = 0; run < 20; run++) {
      const started = performance.now();
      const outcome = await runAtlasQuery(snapshot, query);
      samples.push(performance.now() - started);
      expect(outcome.ok).toBe(true);
    }
    expect(judgeBudget(committedBudget("graph_query_p95"), p95(samples)).status).toBe("pass");
  }, 240_000);

  it("test_workbench_resource_budgets_03: cancelling a large opening in its worker is acknowledged within the target", async () => {
    InThreadAtlasWorker.started = [];
    const pin = await tierPin(atlasTiers[2] ?? { nodes: 1, relations: 0 });
    const task = openAtlasSession(pin.text, "tier", () => new InThreadAtlasWorker());
    expect(task.tier).toBe("worker");
    const started = performance.now();
    task.cancel();
    expect(await task.result).toBeNull();
    const acknowledged = performance.now() - started;
    expect(InThreadAtlasWorker.started.map((worker) => worker.terminated)).toEqual([true]);
    expect(judgeBudget(committedBudget("worker_cancel_ack"), acknowledged).status).toBe("pass");
  }, 240_000);

  it("test_workbench_resource_budgets_04: a request above the injected policy is refused before any allocation", () => {
    const bounds = readBounds(kernel);
    const required = admitKuramotoResources(
      { n: 2, steps: 10, mode: "mean-field" },
      bounds,
    ).bytesRequired;
    if (required === null) throw new Error("the admission declares no byte requirement");
    let allocations = 0;
    const observed = {
      ...kernel,
      scpn_alloc(length: number) {
        allocations++;
        return kernel.scpn_alloc(length);
      },
    };
    const request = {
      mode: "mean-field" as const,
      omega: [0, 0.2],
      theta0: [0, 0.3],
      steps: 10,
      dt: 0.05,
      coupling: 1,
    };
    const policyB = { ...browserResourcePolicy(bounds), memoryBytes: required - 1n };
    const denied = bindKuramoto(observed, policyB)(request);
    expect(denied.ok).toBe(false);
    expect(allocations).toBe(0);
    const atBound = bindKuramoto(observed, { ...policyB, memoryBytes: required })(request);
    expect(atBound.ok).toBe(true);
    expect(allocations).toBeGreaterThan(0);
  });

  it("test_workbench_resource_budgets_05: a missed budget is recorded as a failure, never relaxed", () => {
    const query = committedBudget("graph_query_p95");
    expect(judgeBudget(query, 500).status).toBe("fail");
    expect(judgeBudget(query, 499.999).status).toBe("pass");
    expect(judgeBudget(query, null).status).toBe("unmeasured");
    expect(judgeBudget(committedBudget("retained_memory"), 1).status).toBe("recorded");
    expect(judgeBudget(committedBudget("trajectory_points"), 65_536).status).toBe("fail");
    expect(() => judgeBudget(query, Number.NaN)).toThrow();
    for (const item of cases.cases) {
      const [budget] = parseBudgets(
        JSON.stringify({
          schema: "scpn_qc_studio_resource_budgets_v1",
          note: "case",
          budgets: [item.budget],
        }),
      ) as readonly Budget[];
      expect(budget === undefined ? null : judgeBudget(budget, item.measured).status).toBe(
        item.status,
      );
    }
    expect(committedBudgets.map((budget) => budget.id)).toContain("initial_javascript");
  });

  it("refuses a malformed budget declaration", () => {
    const one = (budget: unknown) =>
      JSON.stringify({
        schema: "scpn_qc_studio_resource_budgets_v1",
        note: "n",
        budgets: [budget],
      });
    const good = { id: "x", metric: "m", comparator: "<", target: 1, unit: "ms" };
    expect(() => parseBudgets("{}")).toThrow("must be a scpn_qc_studio_resource_budgets_v1");
    expect(() =>
      parseBudgets(JSON.stringify({ schema: "scpn_qc_studio_resource_budgets_v1", budgets: [] })),
    ).toThrow("non-empty array");
    expect(() => parseBudgets(one("x"))).toThrow("must be an object");
    expect(() => parseBudgets(one({ ...good, comparator: "~" }))).toThrow(
      "comparator is not one of",
    );
    expect(() => parseBudgets(one({ ...good, target: null }))).toThrow("record-only budget");
    expect(() => parseBudgets(one({ ...good, target: "1" }))).toThrow("finite number");
    expect(() => parseBudgets(one({ ...good, unit: "" }))).toThrow("non-empty id");
    expect(() =>
      parseBudgets(
        JSON.stringify({ schema: "scpn_qc_studio_resource_budgets_v1", budgets: [good, good] }),
      ),
    ).toThrow("repeats an id");
    expect(() => committedBudget("absent")).toThrow("no committed budget absent");
  });
});
