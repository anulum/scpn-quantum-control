// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual browser counter conversion acceptance

// @vitest-environment node
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { readBrowserCoverage } from "./browserCoverage";
import {
  browserOwners,
  coverageObject,
  experimentBrowserOwners,
  panelBrowserOwner,
  parameterBrowserOwners,
  qualifyBrowserRecord,
  resultBrowserOwners,
  workbenchBrowserOwners,
  workflowBrowserOwners,
} from "./browserRecord";

describe("original native counter conversion", () => {
  it("converts every actual workflow owner while retaining native zero ranges and evidence bytes", async () => {
    const filename = process.env["STUDIO_WORKFLOW_COVERAGE"];
    if (!filename) throw new Error("Supply actual experiment_workflow_runner source evidence");
    const before = await readFile(filename, "utf8");
    const maps = await readBrowserCoverage(filename, process.cwd());
    expect(new Set(maps.flatMap((map) => Object.keys(map)))).toEqual(
      new Set([...workflowBrowserOwners].map((owner) => resolve(process.cwd(), `.${owner}`))),
    );
    expect(
      maps.some((map) =>
        Object.values(map).some((file) => Object.values(file.s).some((count) => count === 0)),
      ),
    ).toBe(true);
    expect(await readFile(filename, "utf8")).toBe(before);
  });
  it.each([...workflowBrowserOwners])(
    "refuses actual workflow evidence missing source owner %s",
    async (omitted) => {
      const filename = process.env["STUDIO_WORKFLOW_COVERAGE"];
      if (!filename) throw new Error("Supply actual experiment_workflow_runner source evidence");
      await readBrowserCoverage(filename, process.cwd());
      const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
      if (!Array.isArray(evidence["native_v8_coverage"]))
        throw new Error("Actual workflow counters missing");
      evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
        const native = coverageObject(coverageObject(value)["coverage"]);
        if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
        return new URL(native["url"]).pathname !== omitted;
      });
      const directory = await mkdtemp(join(tmpdir(), "studio-workflow-owner-refusal-"));
      try {
        const rejected = join(directory, "omitted-owner.json");
        await writeFile(rejected, JSON.stringify(evidence), "utf8");
        await expect(readBrowserCoverage(rejected, process.cwd())).rejects.toThrow(
          "Native workspace coverage has missing production owners",
        );
      } finally {
        await rm(directory, { recursive: true });
      }
    },
  );
  it("converts all actual experiment owners while retaining uncovered native ranges", async () => {
    const filename = process.env["STUDIO_EXPERIMENT_COVERAGE"];
    if (!filename) throw new Error("Supply actual local_experiment_journey evidence");
    const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
    if (!Array.isArray(evidence["native_v8_coverage"]))
      throw new Error("Actual native counters missing");
    const untouched = JSON.stringify(evidence["native_v8_coverage"]);
    const maps = await readBrowserCoverage(filename, process.cwd());
    expect(new Set(maps.flatMap((map) => Object.keys(map)))).toEqual(
      new Set([...experimentBrowserOwners].map((owner) => resolve(process.cwd(), `.${owner}`))),
    );
    expect(
      maps.some((map) =>
        Object.values(map).some((file) => Object.values(file.s).some((count) => count === 0)),
      ),
    ).toBe(true);
    const after = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
    expect(JSON.stringify(after["native_v8_coverage"])).toBe(untouched);
  });
  it.each([...experimentBrowserOwners])("requires actual experiment owner %s", async (omitted) => {
    const filename = process.env["STUDIO_EXPERIMENT_COVERAGE"];
    if (!filename) throw new Error("Supply actual local_experiment_journey evidence");
    await readBrowserCoverage(filename, process.cwd());
    const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
    if (!Array.isArray(evidence["native_v8_coverage"]))
      throw new Error("Actual native counters missing");
    evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
      const native = coverageObject(coverageObject(value)["coverage"]);
      if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
      return new URL(native["url"]).pathname !== omitted;
    });
    const directory = await mkdtemp(join(tmpdir(), "studio-experiment-owner-refusal-"));
    try {
      const rejected = join(directory, "omitted-owner.json");
      await writeFile(rejected, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(rejected, process.cwd())).rejects.toThrow(
        "Native workspace coverage has missing production owners",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  });
  it("admits all eight original owners from the actual linked parameter journey", async () => {
    const filename = process.env["STUDIO_PARAMETER_COVERAGE"];
    if (!filename) throw new Error("Supply actual parameter_graph_editor evidence");
    const maps = await readBrowserCoverage(filename, process.cwd());
    expect(new Set(maps.flatMap((map) => Object.keys(map)))).toEqual(
      new Set([...parameterBrowserOwners].map((owner) => resolve(process.cwd(), `.${owner}`))),
    );
  });
  it.each([...parameterBrowserOwners])(
    "keeps %s mandatory in actual parameter evidence",
    async (omitted) => {
      const filename = process.env["STUDIO_PARAMETER_COVERAGE"];
      if (!filename) throw new Error("Supply actual parameter_graph_editor evidence");
      await readBrowserCoverage(filename, process.cwd());
      const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
      if (!Array.isArray(evidence["native_v8_coverage"]))
        throw new Error("Actual native counters missing");
      evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
        const native = coverageObject(coverageObject(value)["coverage"]);
        if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
        return new URL(native["url"]).pathname !== omitted;
      });
      const directory = await mkdtemp(join(tmpdir(), "studio-parameter-owner-refusal-"));
      try {
        const rejected = join(directory, "omitted-owner.json");
        await writeFile(rejected, JSON.stringify(evidence), "utf8");
        await expect(readBrowserCoverage(rejected, process.cwd())).rejects.toThrow(
          "Native workspace coverage has missing production owners",
        );
      } finally {
        await rm(directory, { recursive: true });
      }
    },
  );
  it("admits all fourteen owners from the actual workbench navigation journey", async () => {
    const filename = process.env["STUDIO_WORKBENCH_COVERAGE"];
    if (!filename) throw new Error("Run actual workbench_navigation and supply its evidence path");
    const maps = await readBrowserCoverage(filename, process.cwd());
    const actual = new Set(maps.flatMap((map) => Object.keys(map)));
    expect(actual).toEqual(
      new Set(
        [
          ...browserOwners,
          panelBrowserOwner,
          "/src/features/catalogue/CapabilityCatalogue.tsx",
          "/src/app/Workbench.tsx",
          "/src/app/WorkbenchInspector.tsx",
          "/src/app/RouteBoundary.tsx",
          "/src/app/routing.ts",
          "/src/app/useWorkbenchRoute.ts",
          "/src/app/routes/BuildView.tsx",
          "/src/app/routes/ResultsView.tsx",
          "/src/app/routes/UnavailableView.tsx",
        ].map((owner) => resolve(process.cwd(), `.${owner}`)),
      ),
    );
  });
  it.each([...workbenchBrowserOwners])(
    "keeps %s mandatory in actual workbench evidence",
    async (omitted) => {
      const filename = process.env["STUDIO_WORKBENCH_COVERAGE"];
      if (!filename) throw new Error("Supply actual workbench_navigation evidence");
      await readBrowserCoverage(filename, process.cwd());
      const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
      if (!Array.isArray(evidence["native_v8_coverage"]))
        throw new Error("Actual native counters missing");
      evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
        const native = coverageObject(coverageObject(value)["coverage"]);
        if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
        return new URL(native["url"]).pathname !== omitted;
      });
      const directory = await mkdtemp(join(tmpdir(), "studio-workbench-owner-refusal-"));
      try {
        const rejected = join(directory, "omitted-owner.json");
        await writeFile(rejected, JSON.stringify(evidence), "utf8");
        await expect(readBrowserCoverage(rejected, process.cwd())).rejects.toThrow(
          "Native workspace coverage has missing production owners",
        );
      } finally {
        await rm(directory, { recursive: true });
      }
    },
  );
  it("refuses a recorded script whose source mapping omits executable ownership", async () => {
    const actualFilename = process.env["STUDIO_WORKSPACE_COVERAGE"];
    if (!actualFilename)
      throw new Error("Run native workspace_recovery first and supply its actual evidence path");
    const evidence = coverageObject(JSON.parse(await readFile(actualFilename, "utf8")) as unknown);
    if (!Array.isArray(evidence["native_v8_coverage"]))
      throw new Error("Actual native counters missing");
    const record = coverageObject(evidence["native_v8_coverage"][0]);
    if (typeof record["code"] !== "string") throw new Error("Actual script text missing");
    const originalCode = record["code"];
    const changed = originalCode
      .replace(
        /(\/\/# sourceMappingURL=data:application\/json(?:;charset=utf-8)?;base64,)([A-Za-z0-9+/=]+)/,
        (_match: string, prefix: string, encoded: string) => {
          const map = coverageObject(
            JSON.parse(Buffer.from(encoded, "base64").toString("utf8")) as unknown,
          );
          map["mappings"] = "A";
          return prefix + Buffer.from(JSON.stringify(map), "utf8").toString("base64");
        },
      )
      .padEnd(originalCode.length, " ");
    expect(changed).not.toBe(record["code"]);
    record["code"] = changed;
    record["code_sha256"] = createHash("sha256").update(changed, "utf8").digest("hex");
    const directory = await mkdtemp(join(tmpdir(), "studio-unmapped-owner-refusal-"));
    try {
      const filename = join(directory, "unmapped-owner.json");
      await writeFile(filename, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(filename, process.cwd())).rejects.toThrow(
        "Converted counters escaped or omitted the original owner",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  });
  it("admits the original facade and all workspace owners from the actual damaged-source journey", async () => {
    const filename = process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
    if (!filename)
      throw new Error(
        "Run the actual workspace_panel_refusal journey and supply its evidence path",
      );
    const maps = await readBrowserCoverage(filename, process.cwd());
    expect(new Set(maps.flatMap((map) => Object.keys(map)))).toEqual(
      new Set(
        [...browserOwners, panelBrowserOwner].map((owner) => resolve(process.cwd(), `.${owner}`)),
      ),
    );
  });
  it.each([panelBrowserOwner, "/src/shared/storage/workspaceStore.ts"])(
    "keeps %s mandatory in damaged-source evidence",
    async (omitted) => {
      const actualFilename = process.env["STUDIO_PANEL_REFUSAL_COVERAGE"];
      if (!actualFilename)
        throw new Error(
          "Run the actual workspace_panel_refusal journey and supply its evidence path",
        );
      await readBrowserCoverage(actualFilename, process.cwd());
      const evidence = coverageObject(
        JSON.parse(await readFile(actualFilename, "utf8")) as unknown,
      );
      if (!Array.isArray(evidence["native_v8_coverage"]))
        throw new Error("Actual native counters missing");
      evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
        const native = coverageObject(coverageObject(value)["coverage"]);
        if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
        return new URL(native["url"]).pathname !== omitted;
      });
      const directory = await mkdtemp(join(tmpdir(), "studio-panel-owner-refusal-"));
      try {
        const filename = join(directory, "omitted-owner.json");
        await writeFile(filename, JSON.stringify(evidence), "utf8");
        await expect(readBrowserCoverage(filename, process.cwd())).rejects.toThrow(
          "Native workspace coverage has missing production owners",
        );
      } finally {
        await rm(directory, { recursive: true });
      }
    },
  );
  it("qualifies actual journey scripts and preserves every original owner", async () => {
    const filename = process.env["STUDIO_WORKSPACE_COVERAGE"];
    if (!filename)
      throw new Error("Run native workspace_recovery first and supply its actual evidence path");
    const root = process.cwd();
    const actual = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
    if (!Array.isArray(actual["native_v8_coverage"]) || typeof actual["source_url"] !== "string")
      throw new Error("Actual native counters and source origin missing");
    for (const record of actual["native_v8_coverage"]) {
      const qualified = await qualifyBrowserRecord(record, root, actual["source_url"]);
      expect(qualified.coverage.functions.length).toBeGreaterThan(0);
    }
    const maps = await readBrowserCoverage(filename, root);
    const owners = new Set(maps.flatMap((map) => Object.keys(map)));
    expect(owners).toEqual(new Set([...browserOwners].map((owner) => resolve(root, `.${owner}`))));
    for (const map of maps) {
      for (const file of Object.values(map)) {
        expect(Object.keys(file.statementMap).length).toBeGreaterThan(0);
        expect(
          Object.values(file.s).every((count) => Number.isSafeInteger(count) && count >= 0),
        ).toBe(true);
      }
    }
  });
  it.each([
    { scenario: "workspace_recovery", passed: false },
    { scenario: "workspace_recovery", passed: true, coverage_percentage: "100" },
    { scenario: "other", passed: true },
    {
      scenario: "workspace_recovery",
      passed: true,
      coverage_percentage: "not_calculated",
      native_v8_coverage: [],
    },
  ])("refuses incomplete or substituted evidence %j", async (evidence) => {
    const directory = await mkdtemp(join(tmpdir(), "studio-coverage-refusal-"));
    const filename = join(directory, "invalid.json");
    try {
      await writeFile(filename, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(filename, process.cwd())).rejects.toThrow(
        "Successful native workspace journey evidence required",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  });
  it("refuses an actual journey with one complete production owner omitted", async () => {
    const actualFilename = process.env["STUDIO_WORKSPACE_COVERAGE"];
    if (!actualFilename)
      throw new Error("Run native workspace_recovery first and supply its actual evidence path");
    const root = process.cwd();
    await readBrowserCoverage(actualFilename, root);
    const evidence = coverageObject(JSON.parse(await readFile(actualFilename, "utf8")) as unknown);
    if (!Array.isArray(evidence["native_v8_coverage"]))
      throw new Error("Actual native counters missing");
    const original = evidence["native_v8_coverage"];
    const retained = original.filter((value) => {
      const native = coverageObject(coverageObject(value)["coverage"]);
      if (typeof native["url"] !== "string") throw new Error("Actual native source URL missing");
      return new URL(native["url"]).pathname !== "/src/shared/storage/workspaceStore.ts";
    });
    expect(retained.length).toBeGreaterThan(0);
    expect(retained.length).toBeLessThan(original.length);
    evidence["native_v8_coverage"] = retained;
    const directory = await mkdtemp(join(tmpdir(), "studio-coverage-owner-refusal-"));
    try {
      const filename = join(directory, "omitted-owner.json");
      await writeFile(filename, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(filename, root)).rejects.toThrow(
        "Native workspace coverage has missing production owners",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  });
  it("refuses native counters without their recorded source origin", async () => {
    const actualFilename = process.env["STUDIO_WORKSPACE_COVERAGE"];
    if (!actualFilename)
      throw new Error("Run native workspace_recovery first and supply its actual evidence path");
    await readBrowserCoverage(actualFilename, process.cwd());
    const evidence = coverageObject(JSON.parse(await readFile(actualFilename, "utf8")) as unknown);
    delete evidence["source_url"];
    const directory = await mkdtemp(join(tmpdir(), "studio-coverage-origin-refusal-"));
    try {
      const filename = join(directory, "missing-origin.json");
      await writeFile(filename, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(filename, process.cwd())).rejects.toThrow(
        "Native coverage source origin missing",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  });
});

it("converts actual result source maps while preserving the original native zero counters", async () => {
  const filename = process.env["STUDIO_RESULT_COVERAGE"];
  if (!filename) throw new Error("Supply actual result_value_inspector source evidence");
  const original = await readFile(filename, "utf8");
  const maps = await readBrowserCoverage(filename, process.cwd());
  expect(new Set(maps.flatMap((map) => Object.keys(map)))).toEqual(
    new Set([...resultBrowserOwners].map((owner) => resolve(process.cwd(), `.${owner}`))),
  );
  expect(
    maps.some((map) =>
      Object.values(map).some((file) => Object.values(file.s).some((count) => count === 0)),
    ),
  ).toBe(true);
  expect(await readFile(filename, "utf8")).toBe(original);
});

it.each([...resultBrowserOwners])(
  "refuses actual result evidence missing production owner %s",
  async (omitted) => {
    const filename = process.env["STUDIO_RESULT_COVERAGE"];
    if (!filename) throw new Error("Supply actual result_value_inspector source evidence");
    await readBrowserCoverage(filename, process.cwd());
    const evidence = coverageObject(JSON.parse(await readFile(filename, "utf8")) as unknown);
    if (!Array.isArray(evidence["native_v8_coverage"]))
      throw new Error("Actual result counters missing");
    evidence["native_v8_coverage"] = evidence["native_v8_coverage"].filter((value) => {
      const native = coverageObject(coverageObject(value)["coverage"]);
      if (typeof native["url"] !== "string") throw new Error("Actual source URL missing");
      return new URL(native["url"]).pathname !== omitted;
    });
    const directory = await mkdtemp(join(tmpdir(), "studio-result-owner-refusal-"));
    try {
      const rejected = join(directory, "omitted-owner.json");
      await writeFile(rejected, JSON.stringify(evidence), "utf8");
      await expect(readBrowserCoverage(rejected, process.cwd())).rejects.toThrow(
        "Native workspace coverage has missing production owners",
      );
    } finally {
      await rm(directory, { recursive: true });
    }
  },
);
