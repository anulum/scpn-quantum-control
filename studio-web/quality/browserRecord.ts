// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser coverage source qualification

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { Profiler } from "node:inspector";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

/** Production source owners whose browser counters may augment Node coverage. */
export const browserOwners = new Set([
  "/src/shared/storage/workspaceStore.ts",
  "/src/shared/storage/workspaceArchive.ts",
  "/src/features/workspace/WorkspacePanel.tsx",
  "/src/features/workspace/useWorkspace.ts",
]);

/** Additional original facade required by the damaged-source browser scenario. */
export const panelBrowserOwner = "/src/QuantumStudioPanel.tsx";

/** Complete linked parameter cohort sharing the original archive and store owners. */
export const parameterBrowserOwners = new Set([
  ...browserOwners,
  "/src/features/parameters/parameterDraft.ts",
  "/src/features/parameters/parameterRevision.ts",
  "/src/features/parameters/ParameterEditor.tsx",
  "/src/features/parameters/ParameterWorkspace.tsx",
]);

/** Complete native workbench cohort, including the original storage/controller owners. */
export const workbenchBrowserOwners = new Set([
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
]);

/** Exact experiment cohort, including its single original archive and Workbench owners. */
export const experimentBrowserOwners = new Set([
  ...browserOwners,
  "/src/app/Workbench.tsx",
  "/src/features/experiments/kuramotoArtifacts.ts",
  "/src/features/experiments/experimentArchive.ts",
  "/src/features/experiments/experimentPlan.ts",
  "/src/features/experiments/useExperimentRun.ts",
  "/src/features/experiments/ExperimentRunner.tsx",
]);

/** Exact results cohort retaining original workspace storage and route ownership. */
export const resultBrowserOwners = new Set([
  ...browserOwners,
  "/src/app/Workbench.tsx",
  "/src/app/routes/ResultsView.tsx",
  "/src/features/results/resultModel.ts",
  "/src/features/results/resultSources.ts",
  "/src/features/results/resultExport.ts",
  "/src/features/results/ResultInspector.tsx",
  "/src/features/results/ResultLoader.tsx",
]);

/** Exact workflow cohort retaining its original Workbench and workspace owners. */
export const workflowBrowserOwners = new Set([
  ...browserOwners,
  "/src/app/Workbench.tsx",
  "/src/features/workflows/workflowModel.ts",
  "/src/features/workflows/workflowSweep.ts",
  "/src/features/workflows/workflowJournal.ts",
  "/src/features/workflows/workflowArchive.ts",
  "/src/features/workflows/workflowExecution.ts",
  "/src/features/workflows/useWorkflowRun.ts",
  "/src/features/workflows/WorkflowEditor.tsx",
  "/src/features/workflows/WorkflowRunner.tsx",
]);

/** Validated actual script, original source identity and qualified single-source map. */
export interface QualifiedBrowserRecord {
  /** Absolute original owner filename in this checkout. */
  readonly owner: string;
  /** Actual executed transformed code. */
  readonly code: string;
  /** Native ranges with an equivalent local URL; offsets are unchanged. */
  readonly coverage: Profiler.ScriptCoverage;
  /** Original mapping restricted to the exact verified source text. */
  readonly sourceMap: {
    /** Source Map v3 format retained from the executed script. */
    version: number;
    /** Single absolute checkout owner used for original-source conversion. */
    sources: string[];
    /** Exact source text verified against the checkout SHA-256. */
    sourcesContent: string[];
    /** Original symbolic names from the executed inline map. */
    names: string[];
    /** Original VLQ offset mappings; native ranges remain unchanged. */
    mappings: string;
  };
}

/** Require a structured evidence field without accepting arrays or null as objects. */
export function coverageObject(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("Coverage object required");
  return value as Record<string, unknown>;
}

function text(value: unknown): string {
  if (typeof value !== "string" || value.length === 0)
    throw new Error("Nonempty coverage text required");
  return value;
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function nativeFunctions(value: unknown, codeLength: number): Profiler.FunctionCoverage[] {
  if (!Array.isArray(value) || value.length === 0)
    throw new Error("Native function counters required");
  return value.map((item) => {
    const entry = coverageObject(item);
    if (
      typeof entry["functionName"] !== "string" ||
      typeof entry["isBlockCoverage"] !== "boolean" ||
      !Array.isArray(entry["ranges"]) ||
      entry["ranges"].length === 0
    )
      throw new Error("Malformed native function coverage");
    const ranges = entry["ranges"].map((item) => {
      const range = coverageObject(item);
      const startOffset = range["startOffset"];
      const endOffset = range["endOffset"];
      const count = range["count"];
      if (
        typeof startOffset !== "number" ||
        typeof endOffset !== "number" ||
        typeof count !== "number" ||
        !Number.isSafeInteger(startOffset) ||
        !Number.isSafeInteger(endOffset) ||
        !Number.isSafeInteger(count) ||
        startOffset < 0 ||
        endOffset <= startOffset ||
        endOffset > codeLength ||
        count < 0
      )
        throw new Error("Native coverage range outside actual script");
      return { startOffset, endOffset, count };
    });
    return {
      functionName: entry["functionName"],
      isBlockCoverage: entry["isBlockCoverage"],
      ranges,
    };
  });
}

/** Verify every browser counter against executed code and this checkout's original owner. */
export async function qualifyBrowserRecord(
  value: unknown,
  root: string,
  sourceOrigin: string,
  includeWorkbench = false,
  includeParameters = false,
  includeExperiments = false,
  includeResults = false,
  includeWorkflows = false,
): Promise<QualifiedBrowserRecord> {
  const record = coverageObject(value);
  const native = coverageObject(record["coverage"]);
  const url = new URL(text(native["url"]));
  const ownedOrigin = new URL(sourceOrigin);
  if (
    ownedOrigin.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(ownedOrigin.hostname) ||
    ownedOrigin.username ||
    ownedOrigin.password ||
    !ownedOrigin.port ||
    ownedOrigin.search ||
    ownedOrigin.hash ||
    ownedOrigin.pathname !== "/"
  )
    throw new Error("Owned root loopback source origin required");
  const admitted = includeWorkflows
    ? workflowBrowserOwners.has(url.pathname)
    : includeResults
      ? resultBrowserOwners.has(url.pathname)
      : includeExperiments
        ? experimentBrowserOwners.has(url.pathname)
        : includeParameters
          ? parameterBrowserOwners.has(url.pathname)
          : includeWorkbench
            ? workbenchBrowserOwners.has(url.pathname)
            : browserOwners.has(url.pathname) || url.pathname === panelBrowserOwner;
  if (
    url.origin !== ownedOrigin.origin ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !admitted
  )
    throw new Error("Unowned browser coverage source");
  const code = text(record["code"]);
  if (sha256(code) !== record["code_sha256"]) throw new Error("Executed script hash mismatch");
  const owner = resolve(root, `.${url.pathname}`);
  const original = await readFile(owner, "utf8");
  if (sha256(original) !== record["source_sha256"])
    throw new Error("Original source hash mismatch");
  if (/\/\*\s*(?:istanbul|v8|c8|node:coverage)\s+ignore\b/.test(original))
    throw new Error("Browser owner coverage exclusions are refused");
  const inline = [
    ...code.matchAll(
      /\/\/# sourceMappingURL=data:application\/json(?:;charset=utf-8)?;base64,([A-Za-z0-9+/=]+)/g,
    ),
  ];
  if (inline.length !== 1 || inline[0]?.[1] === undefined)
    throw new Error("One inline original source map required");
  const map = coverageObject(
    JSON.parse(Buffer.from(inline[0][1], "base64").toString("utf8")) as unknown,
  );
  if (
    map["version"] !== 3 ||
    !Array.isArray(map["sources"]) ||
    map["sources"].length !== 1 ||
    typeof map["sources"][0] !== "string" ||
    !Array.isArray(map["sourcesContent"]) ||
    map["sourcesContent"].length !== 1 ||
    map["sourcesContent"][0] !== original ||
    !Array.isArray(map["names"]) ||
    !map["names"].every((name) => typeof name === "string") ||
    typeof map["mappings"] !== "string" ||
    map["mappings"].length === 0 ||
    map["sections"] !== undefined
  )
    throw new Error("Source map does not identify the exact original owner");
  return {
    owner,
    code,
    coverage: {
      scriptId: text(native["scriptId"]),
      url: pathToFileURL(owner).href,
      functions: nativeFunctions(native["functions"], code.length),
    },
    sourceMap: {
      version: 3,
      sources: [owner],
      sourcesContent: [original],
      names: map["names"] as string[],
      mappings: map["mappings"],
    },
  };
}
