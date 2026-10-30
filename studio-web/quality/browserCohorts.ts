// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser owners derived from the compiled registry

import { readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import ts from "typescript";

/** Exact source cohorts shared by native capture and counter qualification. */
export interface BrowserSourceCohorts {
  /** Actual local module entries named by the compiled registry. */ readonly compiled: readonly string[];
  /** Standalone workspace with its compiled document composition. */ readonly workspace: readonly string[];
  /** Damaged-source production facade and its workbench composition. */ readonly panel: readonly string[];
  /** Original linked parameter editing and shared scrolling. */ readonly parameters: readonly string[];
  /** Original routes and all generic shell contribution slots. */ readonly workbench: readonly string[];
  /** Original numerical experiment plus registered job ownership. */ readonly experiments: readonly string[];
  /** Original source results, typed chart projection and scrolling. */ readonly results: readonly string[];
  /** Original workflow execution plus registered job ownership. */ readonly workflows: readonly string[];
}

/** Remove type-only wrappers around the actual build-time module array. */
function arrayExpression(node: ts.Expression): ts.Expression {
  while (
    ts.isAsExpression(node) ||
    ts.isSatisfiesExpression(node) ||
    ts.isParenthesizedExpression(node)
  )
    node = node.expression;
  return node;
}

/** Resolve one local compiled module's actual entry without executing its code. */
async function moduleEntry(root: string, directory: string): Promise<string> {
  const candidates = [`/src/modules/${directory}/index.ts`, `/src/modules/${directory}/index.tsx`];
  const present: string[] = [];
  for (const owner of candidates) {
    try {
      if ((await stat(resolve(root, `.${owner}`))).isFile()) present.push(owner);
    } catch (error: unknown) {
      if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") throw error;
    }
  }
  if (present.length !== 1 || present[0] === undefined)
    throw new Error(`One compiled source entry required for module ${directory}`);
  return present[0];
}

/** Derive local source entries from the same static imports and array used by the application. */
export async function compiledModuleBrowserOwners(root: string): Promise<readonly string[]> {
  const original = await readFile(resolve(root, "src/app/moduleRegistry.tsx"), "utf8");
  const source = ts.createSourceFile(
    "moduleRegistry.tsx",
    original,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const imports = new Map<string, string>();
  let entries: ts.ArrayLiteralExpression | undefined;
  for (const statement of source.statements) {
    if (
      ts.isImportDeclaration(statement) &&
      ts.isStringLiteral(statement.moduleSpecifier) &&
      statement.importClause !== undefined &&
      !statement.importClause.isTypeOnly
    ) {
      const directory = /^\.\.\/modules\/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\/index(?:\.tsx?)?)?$/.exec(
        statement.moduleSpecifier.text,
      )?.[1];
      if (directory === undefined) continue;
      const clause = statement.importClause;
      if (clause.name !== undefined) imports.set(clause.name.text, directory);
      if (clause.namedBindings !== undefined && ts.isNamedImports(clause.namedBindings))
        for (const binding of clause.namedBindings.elements)
          if (!binding.isTypeOnly) imports.set(binding.name.text, directory);
    }
    if (ts.isVariableStatement(statement))
      for (const declaration of statement.declarationList.declarations)
        if (
          ts.isIdentifier(declaration.name) &&
          declaration.name.text === "workbenchModules" &&
          declaration.initializer !== undefined
        ) {
          const value = arrayExpression(declaration.initializer);
          if (!ts.isArrayLiteralExpression(value))
            throw new Error("Static compiled module array required");
          entries = value;
        }
  }
  if (entries === undefined) throw new Error("Compiled workbench module registry required");
  const owners = new Set<string>();
  for (const entry of entries.elements) {
    if (!ts.isIdentifier(entry))
      throw new Error("Compiled registry entry must name a static module import");
    const directory = imports.get(entry.text);
    if (directory === undefined)
      throw new Error(`Local compiled source import required for ${entry.text}`);
    owners.add(await moduleEntry(root, directory));
  }
  return Object.freeze([...owners].sort());
}

/** Retain every original scenario owner and add its actual compiled composition. */
export async function readBrowserCohorts(root: string): Promise<BrowserSourceCohorts> {
  const compiled = await compiledModuleBrowserOwners(root);
  const runtime = [
    "/src/shared/storage/workspaceStore.ts",
    "/src/shared/storage/workspaceArchive.ts",
    "/src/shared/workspace/WorkspaceEditor.tsx",
    "/src/shared/workspace/useWorkspace.ts",
  ];
  const composition = [
    ...compiled,
    "/src/app/moduleRegistry.tsx",
    "/src/shared/contracts/module.ts",
    "/src/app/moduleViews/ModuleDocumentViews.tsx",
    "/src/shared/browser/ScrollRegion.tsx",
    "/src/shared/storage/workspaceDownload.ts",
  ];
  const shell = [
    "/src/app/Workbench.tsx",
    "/src/app/WorkbenchHelp.tsx",
    "/src/app/moduleViews/ModuleStageEditors.tsx",
    "/src/app/moduleViews/ModuleExplorerView.tsx",
    "/src/app/moduleViews/ModuleCommandSearch.tsx",
    "/src/app/moduleViews/ModuleJobDock.tsx",
    "/src/app/moduleViews/moduleJobPolicy.ts",
    "/src/shared/workspace/useWorkspaceDocuments.ts",
  ];
  const group = (...parts: readonly string[][]): readonly string[] =>
    Object.freeze([...new Set(parts.flat())].sort());
  const workspace = group(runtime, composition, ["/src/WorkspacePanel.tsx"]);
  const workbench = group(runtime, composition, shell, [
    "/src/QuantumStudioPanel.tsx",
    "/src/modules/catalogue/CapabilityCatalogue.tsx",
    "/src/app/WorkbenchInspector.tsx",
    "/src/app/RouteBoundary.tsx",
    "/src/app/routing.ts",
    "/src/app/useWorkbenchRoute.ts",
    "/src/modules/build/BuildView.tsx",
    "/src/modules/results/ResultsView.tsx",
    "/src/modules/compare/MethodComparison.tsx",
    "/src/modules/compare/useMethodComparison.ts",
    "/src/modules/compare/methodComparisonArtifact.ts",
    "/src/modules/atlas/UnavailableView.tsx",
  ]);
  return {
    compiled,
    workspace,
    workbench,
    panel: group(runtime, composition, shell, ["/src/QuantumStudioPanel.tsx"]),
    parameters: group(
      [...workspace],
      [
        "/src/shared/parameters/parameterDraft.ts",
        "/src/shared/parameters/parameterRevision.ts",
        "/src/modules/parameters/ParameterEditor.tsx",
        "/src/modules/parameters/ParameterWorkspace.tsx",
        "/src/modules/parameters/CouplingTable.tsx",
      ],
    ),
    experiments: group(runtime, composition, shell, [
      "/src/shared/experiments/kuramotoArtifacts.ts",
      "/src/shared/experiments/experimentArchive.ts",
      "/src/shared/experiments/experimentPlan.ts",
      "/src/shared/experiments/useExperimentRun.ts",
      "/src/modules/experiments/ExperimentRunner.tsx",
      "/src/modules/experiments/moduleJob.ts",
    ]),
    results: group(runtime, composition, shell, [
      "/src/modules/results/ResultsView.tsx",
      "/src/shared/results/resultModel.ts",
      "/src/shared/results/resultSources.ts",
      "/src/modules/results/resultExport.ts",
      "/src/modules/results/ResultInspector.tsx",
      "/src/modules/results/ResultLoader.tsx",
      "/src/modules/results/ResultChart.tsx",
      "/src/modules/results/ResultScrollArea.tsx",
      "/src/shared/results/resultChart.ts",
    ]),
    workflows: group(runtime, composition, shell, [
      "/src/shared/workflows/workflowModel.ts",
      "/src/shared/workflows/workflowSweep.ts",
      "/src/shared/workflows/workflowJournal.ts",
      "/src/shared/workflows/workflowArchive.ts",
      "/src/shared/workflows/workflowExecution.ts",
      "/src/shared/workflows/useWorkflowRun.ts",
      "/src/modules/workflows/WorkflowEditor.tsx",
      "/src/modules/workflows/WorkflowRunner.tsx",
      "/src/modules/workflows/moduleJob.ts",
    ]),
  };
}
