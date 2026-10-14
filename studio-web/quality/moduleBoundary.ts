// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — build-time workbench module import boundaries

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import ts from "typescript";

/** A source dependency that crosses a workbench module's owned boundary. */
export interface ModuleBoundaryFinding {
  /** Source file relative to the inspected source root. */
  readonly file: string;
  /** Literal import target, or the computed expression that cannot be admitted statically. */
  readonly target: string;
  /** Why the dependency violates the module or shared-owner contract. */
  readonly reason: string;
}

function sourceFiles(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return /\.tsx?$/u.test(entry.name) && !/\.test\.tsx?$/u.test(entry.name) ? [path] : [];
  });
}

function within(directory: string, path: string): boolean {
  const suffix = relative(directory, path);
  return (
    suffix === "" || (!suffix.startsWith(`..${sep}`) && suffix !== ".." && !isAbsolute(suffix))
  );
}

/**
 * Inspect actual TypeScript imports, re-exports and lazy imports in modules and shared owners.
 *
 * Modules may reach their own folder, shared owners and external dependencies.
 * Shared owners cannot import modules or the shell, preventing a shared barrel
 * from concealing a module-to-module dependency. Static data and binary resources
 * retain their source-owned paths; they are not executable module imports.
 *
 * @param sourceRoot - The application's source directory, or an isolated source tree for a refusal test.
 * @returns Every forbidden edge in deterministic file and syntax order.
 */
export function inspectModuleBoundaries(sourceRoot: string): readonly ModuleBoundaryFinding[] {
  const root = resolve(sourceRoot);
  const modules = join(root, "modules");
  const shared = join(root, "shared");
  const app = join(root, "app");
  const moduleFiles = sourceFiles(modules);
  if (!existsSync(shared) || moduleFiles.length === 0) {
    throw new Error("Module and shared source inventory must exist and contain module sources");
  }
  const findings: ModuleBoundaryFinding[] = [];
  for (const file of [...moduleFiles, ...sourceFiles(shared)].sort()) {
    const inModule = within(modules, file);
    const modulePath = relative(modules, file);
    const separator = modulePath.indexOf(sep);
    const owner = inModule
      ? join(modules, separator < 0 ? modulePath : modulePath.slice(0, separator))
      : shared;
    const source = ts.createSourceFile(
      file,
      readFileSync(file, "utf8"),
      ts.ScriptTarget.Latest,
      true,
    );
    const inspect = (expression: ts.Expression): void => {
      if (!ts.isStringLiteralLike(expression)) {
        findings.push({
          file: relative(root, file),
          target: expression.getText(source),
          reason: "Computed executable module import",
        });
        return;
      }
      const target = expression.text;
      if (!target.startsWith(".")) return;
      if (/\.(?:json|wasm|css|svg|png|jpg)(?:\?.*)?$/u.test(target)) return;
      const destination = resolve(file, "..", target);
      let reason: string | null = null;
      if (inModule && within(modules, destination) && !within(owner, destination))
        reason = "Module imports another module";
      else if (within(app, destination)) reason = "Module or shared owner imports the shell";
      else if (!inModule && within(modules, destination)) reason = "Shared owner imports a module";
      else if (inModule && !within(owner, destination) && !within(shared, destination))
        reason = "Module imports executable code outside its own and shared folders";
      if (reason !== null) findings.push({ file: relative(root, file), target, reason });
    };
    const visit = (node: ts.Node): void => {
      if (ts.isImportDeclaration(node)) inspect(node.moduleSpecifier);
      else if (ts.isExportDeclaration(node) && node.moduleSpecifier) inspect(node.moduleSpecifier);
      else if (
        ts.isImportEqualsDeclaration(node) &&
        ts.isExternalModuleReference(node.moduleReference) &&
        node.moduleReference.expression
      )
        inspect(node.moduleReference.expression);
      else if (
        ts.isCallExpression(node) &&
        node.expression.kind === ts.SyntaxKind.ImportKeyword &&
        node.arguments[0]
      )
        inspect(node.arguments[0]);
      ts.forEachChild(node, visit);
    };
    visit(source);
  }
  return Object.freeze(findings.map((finding) => Object.freeze(finding)));
}
