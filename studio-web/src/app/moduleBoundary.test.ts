// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the import boundaries between workbench features

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { expect, it } from "vitest";
import { inspectModuleBoundaries } from "../../quality/moduleBoundary";

it("admits the actual module and shared source tree without recorded cross-module exceptions", () => {
  expect(inspectModuleBoundaries(resolve("src"))).toEqual([]);
});

it("refuses static, type, re-export and lazy cross-module edges and shared-barrel bypasses", () => {
  const root = mkdtempSync(join(tmpdir(), "studio-module-boundary-"));
  const sources = {
    "modules/first/source.ts": `
      import type { Document } from '../second/source';
      export { read } from '../second/source';
      const lazy = import('../second/source');
      const computed = import(target);
      import { shell } from '../../app/shell';
      import { core } from '../../panel/core';
    `,
    "modules/second/source.ts": "export const read = 1; export type Document = string;",
    "shared/barrel.ts":
      "export { read } from '../modules/second/source'; import { shell } from '../app/shell';",
    "modules/first/local.ts":
      "import './source'; import { value } from '../../shared/value'; import React from 'react'; import data from '../../../manifest.json?raw';",
    "shared/value.ts": "export const value = 1;",
  };
  try {
    for (const [path, source] of Object.entries(sources)) {
      const file = join(root, path);
      mkdirSync(dirname(file), { recursive: true });
      writeFileSync(file, source);
    }
    const findings = inspectModuleBoundaries(root);
    expect(findings.map(({ reason }) => reason)).toEqual([
      "Module imports another module",
      "Module imports another module",
      "Module imports another module",
      "Computed executable module import",
      "Module or shared owner imports the shell",
      "Module imports executable code outside its own and shared folders",
      "Shared owner imports a module",
      "Module or shared owner imports the shell",
    ]);
    expect(findings.some(({ file }) => file.endsWith("local.ts"))).toBe(false);
    expect(findings.map(({ target }) => target)).toContain("target");
  } finally {
    rmSync(root, { recursive: true });
  }
});
