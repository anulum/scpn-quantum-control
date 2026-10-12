// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the import boundaries between workbench features

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { expect, it } from "vitest";

const root = resolve("src");
const features = join(root, "features");

/**
 * Feature-to-feature imports that exist today, as "importer -> imported".
 *
 * The list only falls: a new edge fails the test, and an edge that is gone
 * must be removed here, so the record stays a measurement and not an allowance.
 */
const recorded = [
  "compare -> experiments",
  "compare -> results",
  "compiler -> programs",
  "experiments -> workspace",
  "results -> experiments",
  "workflows -> experiments",
  "workflows -> parameters",
  "workflows -> results",
  "workflows -> workspace",
  "workspace -> parameters",
  "workspace -> settings",
];

function sources(directory: string): string[] {
  return readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) return sources(path);
    return /\.tsx?$/u.test(name) && !/\.test\.tsx?$/u.test(name) ? [path] : [];
  });
}

function imports(path: string): string[] {
  const text = readFileSync(path, "utf8");
  const found = [...text.matchAll(/(?:from|import)\s*\(?\s*"(\.[^"]*)"/gu)];
  return found.map((match) => resolve(path, "..", match[1] as string));
}

function feature(path: string): string | null {
  const parts = relative(features, path).split(sep);
  return parts[0] === ".." || parts.length < 2 ? null : (parts[0] as string);
}

const files = sources(features);

it("reads every feature source and finds their relative imports", () => {
  expect(files.length).toBeGreaterThan(30);
  expect(files.flatMap(imports).length).toBeGreaterThan(100);
  expect(feature(join(features, "results", "resultModel.ts"))).toBe("results");
  expect(feature(join(root, "app", "routing.ts"))).toBeNull();
  expect(feature(join(features, "loose.ts"))).toBeNull();
});

it("lets no feature import the shell", () => {
  const offenders = files.filter((file) =>
    imports(file).some((target) => target.startsWith(join(root, "app") + sep)),
  );
  expect(offenders.map((file) => relative(root, file))).toEqual([]);
});

it("holds feature-to-feature imports at the recorded edges", () => {
  const edges = new Set<string>();
  for (const file of files) {
    const importer = feature(file);
    for (const target of imports(file)) {
      const imported = feature(target);
      if (importer !== null && imported !== null && importer !== imported)
        edges.add(`${importer} -> ${imported}`);
    }
  }
  expect([...edges].sort()).toEqual(recorded);
});
