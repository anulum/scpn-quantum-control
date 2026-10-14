// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — compiled registry source ownership

// @vitest-environment node
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { expect, it } from "vitest";
import { workbenchRegistry } from "../src/app/moduleRegistry";
import { compiledModuleBrowserOwners, readBrowserCohorts } from "./browserCohorts";

it("discovers the actual compiled entries and follows a one-folder one-registry addition and removal", async () => {
  const root = process.cwd();
  const actual = await compiledModuleBrowserOwners(root);
  const exportedModules = new Set<unknown>();
  for (const owner of actual) {
    const directory = owner.split("/")[3];
    if (directory === undefined) throw new Error("compiled owner has no module directory");
    const namespace: Record<string, unknown> = owner.endsWith("/index.tsx")
      ? await import(`../src/modules/${directory}/index.tsx`)
      : await import(`../src/modules/${directory}/index.ts`);
    for (const value of Object.values(namespace))
      if (workbenchRegistry.modules.some((module) => module === value)) exportedModules.add(value);
  }
  expect(exportedModules).toEqual(new Set(workbenchRegistry.modules));
  const directory = await mkdtemp(join(tmpdir(), "studio-compiled-browser-owners-"));
  try {
    const registry = await readFile(join(root, "src/app/moduleRegistry.tsx"), "utf8");
    const target = join(directory, "src/app/moduleRegistry.tsx");
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, registry);
    for (const owner of actual) {
      const path = join(directory, owner.slice(1));
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, await readFile(join(root, owner.slice(1)), "utf8"));
    }
    expect(await compiledModuleBrowserOwners(directory)).toEqual(actual);
    const module = join(directory, "src/modules/source-notes/index.tsx");
    await mkdir(dirname(module), { recursive: true });
    await writeFile(
      module,
      await readFile(join(root, "test-fixtures/source-notes/index.tsx"), "utf8"),
    );
    await writeFile(
      target,
      'import { sourceNotesModule } from "../modules/source-notes";\n' +
        registry.replace("  workspaceModule,\n", "  workspaceModule,\n  sourceNotesModule,\n"),
    );
    const added = await readBrowserCohorts(directory);
    expect(added.compiled).toEqual([...actual, "/src/modules/source-notes/index.tsx"].sort());
    for (const owners of Object.values(added))
      expect(owners).toContain("/src/modules/source-notes/index.tsx");
    expect(await readdir(join(directory, "src/app"))).toEqual(["moduleRegistry.tsx"]);
    for (const owner of actual)
      expect(await readFile(join(directory, owner.slice(1)), "utf8")).toBe(
        await readFile(join(root, owner.slice(1)), "utf8"),
      );
    await writeFile(target, registry);
    await rm(dirname(module), { recursive: true });
    expect(await compiledModuleBrowserOwners(directory)).toEqual(actual);
  } finally {
    await rm(directory, { recursive: true });
  }
});
