// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — actual compiled source inventory rejects unresolved ownership

// @vitest-environment node
import { mkdir, mkdtemp, rm, symlink, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { expect, it } from "vitest";
import { compiledModuleBrowserOwners } from "./browserCohorts";

/** Materialize the caller's registry source for the real filesystem and TypeScript reader. */
async function registrySource(source: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "studio-registry-refusal-"));
  const path = join(root, "src/app/moduleRegistry.tsx");
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, source);
  return root;
}

it.each([
  [
    "missing array",
    "export const anotherList = [];",
    "Compiled workbench module registry required",
  ],
  [
    "uninitialized array",
    "export let workbenchModules;",
    "Compiled workbench module registry required",
  ],
  [
    "binding pattern",
    "export const { workbenchModules } = source;",
    "Compiled workbench module registry required",
  ],
  [
    "computed array",
    "export const workbenchModules = buildModules();",
    "Static compiled module array required",
  ],
  [
    "computed entry",
    "export const workbenchModules = [buildModule()];",
    "Compiled registry entry must name a static module import",
  ],
  [
    "unimported entry",
    "export const workbenchModules = [absentModule];",
    "Local compiled source import required for absentModule",
  ],
  [
    "external entry",
    "import { example } from 'external-package'; export const workbenchModules = [example];",
    "Local compiled source import required for example",
  ],
  [
    "type-only import",
    "import type { example } from '../modules/example'; export const workbenchModules = [example];",
    "Local compiled source import required for example",
  ],
  [
    "type-only binding",
    "import { type example } from '../modules/example'; export const workbenchModules = [example];",
    "Local compiled source import required for example",
  ],
  [
    "namespace import",
    "import * as example from '../modules/example'; export const workbenchModules = [example];",
    "Local compiled source import required for example",
  ],
])("refuses %s without evaluating the registry source", async (_name, source, reason) => {
  const root = await registrySource(source);
  try {
    await expect(compiledModuleBrowserOwners(root)).rejects.toThrow(reason);
  } finally {
    await rm(root, { recursive: true });
  }
});

it("retains the native filesystem error when a compiled entry cannot resolve its real link", async () => {
  const root = await registrySource(
    "import example from '../modules/example'; export const workbenchModules = [example];",
  );
  const entry = join(root, "src/modules/example/index.ts");
  await mkdir(dirname(entry), { recursive: true });
  await symlink(entry, entry, "junction");
  try {
    await expect(compiledModuleBrowserOwners(root)).rejects.toMatchObject({
      code: "ELOOP",
      path: entry,
    });
  } finally {
    await unlink(entry);
    await rm(root, { recursive: true });
  }
});

it.each(["missing", "directory", "ambiguous"])(
  "refuses a %s module entry instead of inventing a production owner",
  async (kind) => {
    const root = await registrySource(
      "import example from '../modules/example'; export const workbenchModules = [example];",
    );
    try {
      const directory = join(root, "src/modules/example");
      await mkdir(directory, { recursive: true });
      if (kind === "directory") await mkdir(join(directory, "index.ts"));
      if (kind === "ambiguous") {
        await writeFile(join(directory, "index.ts"), "export default {};");
        await writeFile(join(directory, "index.tsx"), "export default {};");
      }
      await expect(compiledModuleBrowserOwners(root)).rejects.toThrow(
        "One compiled source entry required for module example",
      );
    } finally {
      await rm(root, { recursive: true });
    }
  },
);

it("reads named and default static imports under type wrappers without executing their entry files", async () => {
  const root = await registrySource(`
    import defaultModule from '../modules/default-example/index.ts';
    import { namedModule as named } from '../modules/named-example/index.tsx';
    export const workbenchModules = (([defaultModule, named] as const) satisfies readonly unknown[]);
  `);
  try {
    const entries: readonly (readonly [string, string])[] = [
      ["default-example", "ts"],
      ["named-example", "tsx"],
    ];
    for (const [name, suffix] of entries) {
      const directory = join(root, "src/modules", name);
      await mkdir(directory, { recursive: true });
      await writeFile(
        join(directory, `index.${suffix}`),
        "throw new Error('entry execution is forbidden during source discovery');",
      );
    }
    expect(await compiledModuleBrowserOwners(root)).toEqual([
      "/src/modules/default-example/index.ts",
      "/src/modules/named-example/index.tsx",
    ]);
  } finally {
    await rm(root, { recursive: true });
  }
});
