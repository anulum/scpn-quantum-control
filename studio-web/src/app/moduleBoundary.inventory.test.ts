// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — real source inventory and import-equals isolation

// @vitest-environment node
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { inspectModuleBoundaries } from "../../quality/moduleBoundary";

it.each(["empty", "missing shared", "missing modules"])(
  "refuses a %s inventory instead of qualifying an uninspected module tree",
  (kind) => {
    const root = mkdtempSync(join(tmpdir(), "studio-boundary-inventory-"));
    try {
      if (kind !== "missing shared") mkdirSync(join(root, "shared"));
      if (kind !== "missing modules") {
        mkdirSync(join(root, "modules/example"), { recursive: true });
        if (kind === "missing shared")
          writeFileSync(join(root, "modules/example/index.ts"), "export {};");
      }
      expect(() => inspectModuleBoundaries(root)).toThrow(
        "Module and shared source inventory must exist and contain module sources",
      );
    } finally {
      rmSync(root, { recursive: true });
    }
  },
);

it("inspects root entries, import-equals and local exports while excluding test and data files", () => {
  const root = mkdtempSync(join(tmpdir(), "studio-boundary-import-equals-"));
  try {
    mkdirSync(join(root, "modules/second"), { recursive: true });
    mkdirSync(join(root, "shared"));
    writeFileSync(
      join(root, "modules/root.ts"),
      "import other = require('./second/source'); export const local = 1;",
    );
    writeFileSync(
      join(root, "modules/second/source.ts"),
      "export { local }; const local = 1; import localModule = local.Namespace;",
    );
    writeFileSync(
      join(root, "shared/value.ts"),
      "export { local }; const local = 1; const unavailable = import();",
    );
    writeFileSync(join(root, "modules/second/ignored.test.ts"), "import '../../app/shell';");
    writeFileSync(join(root, "modules/second/data.json"), "{}");
    expect(inspectModuleBoundaries(root)).toEqual([
      {
        file: join("modules", "root.ts"),
        target: "./second/source",
        reason: "Module imports another module",
      },
    ]);
  } finally {
    rmSync(root, { recursive: true });
  }
});
