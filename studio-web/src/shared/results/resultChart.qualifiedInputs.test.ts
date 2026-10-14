// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original chart projections refuse a caller's incomplete metadata

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, it } from "vitest";
import { requirePresent } from "../../../test-support/requirePresent";
import { inspectAnalyseProducer } from "./resultSources";
import { projectResultChart } from "./resultChart";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(["matrix coordinate", "finite value"])(
  "refuses a caller's missing %s without guessing geometry or changing the original producer export",
  async (missing) => {
    const filename = requirePresent(process.env["STUDIO_ANALYSE_RESULT_PATH"]);
    const bytes = readFileSync(filename, "utf8");
    const result = await inspectAnalyseProducer(bytes);
    const original = requirePresent(result.panels[0]);
    const first = requirePresent(original.samples[0]);
    const supplied =
      missing === "matrix coordinate"
        ? { ...original, kind: "matrix" as const }
        : { ...original, samples: [{ ...first, value: null }] };
    expect(() =>
      projectResultChart(supplied, {
        coordinate: first.coordinate,
        objectId: first.objectId,
      }),
    ).toThrow(
      missing === "matrix coordinate"
        ? "Original matrix coordinate required"
        : "Original finite result value required",
    );
    expect(readFileSync(filename, "utf8")).toBe(bytes);
    expect(first.value).not.toBeNull();
    expect(original.kind).toBe("series");
  },
);
