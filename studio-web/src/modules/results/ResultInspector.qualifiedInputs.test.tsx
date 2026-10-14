// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original producer results cannot be replaced by an empty caller cache

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { requirePresent } from "../../../test-support/requirePresent";
import { inspectAnalyseProducer } from "../../shared/results/resultSources";
import { ResultInspector } from "./ResultInspector";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  cleanup();
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(["panels", "samples"])(
  "refuses a caller cache missing %s without presenting new values or modifying the original sealed export",
  async (missing) => {
    const path = requirePresent(process.env["STUDIO_ANALYSE_RESULT_PATH"]);
    const bytes = readFileSync(path, "utf8");
    const result = await inspectAnalyseProducer(bytes);
    const panel = requirePresent(result.panels[0]);
    const supplied = {
      ...result,
      panels: missing === "panels" ? [] : [{ ...panel, samples: [] }],
    };
    expect(() => render(<ResultInspector result={supplied} />)).toThrow(
      "Admitted nonempty result required",
    );
    expect(readFileSync(path, "utf8")).toBe(bytes);
    expect(panel.samples.length).toBeGreaterThan(0);
    expect(result.panels.length).toBeGreaterThan(0);
  },
);
