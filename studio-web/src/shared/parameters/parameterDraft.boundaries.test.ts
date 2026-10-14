// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public parameter formatting and cached-state refusal boundaries

import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, expect, it } from "vitest";
import { stationaryPlan } from "../../../test-fixtures/stationary-experiment-plan";
import { requirePresent } from "../../../test-support/requirePresent";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { writeJson } from "../contracts";
import { parameterSourceFromArchive } from "./parameterRevision";
import {
  createParameterDraft,
  parameterDraftReducer,
  parameterElementText,
  parameterField,
  validateParameterSnapshot,
} from "./parameterDraft";

const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
beforeEach(() => {
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: webcrypto });
});
afterEach(() => {
  if (originalCrypto) Object.defineProperty(globalThis, "crypto", originalCrypto);
});

it.each(["specification", "value", "unit", "trainable mask", "unknown key"])(
  "refuses a field cache missing %s before presenting the genuine source's editor row",
  async (missing) => {
    const plan = await stationaryPlan();
    const source = requirePresent(
      await parameterSourceFromArchive(plan.sourceJson, localExperimentCodecs),
    );
    const draft = createParameterDraft(source);
    const before = writeJson(source.revision);
    const supplied = {
      ...draft,
      source: missing === "specification" ? { ...draft.source, specs: [] } : draft.source,
      snapshot: {
        ...draft.snapshot,
        ...(missing === "value" ? { parameters: {} } : {}),
        ...(missing === "unit" ? { units: {} } : {}),
        ...(missing === "trainable mask" ? { trainableMasks: {} } : {}),
      },
    };
    expect(() => parameterField(supplied, missing === "unknown key" ? "absent" : "theta0")).toThrow(
      "Parameter source metadata unavailable",
    );
    const original = parameterField(draft, "theta0");
    expect(original.value).toBe(draft.snapshot.parameters["theta0"]);
    expect(original.unit).toBe(draft.snapshot.units["theta0"]);
    expect(original.trainable).toBe(draft.snapshot.trainableMasks["theta0"]);
    expect(writeJson(source.revision)).toBe(before);
  },
);

it("refuses an undefined cached unit before an external save can create a child revision", async () => {
  const plan = await stationaryPlan();
  const source = requirePresent(
    await parameterSourceFromArchive(plan.sourceJson, localExperimentCodecs),
  );
  const draft = createParameterDraft(source);
  const units = { ...draft.snapshot.units };
  Reflect.set(units, "theta0", undefined);
  const before = writeJson(source.revision);
  expect(() => validateParameterSnapshot(source, { ...draft.snapshot, units })).toThrow(
    "$.units.theta0: unsupported value type",
  );
  expect(writeJson(source.revision)).toBe(before);
  expect(draft.snapshot.units["theta0"]).toBeDefined();
});

it.each(["not-binary64", "7ff0000000000000", "7ff8000000000000"])(
  "refuses %s at the public binary64 display boundary instead of presenting a finite scalar",
  (value) => {
    const supplied = { dtype: "float64", shape: [1n], values: [value] };
    expect(() => parameterElementText(supplied, 0)).toThrow(
      "Canonical finite binary64 element required",
    );
    expect(supplied.values).toEqual([value]);
  },
);

it.each(["unit", "trainable mask"])(
  "retains genuine original values and source when a caller's cached %s is absent",
  async (missing) => {
    const plan = await stationaryPlan();
    const source = requirePresent(
      await parameterSourceFromArchive(plan.sourceJson, localExperimentCodecs),
    );
    const draft = createParameterDraft(source);
    const sourceBefore = writeJson(source.revision);
    const supplied = {
      ...draft,
      snapshot: {
        ...draft.snapshot,
        ...(missing === "unit" ? { units: {} } : { trainableMasks: {} }),
      },
    };
    const original = requirePresent(draft.snapshot.parameters["theta0"]);
    const result = parameterDraftReducer(
      supplied,
      missing === "unit"
        ? {
            type: "value",
            key: "theta0",
            index: 0,
            text: "1",
            unit: requirePresent(draft.snapshot.units["theta0"]),
          }
        : { type: "mask", key: "theta0", index: 0, enabled: true },
    );
    expect(result.refusal).toBe(
      missing === "unit"
        ? "Parameter source unit unavailable"
        : "Parameter trainable mask unavailable",
    );
    expect(result.snapshot.parameters["theta0"]).toEqual(original);
    expect(result.source).toBe(draft.source);
    expect(result.past).toEqual(draft.past);
    expect(writeJson(source.revision)).toBe(sourceBefore);
  },
);
