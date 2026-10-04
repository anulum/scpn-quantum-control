// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original result metadata admission cases

import { expect, it } from "vitest";
import { admitResultSnapshot, rawResultText, resultDisplayIndices, resultLimits } from "./resultModel";

const point = () => ({ coordinate: 0, objectId: "a", columnId: null, x: 0, y: null, value: -0, status: "finite", interval: null });
const panel = () => ({ id: "a", title: "Native", kind: "series", coordinateLabel: "Time", coordinateUnit: "s", xLabel: "Time", xUnit: "s", yLabel: "", yUnit: "", valueLabel: "Amplitude", valueUnit: "V", valueDtype: "float64", samples: [point()] });
const fixture = () => ({ version: 1, title: "Native", caption: "Actual fixture", claimBoundary: "Test source declaration", sourceSha256: "a".repeat(64), partial: false, panels: [panel()] });

it("preserves independent raw units, signed zero, sparse coordinate order and immutable source copies", () => {
  const raw = fixture();
  raw.panels[0]!.samples.push({ ...point(), coordinate: 2, x: 2, value: 0.12345678901234568 });
  const result = admitResultSnapshot(raw);
  raw.panels[0]!.samples[1]!.value = 123;
  expect(result.panels[0]!.samples[1]!.value).toBe(0.12345678901234568);
  expect(Object.is(result.panels[0]!.samples[0]!.value, -0)).toBe(true);
  expect(Object.isFrozen(result.panels[0]!.samples[0])).toBe(true);
  expect(result.panels[0]!.coordinateUnit).toBe("s");
  expect(rawResultText(-0)).toBe("-0");
  expect(rawResultText(0.12345678901234568)).toBe("0.12345678901234568");
});

it.each([null, 1, [], "source"])("refuses nonobject producer metadata %s", value => { expect(() => admitResultSnapshot(value)).toThrow(); });
it.each([
  { version: 2 }, { sourceSha256: "A".repeat(64) }, { sourceSha256: "a" }, { partial: "no" },
  { title: "" }, { title: "x".repeat(2049) }, { title: "\ud800" }, { title: "\udc00" },
  { panels: null }, { panels: [] }, { panels: Array(17).fill(null) }, { unexpected: true },
])("refuses unsupported identity, text and source budgets before source mutation %s", override => {
  const raw = fixture(), before = structuredClone(raw);
  expect(() => admitResultSnapshot({ ...raw, ...override })).toThrow();
  expect(raw).toEqual(before);
});

it.each([
  { kind: "inferred" }, { samples: [] }, { samples: null }, { samples: Array(resultLimits.samples + 1).fill(null) },
  { valueDtype: "float32" }, { coordinateUnit: "" }, { missing: 1 },
])("refuses malformed panel and unqualified representations %s", override => {
  expect(() => admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), ...override }] })).toThrow();
});

it.each([
  { value: NaN }, { value: Infinity }, { coordinate: NaN }, { x: Infinity }, { status: "unknown" },
  { status: "missing", value: 0 }, { y: 1 }, { columnId: "other" }, { objectId: "" },
  { interval: undefined }, { interval: [] }, { interval: { lower: 1, upper: 2, method: "source", level: null } },
  { interval: { lower: -1, upper: 1, method: "", level: null } },
  { interval: { lower: -1, upper: 1, method: "source", level: 0 } },
  { interval: { lower: -1, upper: 1, method: "source", level: 1 } },
  { status: "nan", value: null, interval: { lower: -1, upper: 1, method: "source", level: 0.95 } },
])("refuses invalid raw samples and invented uncertainty %s", override => {
  expect(() => admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), samples: [{ ...point(), ...override }] }] })).toThrow();
});

it("admits every declared display form and source-specific intervals without a model inference", () => {
  const intervals = [{ lower: -1, upper: 1, method: "source bootstrap percentile", level: 0.95 }, { lower: -1, upper: 1, method: "source credible interval", level: null }];
  const panels = [
    { ...panel(), samples: [{ ...point(), interval: intervals[0] }] },
    { ...panel(), id: "hist", kind: "histogram", samples: [{ ...point(), value: 0, interval: intervals[1] }] },
    { ...panel(), id: "spectrum", kind: "spectrum", xLabel: "Frequency", xUnit: "Hz" },
    { ...panel(), id: "matrix", kind: "matrix", yLabel: "Row", yUnit: "index", samples: [{ ...point(), y: 1, columnId: "b" }] },
  ];
  const result = admitResultSnapshot({ ...fixture(), partial: true, panels });
  expect(result.panels.map(item => item.kind)).toEqual(["series", "histogram", "spectrum", "matrix"]);
  expect(result.panels[0]!.samples[0]!.interval).toEqual(intervals[0]);
  expect(result.panels[1]!.samples[0]!.interval!.level).toBeNull();
  expect(result.partial).toBe(true);
});

it.each([{ y: null, columnId: "b" }, { y: 1, columnId: null }, { y: 1, columnId: "" }])("requires complete source matrix identities %s", override => {
  expect(() => admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), kind: "matrix", samples: [{ ...point(), ...override }] }] })).toThrow();
});

it("refuses duplicate identities, reordered series, mismatched linkage and exhausted combined budgets", () => {
  const raw = fixture();
  expect(() => admitResultSnapshot({ ...raw, panels: [panel(), panel()] })).toThrow("Unique");
  expect(() => admitResultSnapshot({ ...raw, panels: [{ ...panel(), samples: [point(), point()] }] })).toThrow("Duplicate");
  expect(() => admitResultSnapshot({ ...raw, panels: [{ ...panel(), samples: [point(), { ...point(), coordinate: 1, x: -1 }] }] })).toThrow("increasing");
  expect(() => admitResultSnapshot({ ...raw, panels: [panel(), { ...panel(), id: "b", coordinateUnit: "ms" }] })).toThrow("identical");
  const large = { ...panel(), samples: Array.from({ length: resultLimits.samples }, (_, index) => ({ ...point(), coordinate: index, x: index })) };
  expect(() => admitResultSnapshot({ ...raw, panels: [large, { ...panel(), id: "b" }] })).toThrow("Bounded");
});

it("retains explicit missing/NaN/refusal states and only admits exact int64 display values", () => {
  const unavailable = ["missing", "nan", "refused"].map((status, index) => ({ ...point(), coordinate: index, x: index, value: null, status }));
  const result = admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), samples: unavailable }] });
  expect(result.panels[0]!.samples.map(item => item.status)).toEqual(["missing", "nan", "refused"]);
  expect(admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), valueDtype: "int64", samples: [{ ...point(), value: 32 }] }] }).panels[0]!.samples[0]!.value).toBe(32);
  for (const value of [-0, 0.5, Number.MAX_SAFE_INTEGER + 1]) expect(() => admitResultSnapshot({ ...fixture(), panels: [{ ...panel(), valueDtype: "int64", samples: [{ ...point(), value }] }] })).toThrow("exact");
});

it("does not invoke untrusted metadata accessors", () => {
  const raw = fixture(); let calls = 0;
  Object.defineProperty(raw, "caption", { get() { ++calls; throw new Error("untrusted"); } });
  expect(() => admitResultSnapshot(raw)).toThrow(); expect(calls).toBe(0);
});

it("decimates only display indices, retaining endpoints and exact selected sample under bounded budgets", () => {
  expect(resultDisplayIndices(0)).toEqual([]);
  expect(resultDisplayIndices(3)).toEqual([0, 1, 2]);
  const indices = resultDisplayIndices(65536, 12345);
  expect(indices.length).toBeLessThanOrEqual(1000);
  expect(indices[0]).toBe(0); expect(indices.at(-1)).toBe(65535); expect(indices).toContain(12345);
  expect(resultDisplayIndices(2000)).toHaveLength(999);
  for (const length of [-1, 0.5, 65537]) expect(() => resultDisplayIndices(length)).toThrow();
  for (const selected of [-1, 0.5, 3]) expect(() => resultDisplayIndices(3, selected)).toThrow();
});


it.each(["\u0000", "\u000b", "\u001f"])("refuses source text that cannot be represented in XML without corruption", value => {
  expect(() => admitResultSnapshot({ ...fixture(), caption: value })).toThrow();
});
