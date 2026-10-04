// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — complete raw result exports and truthful chart geometry

import { afterEach, expect, it, vi } from "vitest";
import { readJson } from "../../shared/contracts";
import { admitResultSnapshot } from "./resultModel";
import { downloadResultExport, resultCsv, resultPanelSvg, resultSvg } from "./resultExport";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const point = (x: number, value: number | null = x) => ({ coordinate: x, objectId: "a", columnId: null, x, y: null, value, status: value === null ? "missing" : "finite", interval: null });
const panel = { id: "native", title: "Native", kind: "series", coordinateLabel: "Time", coordinateUnit: "s", xLabel: "Time", xUnit: "s", yLabel: "", yUnit: "", valueLabel: "Amplitude", valueUnit: "V", valueDtype: "float64" };
const snapshot = (samples: unknown[] = [point(0, -0), point(0.125, 0.12345678901234568), point(2, 1)], panels: unknown[] = []) => admitResultSnapshot({ version: 1, title: "Actual raw source", caption: 'source,"caption" <script>', claimBoundary: "classical fixture & source declaration", sourceSha256: "a".repeat(64), partial: false, panels: [{ ...panel, samples }, ...panels] });
const selection = { coordinate: 0, objectId: "a" };
const documentOf = (svg: string) => new DOMParser().parseFromString(svg, "image/svg+xml");

it("exports exact source values, absent and estimator-specific intervals with quoted provenance", () => {
  const result = snapshot([
    point(0, -0), point(0.125, 0.12345678901234568), { ...point(2, 1), interval: { lower: 0.5, upper: 1.5, method: 'bootstrap,"percentile"', level: 0.95 } },
    { ...point(3, null), status: "nan" }, { ...point(4, 2), interval: { lower: 1, upper: 3, method: "source credible interval", level: null } },
  ]);
  const csv = resultCsv(result);
  expect(csv).toContain('"caption","source,""caption"" <script>"');
  expect(csv).toContain(',"float64","-0","finite","","","Not estimated",""');
  expect(csv).toContain('"0.12345678901234568"');
  expect(csv).toContain('"0.5","1.5","bootstrap,""percentile""","0.95"');
  expect(csv).toContain(',"float64","","nan","","","Not estimated",""');
  expect(csv.endsWith("\r\n")).toBe(true);
});

it("exports valid escaped SVG with complete original lossless metadata and evidence caption", () => {
  const result = snapshot();
  const svg = resultSvg(result, selection), doc = documentOf(svg);
  expect(doc.querySelector("parsererror")).toBeNull();
  expect(doc.querySelector("script")).toBeNull();
  expect(doc.querySelector("desc")!.textContent).toContain(result.caption);
  expect(doc.documentElement.textContent).toContain(result.sourceSha256);
  const metadata = admitResultSnapshot(readJson(doc.querySelector("metadata")!.textContent!));
  expect(metadata).toEqual(result);
  expect(Object.is(metadata.panels[0]!.samples[0]!.value, -0)).toBe(true);
  expect(metadata.panels[0]!.samples[1]!.value).toBe(0.12345678901234568);
});

it("keeps every original large-series sample in CSV/SVG while rendering bounded markers and gaps", () => {
  const samples = Array.from({ length: 2001 }, (_, index) => point(index));
  samples[1001] = point(1001, null);
  const result = snapshot(samples);
  const doc = documentOf(resultPanelSvg(result.panels[0]!, { coordinate: 1234, objectId: "a" }));
  expect(doc.querySelectorAll("circle").length).toBeLessThanOrEqual(1000);
  expect(doc.querySelector('[data-coordinate="1234"]')).toBeTruthy();
  expect(doc.querySelectorAll("polyline")).toHaveLength(2);
  expect(resultCsv(result).split("\r\n")).toHaveLength(2007);
  const exported = documentOf(resultSvg(result, selection));
  const metadata = admitResultSnapshot(readJson(exported.querySelector("metadata")!.textContent!));
  expect(metadata.panels[0]!.samples).toHaveLength(2001);
  expect(metadata.panels[0]!.samples.at(-1)!.value).toBe(2000);
});

it("renders declared histogram/spectrum/matrix forms at the exact selected source coordinate", () => {
  const result = snapshot([point(0, 0), point(2, 1)], [
    { ...panel, id: "hist", title: "Bins", kind: "histogram", samples: [{ ...point(2, 4), x: 0.5, interval: { lower: 3, upper: 5, method: "source estimator", level: null } }] },
    { ...panel, id: "spectrum", title: "Spectrum", kind: "spectrum", xLabel: "Frequency", xUnit: "Hz", samples: [{ ...point(2, 0.25), x: 12.5 }] },
    { ...panel, id: "matrix", title: "Matrix", kind: "matrix", yLabel: "Row", yUnit: "index", samples: [{ ...point(2, 0.5), x: 1, y: 2, columnId: "b" }] },
  ]);
  const selected = { coordinate: 2, objectId: "a" };
  expect(documentOf(resultPanelSvg(result.panels[1]!, selection)).documentElement.textContent).toContain("No source samples");
  expect(documentOf(resultPanelSvg(result.panels[1]!, selected)).querySelectorAll("line")).toHaveLength(2);
  expect(documentOf(resultPanelSvg(result.panels[2]!, selected)).querySelector("circle")!.getAttribute("data-coordinate")).toBe("2");
  expect(documentOf(resultPanelSvg(result.panels[3]!, selected)).querySelector("rect")).toBeTruthy();
});

it("keeps finite geometry for constant axes, extreme raw binary64 spans and all-unavailable samples", () => {
  for (const samples of [[point(0, 1)], [point(-1e308, -1e308), point(1e308, 1e308)], [point(0, 1e7), point(1, 1e8)], [point(0, null)]]) {
    const doc = documentOf(resultPanelSvg(snapshot(samples).panels[0]!, selection));
    for (const element of doc.querySelectorAll("[cx],[cy],[x],[y]")) for (const name of ["cx", "cy", "x", "y"]) if (element.hasAttribute(name)) expect(Number.isFinite(Number(element.getAttribute(name)))).toBe(true);
    expect(doc.querySelector("parsererror")).toBeNull();
  }
});

it("downloads local complete CSV/SVG and always releases object URLs including an actual link fault", () => {
  const create = vi.fn(() => "blob:owned-result"), revoke = vi.fn();
  vi.stubGlobal("URL", class extends URL { static override createObjectURL = create; static override revokeObjectURL = revoke; });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  const result = snapshot();
  downloadResultExport(result, "csv", selection); downloadResultExport(result, "svg", selection);
  expect(create).toHaveBeenCalledTimes(2); expect(revoke).toHaveBeenCalledTimes(2);
  expect((create.mock.calls[0] as unknown as [Blob])[0].type).toBe("text/csv;charset=utf-8");
  expect((create.mock.calls[1] as unknown as [Blob])[0].type).toBe("image/svg+xml;charset=utf-8");
  click.mockImplementationOnce(() => { throw new Error("actual local link fault"); });
  expect(() => downloadResultExport(result, "csv", selection)).toThrow("link fault");
  expect(revoke).toHaveBeenCalledTimes(3);
});


it("retains separately selected object markers in source histogram and matrix forms", () => {
  const result = snapshot(undefined, [
    { ...panel, id: "h", kind: "histogram", samples: [
      { ...point(2, -1), x: -0.5 }, { ...point(2, 3), objectId: "b", x: 0.5 },
    ] },
    { ...panel, id: "m", kind: "matrix", yLabel: "Row", yUnit: "index", samples: [
      { ...point(2, 0.5), x: 0, y: 0, columnId: "b", interval: { lower: 0, upper: 1, method: "actual declared interval", level: null } },
      { ...point(2, 1), objectId: "b", x: 1, y: 1, columnId: "a" },
    ] },
  ]);
  for (const p of result.panels.slice(1)) {
    const doc = documentOf(resultPanelSvg(p, { coordinate: 2, objectId: "a" }));
    expect(doc.querySelector('[data-object="a"]')!.getAttribute("stroke-width")).toBe("3");
    expect(doc.querySelector('[data-object="b"]')!.getAttribute("stroke-width")).toBe("1");
    expect(doc.querySelectorAll("[data-index]")).toHaveLength(2);
  }
});
