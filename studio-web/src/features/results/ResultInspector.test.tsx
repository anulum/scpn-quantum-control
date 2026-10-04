// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — independent linked result value cases

import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ResultInspector } from "./ResultInspector";
import { admitResultSnapshot } from "./resultModel";
import { resultCsv } from "./resultExport";

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
const digest = "0123456789abcdef".repeat(4);
const sample = (coordinate: number, value: number | null, status = "finite", objectId = "R") => ({ coordinate, objectId, columnId: null, x: coordinate, y: null, value, status, interval: null });
const panel = { id: "order", title: "Native order", kind: "series", coordinateLabel: "Time", coordinateUnit: "model-time", xLabel: "Time", xUnit: "model-time", yLabel: "", yUnit: "", valueLabel: "R", valueUnit: "1", valueDtype: "float64" };
const snapshot = (samples = [sample(0, -0), sample(0.125, 0.12345678901234568), sample(2, 1)], panels: unknown[] = []) => admitResultSnapshot({ version: 1, title: "Original source", caption: "Exact independent source fixture", claimBoundary: "classical fixture only", sourceSha256: digest, partial: false, panels: [{ ...panel, samples }, ...panels] });

it("test_result_value_inspector_01 retains actual nonuniform coordinates in chart and raw table", () => {
  render(<ResultInspector result={snapshot()} />);
  const chart = screen.getByRole("img", { name: "Native order chart" });
  const points = chart.querySelectorAll("[data-coordinate]");
  expect(Array.from(points, point => point.getAttribute("data-coordinate"))).toEqual(["0", "0.125", "2"]);
  expect(Number(points[1]!.getAttribute("cx")) - Number(points[0]!.getAttribute("cx"))).toBeLessThan(Number(points[2]!.getAttribute("cx")) - Number(points[1]!.getAttribute("cx")));
  expect(screen.getByRole("table", { name: "Native order raw values" }).textContent).toContain("0.12345678901234568");
  expect(screen.getByText("-0")).toBeTruthy();
});

it("test_result_value_inspector_02 exposes absent intervals as not estimated", () => {
  render(<ResultInspector result={snapshot()} />);
  expect(screen.getAllByText("Not estimated")).toHaveLength(3);
  expect(document.body.textContent).not.toContain("±0");
});

it("test_result_value_inspector_03 exports original values despite rounded chart labels", () => {
  const result = snapshot();
  render(<ResultInspector result={result} />);
  expect(Array.from(screen.getByRole("img", { name: "Native order chart" }).querySelectorAll("text"), node => node.textContent)).toContain("0.123");
  const csv = resultCsv(result);
  expect(csv).toContain("0.12345678901234568");
  expect(csv).toContain(',"-0",');
  expect(csv).toContain(digest);
  expect(csv).toContain("Exact independent source fixture");
  expect(screen.getByRole("button", { name: "Export raw CSV" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Export SVG" })).toBeTruthy();
});

it("test_result_value_inspector_04 keeps missing, NaN and refused source samples explicit", () => {
  render(<ResultInspector result={snapshot([sample(0, 0), sample(0.125, null, "missing"), sample(2, null, "nan"), sample(3, null, "refused"), sample(4, 1)])} />);
  const table = screen.getByRole("table", { name: "Native order raw values" });
  for (const state of ["Missing", "NaN", "Refused"]) expect(within(table).getAllByText(state)).toHaveLength(2);
  expect(screen.getByRole("img", { name: "Native order chart" }).querySelectorAll("polyline")).toHaveLength(2);
});

it("test_result_value_inspector_05 links selected coordinate to matrix, histogram and raw rows", () => {
  const result = snapshot(undefined, [
    { ...panel, id: "matrix", title: "Source matrix", kind: "matrix", xLabel: "Column", xUnit: "index", yLabel: "Row", yUnit: "index", samples: [{ ...sample(2, 0.25), x: 0, y: 0, columnId: "R" }] },
    { ...panel, id: "hist", title: "Source histogram", kind: "histogram", xLabel: "Bin", xUnit: "rad", samples: [{ ...sample(2, 4), x: 0.5 }] },
  ]);
  render(<ResultInspector result={result} />);
  fireEvent.click(screen.getByRole("button", { name: "Select Native order sample 2 R" }));
  expect(screen.getByRole("status", { name: "Result selection" }).textContent).toContain("2");
  expect(screen.getByRole("img", { name: "Source matrix chart" }).querySelector('[data-selected="true"]')).toBeTruthy();
  expect(screen.getByRole("img", { name: "Source histogram chart" }).querySelector('[data-selected="true"]')).toBeTruthy();
  expect(screen.getByRole("table", { name: "Source matrix raw values" }).querySelector('[aria-selected="true"]')).toBeTruthy();
});


it("pages complete raw series independently of bounded chart markers and linked selection", () => {
  const samples = Array.from({ length: 41 }, (_, index) => sample(index, index));
  const series = snapshot(samples, [
    { ...panel, id: "other", title: "Other object", kind: "series", samples: samples.map(value => ({ ...value, objectId: "S" })) },
    { ...panel, id: "matrix", title: "Sparse matrix", kind: "matrix", yLabel: "Row", yUnit: "index", samples: [{ ...sample(2, 1), y: 0, x: 0, columnId: "R" }] },
  ]);
  const view = render(<ResultInspector result={series} />);
  expect(screen.getByRole("button", { name: "Previous Native order rows" })).toMatchObject({ disabled: true });
  fireEvent.click(screen.getByRole("button", { name: "Next Native order rows" }));
  expect(screen.getByRole("table", { name: "Native order raw values" }).querySelectorAll("tbody tr")).toHaveLength(20);
  expect(screen.getByRole("button", { name: "Select Native order sample 20 R" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Previous Native order rows" }));
  expect(screen.getByRole("button", { name: "Select Native order sample 0 R" })).toBeTruthy();
  const chart = screen.getByRole("img", { name: "Native order chart" });
  const point = chart.querySelector('[data-coordinate="40"]')!;
  fireEvent.click(point.querySelector("title")!);
  expect(screen.getByRole("status", { name: "Result selection" }).textContent).toContain("40");
  expect(screen.getByRole("button", { name: "Next Native order rows" })).toMatchObject({ disabled: true });
  expect(screen.getByRole("button", { name: "Select Other object sample 40 S" })).toBeTruthy();
  expect(screen.getByRole("img", { name: "Sparse matrix chart" }).textContent).toContain("No source samples");
  expect(screen.getByRole("button", { name: "Select Sparse matrix sample 2 R R" })).toBeTruthy();
  expect(series.panels[0]!.samples).toHaveLength(41);
  const next = admitResultSnapshot({ ...series, sourceSha256: "b".repeat(64), panels: [{ ...series.panels[0]!, samples: Array.from({ length: 41 }, (_, index) => sample(0.125 + index, -0)) }] });
  view.rerender(<ResultInspector result={next} />);
  expect(screen.getByRole("status", { name: "Result selection" }).textContent).toContain("0.125");
  expect(screen.getByRole("button", { name: "Next Native order rows" })).toMatchObject({ disabled: false });
  expect(screen.getByRole("button", { name: "Previous Native order rows" })).toMatchObject({ disabled: true });
  expect(screen.getAllByText("-0")).toHaveLength(20);
  fireEvent.click(screen.getByRole("button", { name: "Next Native order rows" }));
  expect(screen.getByRole("button", { name: "Select Native order sample 20.125 R" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Previous Native order rows" }));
  fireEvent.click(screen.getByRole("button", { name: "Select Native order sample 0.125 R" }));
  expect(next.panels[0]!.samples[0]!.value).toBe(-0);
});

it("keeps partial output and each source estimator level explicit without generic uncertainty", () => {
  const result = admitResultSnapshot({ ...snapshot(), partial: true, panels: [{ ...panel, samples: [
    { ...sample(0, 0), interval: { lower: -0.5, upper: 0.5, method: "source percentile", level: 0.95 } },
    { ...sample(0.125, 1), interval: { lower: 0.5, upper: 1.5, method: "source credible interval", level: null } },
  ] }] });
  render(<ResultInspector result={result} />);
  expect(screen.getByText(/Partial source output/)).toBeTruthy();
  expect(screen.getByText("-0.5–0.5 · source percentile · level 0.95")).toBeTruthy();
  expect(screen.getByText("0.5–1.5 · source credible interval · level unspecified")).toBeTruthy();
  const chart = screen.getByRole("img", { name: "Native order chart" });
  fireEvent.click(chart.querySelector("text")!);
  expect(screen.getByRole("status", { name: "Result selection" }).textContent).toContain("Selected Time: 0 model-time");
  const marker = chart.querySelector("[data-index]")!;
  for (const value of ["-1", "1.5", "NaN", "2"]) {
    marker.setAttribute("data-index", value); fireEvent.click(marker);
    expect(screen.getByRole("status", { name: "Result selection" }).textContent).toContain("Selected Time: 0 model-time");
  }
});

it("exports both complete raw forms through public controls and visibly preserves a local failure", () => {
  const result = snapshot();
  const create = vi.fn(() => "blob:source-inspector"), revoke = vi.fn();
  vi.stubGlobal("URL", class extends URL { static override createObjectURL = create; static override revokeObjectURL = revoke; });
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
  render(<ResultInspector result={result} />);
  fireEvent.click(screen.getByRole("button", { name: "Export raw CSV" }));
  fireEvent.click(screen.getByRole("button", { name: "Export SVG" }));
  expect(create).toHaveBeenCalledTimes(2); expect(revoke).toHaveBeenCalledTimes(2);
  click.mockImplementationOnce(() => { throw new Error("actual local download fault"); });
  fireEvent.click(screen.getByRole("button", { name: "Export raw CSV" }));
  expect(screen.getByRole("alert").textContent).toContain("original samples and saved workspace retained");
  expect(screen.getByRole("table", { name: "Native order raw values" }).textContent).toContain("0.12345678901234568");
  expect(resultCsv(result)).toContain('"-0"');
  fireEvent.click(screen.getByRole("button", { name: "Export SVG" }));
  expect(screen.queryByRole("alert")).toBeNull(); expect(revoke).toHaveBeenCalledTimes(4);
});


it("bounds displayed large-series markers and table pages while retaining the complete source", () => {
  const source = snapshot(Array.from({ length: 2001 }, (_, index) => sample(index, index)));
  render(<ResultInspector result={source} />);
  const chart = screen.getByRole("img", { name: "Native order chart" });
  expect(chart.querySelectorAll("[data-index]").length).toBeLessThanOrEqual(1000);
  expect(chart.querySelector('[data-coordinate="2000"]')).toBeTruthy();
  expect(screen.getByRole("table", { name: "Native order raw values" }).querySelectorAll("tbody tr")).toHaveLength(20);
  expect(screen.getByText("Raw rows 0–19 of 2001; display selection does not change these values.")).toBeTruthy();
  expect(source.panels[0]!.samples).toHaveLength(2001);
  expect(resultCsv(source)).toContain('"2000"');
});
