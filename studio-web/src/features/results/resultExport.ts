// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact source result exports and display geometry

import { writeJson } from "../../shared/contracts";
import { rawResultText, resultDisplayIndices } from "./resultModel";
import type { ResultPanel, ResultSample, ResultSnapshot } from "./resultModel";

/** Explicit source linkage selected by the user, independent of plot rounding. */
export interface ResultSelection {
  /** Original linked coordinate. */ readonly coordinate: number;
  /** Original source object identity. */ readonly objectId: string;
}

function xml(value: string): string { return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&apos;"); }
function cell(value: string | number | null): string { return '"' + (value === null ? "" : typeof value === "number" ? rawResultText(value) : value).replaceAll('"', '""') + '"'; }

/** Export every original sample and its interval method; no displayed rounding or decimation. */
export function resultCsv(result: ResultSnapshot): string {
  const rows: (string | number | null)[][] = [
    ["source_sha256", result.sourceSha256], ["caption", result.caption], ["claim_boundary", result.claimBoundary], ["partial", String(result.partial)],
    ["panel_id", "kind", "coordinate_label", "coordinate_unit", "coordinate", "object_id", "column_id", "x_label", "x_unit", "x", "y_label", "y_unit", "y", "value_label", "value_unit", "value_dtype", "value", "state", "interval_lower", "interval_upper", "interval_method", "interval_level"],
  ];
  for (const panel of result.panels) for (const sample of panel.samples) rows.push([panel.id, panel.kind, panel.coordinateLabel, panel.coordinateUnit, sample.coordinate, sample.objectId, sample.columnId, panel.xLabel, panel.xUnit, sample.x, panel.yLabel, panel.yUnit, sample.y, panel.valueLabel, panel.valueUnit, panel.valueDtype, sample.value, sample.status, sample.interval?.lower ?? null, sample.interval?.upper ?? null, sample.interval?.method ?? "Not estimated", sample.interval?.level ?? null]);
  return rows.map(row => row.map(cell).join(",")).join("\r\n") + "\r\n";
}

function range(values: readonly number[]): readonly [number, number] {
  let lower = values[0]!, upper = lower;
  for (const value of values) { lower = Math.min(lower, value); upper = Math.max(upper, value); }
  return [lower, upper];
}
function position(value: number, limits: readonly [number, number], origin: number, span: number): number {
  const [lower, upper] = limits;
  if (lower === upper) return origin + span / 2;
  const width = upper - lower;
  const fraction = Number.isFinite(width) ? (value - lower) / width : (value / 2 - lower / 2) / (upper / 2 - lower / 2);
  return origin + span * fraction;
}
function selected(sample: ResultSample, selection: ResultSelection): boolean { return sample.coordinate === selection.coordinate && sample.objectId === selection.objectId; }
function rounded(value: number): string { return Math.abs(value) < 1e6 ? value.toFixed(3) : value.toPrecision(4); }

/** Render only bounded original sample indices; gaps break lines before any decimation. */
export function resultPanelSvg(panel: ResultPanel, selection: ResultSelection): string {
  const sourceIndices: number[] = [];
  for (let index = 0; index < panel.samples.length; index++) if (panel.kind === "series" || panel.samples[index]!.coordinate === selection.coordinate) sourceIndices.push(index);
  const samples = sourceIndices.map(index => panel.samples[index]!);
  const title = xml(panel.title + " chart");
  if (samples.length === 0) return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="600" height="240" role="img" aria-label="' + title + '"><text x="30" y="120">No source samples at selected coordinate</text></svg>';
  const selectedIndex = samples.findIndex(sample => selected(sample, selection));
  const indices = new Set(resultDisplayIndices(samples.length, selectedIndex < 0 ? null : selectedIndex));
  const xs = range(samples.map(sample => sample.x));
  const yValues = samples.flatMap(sample => panel.kind === "matrix" ? [sample.y!] : sample.value === null ? [] : [sample.value, ...(sample.interval === null ? [] : [sample.interval.lower, sample.interval.upper])]);
  if (panel.kind === "histogram") yValues.push(0);
  const ys = range(yValues);
  const yRange: readonly [number, number] = Number.isFinite(ys[0]) ? ys : [0, 1];
  const px = (value: number) => position(value, xs, 40, 520);
  const py = (value: number) => position(value, yRange, 195, -145);
  const paths = new Map<string, string[]>(), lines: string[] = [], marks: string[] = [];
  const flush = (objectId: string) => {
    const points = paths.get(objectId);
    if (points) { lines.push('<polyline fill="none" stroke="currentColor" points="' + points.join(" ") + '"/>'); paths.delete(objectId); }
  };
  for (let index = 0; index < samples.length; index++) {
    const sample = samples[index]!;
    if (sample.status !== "finite") { flush(sample.objectId); continue; }
    if (!indices.has(index)) continue;
    const x = px(sample.x), y = py(panel.kind === "matrix" ? sample.y! : sample.value!);
    const sourceIndex = sourceIndices[index]!;
    const attrs = ' data-index="' + sourceIndex + '" data-coordinate="' + xml(rawResultText(sample.coordinate)) + '" data-object="' + xml(sample.objectId) + '" data-selected="' + String(selected(sample, selection)) + '"';
    const tooltip = xml(panel.valueLabel + "=" + rawResultText(sample.value!) + " " + panel.valueUnit + " · " + panel.coordinateLabel + "=" + rawResultText(sample.coordinate) + " " + panel.coordinateUnit + (sample.interval === null ? " · Not estimated" : " · " + sample.interval.method));
    if (panel.kind === "series" || panel.kind === "spectrum") {
      const points = paths.get(sample.objectId) ?? []; points.push(String(x) + "," + String(y)); paths.set(sample.objectId, points);
      marks.push('<circle cx="' + x + '" cy="' + y + '" r="' + (selected(sample, selection) ? 6 : 4) + '" fill="currentColor"' + attrs + '><title>' + tooltip + '</title></circle>');
    } else if (panel.kind === "histogram") {
      marks.push('<line x1="' + x + '" x2="' + x + '" y1="' + py(0) + '" y2="' + y + '" stroke="currentColor" stroke-width="' + (selected(sample, selection) ? 3 : 1) + '"' + attrs + '><title>' + tooltip + '</title></line>');
    } else {
      marks.push('<rect x="' + (x - 9) + '" y="' + (y - 9) + '" width="18" height="18" fill="none" stroke="currentColor" stroke-width="' + (selected(sample, selection) ? 3 : 1) + '"' + attrs + '><title>' + tooltip + '</title></rect>');
    }
    marks.push('<text x="' + (x + 5) + '" y="' + (y - 6) + '" font-size="10">' + xml(rounded(sample.value!)) + '</text>');
    if (sample.interval !== null && panel.kind !== "matrix") marks.push('<line x1="' + x + '" x2="' + x + '" y1="' + py(sample.interval.lower) + '" y2="' + py(sample.interval.upper) + '" stroke="currentColor"><title>' + xml(sample.interval.method) + '</title></line>');
  }
  for (const objectId of paths.keys()) flush(objectId);
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="600" height="240" role="img" aria-label="' + title + '"><title>' + xml(panel.title) + '</title><desc>' + xml(panel.xLabel + " (" + panel.xUnit + ") · " + panel.valueLabel + " (" + panel.valueUnit + "); rounded display labels only; histogram marker width has no bin-width meaning") + '</desc>' + lines.join("") + marks.join("") + '<text x="40" y="230">' + xml(panel.xLabel + " (" + panel.xUnit + ") · " + panel.valueLabel + " (" + panel.valueUnit + ")") + '</text></svg>';
}

/** Export source-bound SVG with complete raw metadata and the current linked view. */
export function resultSvg(result: ResultSnapshot, selection: ResultSelection): string {
  const caption = Array.from(result.caption + " · " + result.claimBoundary);
  const captionLines = Array.from({ length: Math.ceil(caption.length / 90) }, (_, index) => '<tspan x="10" dy="12">' + xml(caption.slice(index * 90, (index + 1) * 90).join("")) + '</tspan>').join("");
  const headerHeight = 75 + Math.ceil(caption.length / 90) * 12;
  const panels = result.panels.map((panel, index) => '<g transform="translate(0,' + (headerHeight + index * 240) + ')">' + resultPanelSvg(panel, selection) + '</g>').join("");
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 ' + (headerHeight + result.panels.length * 240) + '"><title>' + xml(result.title) + '</title><desc>' + xml(result.caption + " · " + result.claimBoundary + " · source SHA256 " + result.sourceSha256) + '</desc><metadata>' + xml(writeJson(result)) + '</metadata><text x="10" y="20">' + xml(result.title) + '</text><text x="10" y="45" font-size="9">source SHA256 ' + result.sourceSha256 + '</text><text x="10" y="55" font-size="10">' + captionLines + '</text>' + panels + '</svg>';
}

/** Download a local immutable export and release its browser URL in every outcome. */
export function downloadResultExport(result: ResultSnapshot, kind: "csv" | "svg", selection: ResultSelection): void {
  const bytes = kind === "csv" ? resultCsv(result) : resultSvg(result, selection);
  const url = URL.createObjectURL(new Blob([bytes], { type: kind === "csv" ? "text/csv;charset=utf-8" : "image/svg+xml;charset=utf-8" }));
  try {
    const link = document.createElement("a"); link.href = url; link.download = "result-" + result.sourceSha256 + "." + kind; link.click();
  } finally { URL.revokeObjectURL(url); }
}
