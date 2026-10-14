// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — exact source result exports and display geometry

import { writeJson } from "../../shared/contracts";
import { rawResultText } from "../../shared/results/resultModel";
import type { ResultPanel, ResultSnapshot } from "../../shared/results/resultModel";

import { projectResultChart } from "../../shared/results/resultChart";
import type { ResultChartElement, ResultSelection } from "../../shared/results/resultChart";
export type { ResultSelection } from "../../shared/results/resultChart";

function xml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}
function cell(value: string | number | null): string {
  const text = value === null ? "" : typeof value === "number" ? rawResultText(value) : value;
  return `"${text.replaceAll('"', '""')}"`;
}

/** Export every original sample and its interval method; no displayed rounding or decimation. */
export function resultCsv(result: ResultSnapshot): string {
  const rows: (string | number | null)[][] = [
    ["source_sha256", result.sourceSha256],
    ["caption", result.caption],
    ["claim_boundary", result.claimBoundary],
    ["partial", String(result.partial)],
    [
      "panel_id",
      "kind",
      "coordinate_label",
      "coordinate_unit",
      "coordinate",
      "object_id",
      "column_id",
      "x_label",
      "x_unit",
      "x",
      "y_label",
      "y_unit",
      "y",
      "value_label",
      "value_unit",
      "value_dtype",
      "value",
      "state",
      "interval_lower",
      "interval_upper",
      "interval_method",
      "interval_level",
    ],
  ];
  for (const panel of result.panels)
    for (const sample of panel.samples)
      rows.push([
        panel.id,
        panel.kind,
        panel.coordinateLabel,
        panel.coordinateUnit,
        sample.coordinate,
        sample.objectId,
        sample.columnId,
        panel.xLabel,
        panel.xUnit,
        sample.x,
        panel.yLabel,
        panel.yUnit,
        sample.y,
        panel.valueLabel,
        panel.valueUnit,
        panel.valueDtype,
        sample.value,
        sample.status,
        sample.interval?.lower ?? null,
        sample.interval?.upper ?? null,
        sample.interval?.method ?? "Not estimated",
        sample.interval?.level ?? null,
      ]);
  return `${rows.map((row) => row.map(cell).join(",")).join("\r\n")}\r\n`;
}

function chartElementSvg(node: ResultChartElement): string {
  const attributes = Object.entries(node.attributes)
    .map(([name, value]) => ` ${name}="${xml(String(value))}"`)
    .join("");
  if (node.selfClosing) return `<${node.tag}${attributes}/>`;
  const content = node.children
    .map((child) => (typeof child === "string" ? xml(child) : chartElementSvg(child)))
    .join("");
  return `<${node.tag}${attributes}>${content}</${node.tag}>`;
}

/** Serialize the same bounded original geometry rendered by the browser chart. */
export function resultPanelSvg(panel: ResultPanel, selection: ResultSelection): string {
  const chart = projectResultChart(panel, selection);
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 240" width="600" height="240" role="img" aria-label="${xml(chart.label)}">${chart.elements.map(chartElementSvg).join("")}</svg>`;
}

/** Export source-bound SVG with complete raw metadata and the current linked view. */
export function resultSvg(result: ResultSnapshot, selection: ResultSelection): string {
  const caption = Array.from(`${result.caption} · ${result.claimBoundary}`);
  const captionLines = Array.from(
    { length: Math.ceil(caption.length / 90) },
    (_, index) =>
      `<tspan x="10" dy="12">${xml(caption.slice(index * 90, (index + 1) * 90).join(""))}</tspan>`,
  ).join("");
  const headerHeight = 75 + Math.ceil(caption.length / 90) * 12;
  const panels = result.panels
    .map(
      (panel, index) =>
        `<g transform="translate(0,${headerHeight + index * 240})">${resultPanelSvg(panel, selection)}</g>`,
    )
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 ${headerHeight + result.panels.length * 240}"><title>${xml(result.title)}</title><desc>${xml(`${result.caption} · ${result.claimBoundary} · source SHA256 ${result.sourceSha256}`)}</desc><metadata>${xml(writeJson(result))}</metadata><text x="10" y="20">${xml(result.title)}</text><text x="10" y="45" font-size="9">source SHA256 ${result.sourceSha256}</text><text x="10" y="55" font-size="10">${captionLines}</text>${panels}</svg>`;
}

/** Download a local immutable export and release its browser URL in every outcome. */
export function downloadResultExport(
  result: ResultSnapshot,
  kind: "csv" | "svg",
  selection: ResultSelection,
): void {
  const bytes = kind === "csv" ? resultCsv(result) : resultSvg(result, selection);
  const url = URL.createObjectURL(
    new Blob([bytes], {
      type: kind === "csv" ? "text/csv;charset=utf-8" : "image/svg+xml;charset=utf-8",
    }),
  );
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = `result-${result.sourceSha256}.${kind}`;
    link.click();
  } finally {
    URL.revokeObjectURL(url);
  }
}
