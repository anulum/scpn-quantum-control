// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — linked source result charts and raw tables

import { useState } from "react";
import { downloadResultExport, resultPanelSvg } from "./resultExport";
import type { ResultSelection } from "./resultExport";
import { rawResultText, resultLimits } from "./resultModel";
import type { ResultPanel, ResultSample, ResultSnapshot } from "./resultModel";

/** Already admitted immutable original result, independent of workspace persistence. */
export interface ResultInspectorProps {
  /** Source-owned values and explicit metadata from an admitted producer adapter. */ readonly result: ResultSnapshot;
}
const states = { finite: "Finite", missing: "Missing", nan: "NaN", refused: "Refused" } as const;

/** Link exact source selection across bounded display charts, complete raw tables and exports. */
export function ResultInspector({ result }: ResultInspectorProps) {
  const first = result.panels[0]!.samples[0]!;
  const [chosen, setChosen] = useState({ source: result.sourceSha256, coordinate: first.coordinate, objectId: first.objectId });
  const [pages, setPages] = useState<{ source: string; offsets: Readonly<Record<string, number>> }>({ source: result.sourceSha256, offsets: {} });
  const [exportError, setExportError] = useState("");
  const selection: ResultSelection = chosen.source === result.sourceSha256 ? chosen : first;
  const select = (sample: ResultSample) => {
    setChosen({ source: result.sourceSha256, coordinate: sample.coordinate, objectId: sample.objectId });
    const offsets = Object.fromEntries(result.panels.map(panel => {
      const exact = panel.samples.findIndex(item => item.coordinate === sample.coordinate && item.objectId === sample.objectId);
      const index = exact < 0 ? panel.samples.findIndex(item => item.coordinate === sample.coordinate) : exact;
      return [panel.id, index < 0 ? 0 : Math.floor(index / resultLimits.tableRows) * resultLimits.tableRows];
    }));
    setPages({ source: result.sourceSha256, offsets });
  };
  const exportResult = (kind: "csv" | "svg") => {
    setExportError("");
    try { downloadResultExport(result, kind, selection); }
    catch { setExportError("Result export unavailable; original samples and saved workspace retained."); }
  };
  const chartClick = (panel: ResultPanel, target: EventTarget) => {
    // React normalises native text-node click targets to their parent element.
    const marker = (target as Element).closest("[data-index]");
    if (marker === null) return;
    const index = Number(marker.getAttribute("data-index"));
    if (Number.isSafeInteger(index) && index >= 0 && index < panel.samples.length) select(panel.samples[index]!);
  };
  const unavailable = result.panels.reduce((count, panel) => count + panel.samples.filter(sample => sample.status !== "finite").length, 0);
  return <section className="qsp-workspace" aria-label="Result value inspector">
    <h4>Result value inspector</h4>
    <h5>{result.title}</h5>
    <p>{result.caption}</p><p>{result.claimBoundary}</p>
    <p>Source SHA256: <code>{result.sourceSha256}</code>. This identifies source bytes; producer declarations remain subject to their stated qualification limits.</p>
    {result.partial && <p role="status">Partial source output. Missing samples and unfinished evidence remain explicit.</p>}
    {unavailable > 0 && <p role="status">{unavailable} missing, NaN or refused source samples. No interpolation or zero uncertainty is supplied.</p>}
    <p>Chart labels are rounded for display. Up to {resultLimits.displayPoints} original points are displayed per panel; complete raw tables and exports retain every sample. Missing values break connecting lines.</p>
    <p role="status" aria-label="Result selection">Selected {result.panels[0]!.coordinateLabel}: {rawResultText(selection.coordinate)} {result.panels[0]!.coordinateUnit} · object {selection.objectId}</p>
    <div className="qsp-workspace-actions">
      <button type="button" onClick={() => exportResult("csv")}>Export raw CSV</button>
      <button type="button" onClick={() => exportResult("svg")}>Export SVG</button>
    </div>
    {exportError && <p role="alert">{exportError}</p>}
    {result.panels.map(panel => {
      const count = panel.samples.length;
      const last = Math.floor((count - 1) / resultLimits.tableRows) * resultLimits.tableRows;
      const offset = Math.min(pages.source === result.sourceSha256 ? pages.offsets[panel.id] ?? 0 : 0, last);
      const end = Math.min(offset + resultLimits.tableRows, count);
      const page = (next: number) => setPages({ source: result.sourceSha256, offsets: { ...(pages.source === result.sourceSha256 ? pages.offsets : {}), [panel.id]: next } });
      return <section key={panel.id} aria-label={panel.title + " result"}>
        <h5>{panel.title}</h5>
        <div className="qsp-data-scroll" onClick={event => chartClick(panel, event.target)} dangerouslySetInnerHTML={{ __html: resultPanelSvg(panel, selection) }} />
        <div className="qsp-data-scroll" role="region" aria-label={panel.title + " raw table scrolling"} tabIndex={0}>
          <table>
            <caption>{panel.title} raw values</caption>
            <thead><tr><th scope="col">{panel.coordinateLabel} ({panel.coordinateUnit})</th><th scope="col">Object</th><th scope="col">Column object</th><th scope="col">{panel.xLabel} ({panel.xUnit})</th><th scope="col">{panel.yLabel} ({panel.yUnit})</th><th scope="col">{panel.valueLabel} ({panel.valueUnit})</th><th scope="col">State</th><th scope="col">Source interval and method</th><th scope="col">Selection</th></tr></thead>
            <tbody>{panel.samples.slice(offset, end).map((sample, local) => <tr key={offset + local} aria-selected={sample.coordinate === selection.coordinate && sample.objectId === selection.objectId}>
              <th scope="row">{rawResultText(sample.coordinate)}</th><td>{sample.objectId}</td><td>{sample.columnId ?? "Not applicable"}</td><td>{rawResultText(sample.x)}</td><td>{sample.y === null ? "Not applicable" : rawResultText(sample.y)}</td>
              <td>{sample.value === null ? states[sample.status] : rawResultText(sample.value)}</td><td>{states[sample.status]}</td>
              <td>{sample.interval === null ? "Not estimated" : rawResultText(sample.interval.lower) + "–" + rawResultText(sample.interval.upper) + " · " + sample.interval.method + (sample.interval.level === null ? " · level unspecified" : " · level " + rawResultText(sample.interval.level))}</td>
              <td><button type="button" onClick={() => select(sample)}>Select {panel.title} sample {rawResultText(sample.coordinate)} {sample.objectId}{sample.columnId === null ? "" : " " + sample.columnId}</button></td>
            </tr>)}</tbody>
          </table>
        </div>
        <p role="status">Raw rows {offset}–{end - 1} of {count}; display selection does not change these values.</p>
        <div className="qsp-workspace-actions">
          <button type="button" disabled={offset === 0} onClick={() => page(offset - resultLimits.tableRows)}>Previous {panel.title} rows</button>
          <button type="button" disabled={end === count} onClick={() => page(offset + resultLimits.tableRows)}>Next {panel.title} rows</button>
        </div>
      </section>;
    })}
  </section>;
}
