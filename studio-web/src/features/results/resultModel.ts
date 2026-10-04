// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — read-only source result admission

import { dataEntries } from "../../shared/contracts/canonical";

/** Observable producer metadata refusal; no saved state is changed. */
export class ResultRefusal extends Error {}

/** Source-reported sample validity, including unavailable numerical results. */
export type ResultValueState = "finite" | "missing" | "nan" | "refused";
/** Declared display form; the inspector never computes bins or a spectrum. */
export type ResultPanelKind = "series" | "histogram" | "spectrum" | "matrix";

/** Original estimator-specific interval; no generic error bar is inferred. */
export interface ResultInterval {
  /** Original lower bound in the value unit. */ readonly lower: number;
  /** Original upper bound in the value unit. */ readonly upper: number;
  /** Producer's estimator and interval construction method. */ readonly method: string;
  /** Producer-reported confidence/credible level, or explicitly unspecified. */ readonly level: number | null;
}
/** One immutable original value with explicit linkage and plotting coordinates. */
export interface ResultSample {
  /** Producer coordinate used for linkage, in the panel's declared coordinate unit. */ readonly coordinate: number;
  /** Explicit source object identity shared by linked views. */ readonly objectId: string;
  /** Matrix column object identity, otherwise null. */ readonly columnId: string | null;
  /** Original horizontal coordinate; never a sample index substituted for time. */ readonly x: number;
  /** Matrix vertical coordinate, otherwise null. */ readonly y: number | null;
  /** Source scalar, or null with an explicit nonfinite/unavailable state. */ readonly value: number | null;
  /** Producer-declared sample validity. */ readonly status: ResultValueState;
  /** Original estimator interval, or no estimate. */ readonly interval: ResultInterval | null;
}
/** Producer-declared axes and raw values for one chart and table. */
export interface ResultPanel {
  /** Unique stable panel identity. */ readonly id: string;
  /** Source-supplied accessible chart/table title. */ readonly title: string;
  /** Source display form. */ readonly kind: ResultPanelKind;
  /** Explicit linkage coordinate meaning, such as time or filtration threshold. */ readonly coordinateLabel: string;
  /** Source linkage unit; no conversion is performed. */ readonly coordinateUnit: string;
  /** Source horizontal axis meaning. */ readonly xLabel: string;
  /** Source horizontal unit. */ readonly xUnit: string;
  /** Source matrix vertical meaning, otherwise empty. */ readonly yLabel: string;
  /** Source matrix vertical unit, otherwise empty. */ readonly yUnit: string;
  /** Source scalar meaning. */ readonly valueLabel: string;
  /** Source scalar unit; dimensionless values use "1". */ readonly valueUnit: string;
  /** Original scalar representation; unsafe int64 values refuse display conversion. */ readonly valueDtype: "float64" | "int64";
  /** Original ordered raw samples; pagination/decimation never replace this array. */ readonly samples: readonly ResultSample[];
}
/** Transient admitted producer metadata; no new persisted workspace schema. */
export interface ResultSnapshot {
  /** Supported producer metadata version. */ readonly version: 1;
  /** Producer title. */ readonly title: string;
  /** Source evidence caption carried into every export. */ readonly caption: string;
  /** Producer qualification limits carried into every export. */ readonly claimBoundary: string;
  /** Exact raw producer bytes or original run output artifact SHA256. */ readonly sourceSha256: string;
  /** Producer explicitly reported partial output. */ readonly partial: boolean;
  /** Admitted bounded original panels. */ readonly panels: readonly ResultPanel[];
}
/** Product rendering/import ceilings, independent of host capacity. */
export const resultLimits = Object.freeze({
  /** Maximum linked source panels. */ panels: 16,
  /** Maximum original samples across all panels. */ samples: 65536,
  /** Maximum displayed original markers per panel. */ displayPoints: 1000,
  /** Maximum raw table rows per page. */ tableRows: 20,
  /** Maximum original UTF-8 producer import bytes. */ importBytes: 2 * 1024 * 1024,
});

function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new ResultRefusal("Result object required");
  const entries = dataEntries(value);
  if (entries.length !== keys.length || entries.some(([key]) => !keys.includes(key))) throw new ResultRefusal("Missing or unsupported result metadata fields");
  return Object.fromEntries(entries);
}
function text(value: unknown, empty = false): string {
  if (typeof value !== "string" || value.length > 2048 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/u.test(value) || (!empty && value.length === 0) || /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)) throw new ResultRefusal("Bounded Unicode result text required");
  return value;
}
function finite(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new ResultRefusal("Finite original result coordinate or bound required");
  return value;
}
function interval(value: unknown, scalar: number | null): ResultInterval | null {
  if (value === null) return null;
  const raw = object(value, ["lower", "upper", "method", "level"]);
  const lower = finite(raw["lower"]), upper = finite(raw["upper"]);
  const level = raw["level"] === null ? null : finite(raw["level"]);
  if (scalar === null || lower > scalar || scalar > upper || (level !== null && (level <= 0 || level >= 1))) throw new ResultRefusal("Source interval must bracket a finite value and declare a valid level");
  return Object.freeze({ lower, upper, method: text(raw["method"]), level });
}
function sample(value: unknown, kind: ResultPanelKind): ResultSample {
  const raw = object(value, ["coordinate", "objectId", "columnId", "x", "y", "value", "status", "interval"]);
  const status = raw["status"];
  if (status !== "finite" && status !== "missing" && status !== "nan" && status !== "refused") throw new ResultRefusal("Explicit result sample state required");
  const scalar = status === "finite" ? finite(raw["value"]) : null;
  if (status !== "finite" && raw["value"] !== null) throw new ResultRefusal("Unavailable samples must carry null and their explicit state");
  const y = raw["y"] === null ? null : finite(raw["y"]);
  const columnId = raw["columnId"] === null ? null : text(raw["columnId"]);
  if ((kind === "matrix" && (y === null || columnId === null)) || (kind !== "matrix" && (y !== null || columnId !== null))) throw new ResultRefusal("Matrix coordinates and source column identities required only for matrices");
  return Object.freeze({ coordinate: finite(raw["coordinate"]), objectId: text(raw["objectId"]), columnId, x: finite(raw["x"]), y, value: scalar, status, interval: interval(raw["interval"], scalar) });
}
function panel(value: unknown, remaining: number): ResultPanel {
  const raw = object(value, ["id", "title", "kind", "coordinateLabel", "coordinateUnit", "xLabel", "xUnit", "yLabel", "yUnit", "valueLabel", "valueUnit", "valueDtype", "samples"]);
  const kind = raw["kind"];
  if (kind !== "series" && kind !== "histogram" && kind !== "spectrum" && kind !== "matrix") throw new ResultRefusal("Supported source result display form required");
  if (!Array.isArray(raw["samples"]) || raw["samples"].length < 1 || raw["samples"].length > remaining) throw new ResultRefusal("Bounded nonempty original result samples required");
  const dtype = raw["valueDtype"];
  if (dtype !== "float64" && dtype !== "int64") throw new ResultRefusal("Unsupported source result scalar dtype");
  const samples = raw["samples"].map(value => sample(value, kind));
  if (dtype === "int64" && samples.some(item => item.value !== null && (!Number.isSafeInteger(item.value) || Object.is(item.value, -0)))) throw new ResultRefusal("Original int64 values must remain exact safe integers");
  const identities = new Set<string>(), order = new Map<string, number>();
  for (const item of samples) {
    const identity = JSON.stringify([item.coordinate, item.objectId, item.columnId]);
    if (identities.has(identity)) throw new ResultRefusal("Duplicate source sample identity");
    identities.add(identity);
    const prior = order.get(item.objectId);
    if (kind === "series" && prior !== undefined && item.x <= prior) throw new ResultRefusal("Series coordinates must retain strictly increasing source order per object");
    order.set(item.objectId, item.x);
  }
  return Object.freeze({ id: text(raw["id"]), title: text(raw["title"]), kind, coordinateLabel: text(raw["coordinateLabel"]), coordinateUnit: text(raw["coordinateUnit"]), xLabel: text(raw["xLabel"]), xUnit: text(raw["xUnit"]), yLabel: text(raw["yLabel"], true), yUnit: text(raw["yUnit"], true), valueLabel: text(raw["valueLabel"]), valueUnit: text(raw["valueUnit"]), valueDtype: dtype, samples: Object.freeze(samples) });
}

/** Admit a bounded source projection before replacing visible result state. */
export function admitResultSnapshot(value: unknown): ResultSnapshot {
  const raw = object(value, ["version", "title", "caption", "claimBoundary", "sourceSha256", "partial", "panels"]);
  if (raw["version"] !== 1) throw new ResultRefusal("Unsupported result metadata major version");
  if (typeof raw["sourceSha256"] !== "string" || !/^[0-9a-f]{64}$/.test(raw["sourceSha256"])) throw new ResultRefusal("Exact lowercase raw source SHA256 required");
  if (typeof raw["partial"] !== "boolean") throw new ResultRefusal("Producer completeness declaration required");
  if (!Array.isArray(raw["panels"]) || raw["panels"].length < 1 || raw["panels"].length > resultLimits.panels) throw new ResultRefusal("Bounded nonempty result panels required");
  let remaining = resultLimits.samples;
  const ids = new Set<string>();
  const panels = raw["panels"].map(value => {
    const result = panel(value, remaining);
    if (ids.has(result.id)) throw new ResultRefusal("Unique result panel identities required");
    ids.add(result.id); remaining -= result.samples.length;
    return result;
  });
  const first = panels[0]!;
  if (panels.some(item => item.coordinateLabel !== first.coordinateLabel || item.coordinateUnit !== first.coordinateUnit)) throw new ResultRefusal("Linked panels must declare identical coordinate meaning and units");
  return Object.freeze({ version: 1, title: text(raw["title"]), caption: text(raw["caption"]), claimBoundary: text(raw["claimBoundary"]), sourceSha256: raw["sourceSha256"], partial: raw["partial"], panels: Object.freeze(panels) });
}

/** Exact round-trippable binary64 text, retaining source signed zero. */
export function rawResultText(value: number): string { return Object.is(value, -0) ? "-0" : String(value); }

/** Keep all original values; choose only bounded indices for chart rendering. */
export function resultDisplayIndices(length: number, selectedIndex: number | null = null): readonly number[] {
  if (!Number.isSafeInteger(length) || length < 0 || length > resultLimits.samples || (selectedIndex !== null && (!Number.isSafeInteger(selectedIndex) || selectedIndex < 0 || selectedIndex >= length))) throw new ResultRefusal("Bounded source display length and selected index required");
  if (length <= resultLimits.displayPoints) return Array.from({ length }, (_, index) => index);
  const count = resultLimits.displayPoints - 1;
  const indices = new Set(Array.from({ length: count }, (_, index) => Math.floor(index * (length - 1) / (count - 1))));
  if (selectedIndex !== null) indices.add(selectedIndex);
  return Object.freeze([...indices].sort((left, right) => left - right));
}
