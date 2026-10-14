// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned result chart geometry shared by rendering and export

import { rawResultText, resultDisplayIndices, ResultRefusal } from "./resultModel";
import type { ResultPanel, ResultSample } from "./resultModel";

/** Exact linked source selection, independent of rounded plot labels. */
export interface ResultSelection {
  /** Original coordinate. */ readonly coordinate: number;
  /** Original object identity. */ readonly objectId: string;
}
/** Only graphical primitives emitted by the source chart projection. */
export type ResultChartTag = "circle" | "line" | "rect" | "polyline" | "text" | "title" | "desc";
/** Numerical, display and source-link attributes; executable attributes are absent. */
export type ResultChartAttribute =
  | "x"
  | "y"
  | "cx"
  | "cy"
  | "r"
  | "x1"
  | "x2"
  | "y1"
  | "y2"
  | "width"
  | "height"
  | "fill"
  | "stroke"
  | "stroke-width"
  | "font-size"
  | "points"
  | "data-index"
  | "data-coordinate"
  | "data-object"
  | "data-selected";
/** Typed SVG element with escaped text at the rendering or serialization boundary. */
export interface ResultChartElement {
  /** Stable source identity or source-bound annotation identity. */ readonly key: string;
  /** Fixed permitted graphical element. */ readonly tag: ResultChartTag;
  /** Original projected geometry and exact source-link spelling. */ readonly attributes: Readonly<
    Partial<Record<ResultChartAttribute, string | number>>
  >;
  /** Original text or fixed child elements, never markup. */ readonly children: readonly (
    | ResultChartElement
    | string
  )[];
  /** Match the original polyline export's empty-element spelling. */ readonly selfClosing: boolean;
  /** Exact original source index for interactive markers, otherwise null. */ readonly sourceIndex:
    | number
    | null;
  /** Original admitted sample for interactive markers, otherwise null. */ readonly sample: ResultSample | null;
}
/** Bounded original chart projection, shared by browser SVG and source SVG export. */
export interface ResultChartGeometry {
  /** Chart accessible name. */ readonly label: string;
  /** Original ordered SVG elements, with gaps and intervals retained. */ readonly elements: readonly ResultChartElement[];
}

function element(
  key: string,
  tag: ResultChartTag,
  attributes: ResultChartElement["attributes"],
  children: ResultChartElement["children"] = [],
  sourceIndex: number | null = null,
  sample: ResultSample | null = null,
): ResultChartElement {
  return { key, tag, attributes, children, selfClosing: tag === "polyline", sourceIndex, sample };
}
function range(values: readonly number[]): readonly [number, number] {
  let lower = values[0] ?? Number.NaN,
    upper = lower;
  for (const value of values) {
    lower = Math.min(lower, value);
    upper = Math.max(upper, value);
  }
  return [lower, upper];
}
function position(
  value: number,
  limits: readonly [number, number],
  origin: number,
  span: number,
): number {
  const [lower, upper] = limits;
  if (lower === upper) return origin + span / 2;
  const width = upper - lower;
  const fraction = Number.isFinite(width)
    ? (value - lower) / width
    : (value / 2 - lower / 2) / (upper / 2 - lower / 2);
  return origin + span * fraction;
}
function selected(sample: ResultSample, selection: ResultSelection): boolean {
  return sample.coordinate === selection.coordinate && sample.objectId === selection.objectId;
}
function rounded(value: number): string {
  return Math.abs(value) < 1e6 ? value.toFixed(3) : value.toPrecision(4);
}

/** Require the same original matrix coordinate for both axis and cell projection. */
function matrixCoordinate(value: number | null): number {
  if (value === null) throw new ResultRefusal("Original matrix coordinate required");
  return value;
}

/** Preserve original source indices and geometry; missing samples break paths before decimation. */
export function projectResultChart(
  panel: ResultPanel,
  selection: ResultSelection,
): ResultChartGeometry {
  const indexed = panel.samples.flatMap((sample, sourceIndex) =>
    panel.kind === "series" || sample.coordinate === selection.coordinate
      ? [{ sample, sourceIndex }]
      : [],
  );
  const label = `${panel.title} chart`;
  if (indexed.length === 0)
    return {
      label,
      elements: [
        element("empty", "text", { x: 30, y: 120 }, ["No source samples at selected coordinate"]),
      ],
    };
  const samples = indexed.map(({ sample }) => sample);
  const selectedIndex = samples.findIndex((sample) => selected(sample, selection));
  const indices = new Set(
    resultDisplayIndices(samples.length, selectedIndex < 0 ? null : selectedIndex),
  );
  const xs = range(samples.map((sample) => sample.x));
  const yValues = samples.flatMap((sample) => {
    if (panel.kind === "matrix") return [matrixCoordinate(sample.y)];
    return sample.value === null
      ? []
      : [
          sample.value,
          ...(sample.interval === null ? [] : [sample.interval.lower, sample.interval.upper]),
        ];
  });
  if (panel.kind === "histogram") yValues.push(0);
  const ys = range(yValues),
    yRange: readonly [number, number] = Number.isFinite(ys[0]) ? ys : [0, 1];
  const px = (value: number) => position(value, xs, 40, 520),
    py = (value: number) => position(value, yRange, 195, -145);
  const paths = new Map<string, string[]>(),
    segments = new Map<string, number>(),
    lines: ResultChartElement[] = [],
    marks: ResultChartElement[] = [];
  const flush = (objectId: string) => {
    const points = paths.get(objectId);
    if (points) {
      const segment = segments.get(objectId) ?? 0;
      segments.set(objectId, segment + 1);
      lines.push(
        element(JSON.stringify(["path", objectId, segment]), "polyline", {
          fill: "none",
          stroke: "currentColor",
          points: points.join(" "),
        }),
      );
      paths.delete(objectId);
    }
  };
  for (const [index, { sample, sourceIndex }] of indexed.entries()) {
    if (sample.status !== "finite") {
      flush(sample.objectId);
      continue;
    }
    if (!indices.has(index)) continue;
    if (sample.value === null) throw new ResultRefusal("Original finite result value required");
    const ordinate = panel.kind === "matrix" ? matrixCoordinate(sample.y) : sample.value;
    const x = px(sample.x),
      y = py(ordinate),
      active = selected(sample, selection);
    const key = JSON.stringify([sample.coordinate, sample.objectId, sample.columnId]);
    const attrs = {
      "data-index": sourceIndex,
      "data-coordinate": rawResultText(sample.coordinate),
      "data-object": sample.objectId,
      "data-selected": String(active),
    };
    const tooltip = `${panel.valueLabel}=${rawResultText(sample.value)} ${panel.valueUnit} · ${panel.coordinateLabel}=${rawResultText(sample.coordinate)} ${panel.coordinateUnit}${sample.interval === null ? " · Not estimated" : ` · ${sample.interval.method}`}`;
    const title = element(`${key}:title`, "title", {}, [tooltip]);
    if (panel.kind === "series" || panel.kind === "spectrum") {
      const points = paths.get(sample.objectId) ?? [];
      points.push(`${x},${y}`);
      paths.set(sample.objectId, points);
      marks.push(
        element(
          key,
          "circle",
          { cx: x, cy: y, r: active ? 6 : 4, fill: "currentColor", ...attrs },
          [title],
          sourceIndex,
          sample,
        ),
      );
    } else if (panel.kind === "histogram") {
      marks.push(
        element(
          key,
          "line",
          {
            x1: x,
            x2: x,
            y1: py(0),
            y2: y,
            stroke: "currentColor",
            "stroke-width": active ? 3 : 1,
            ...attrs,
          },
          [title],
          sourceIndex,
          sample,
        ),
      );
    } else {
      marks.push(
        element(
          key,
          "rect",
          {
            x: x - 9,
            y: y - 9,
            width: 18,
            height: 18,
            fill: "none",
            stroke: "currentColor",
            "stroke-width": active ? 3 : 1,
            ...attrs,
          },
          [title],
          sourceIndex,
          sample,
        ),
      );
    }
    marks.push(
      element(`${key}:value`, "text", { x: x + 5, y: y - 6, "font-size": 10 }, [
        rounded(sample.value),
      ]),
    );
    if (sample.interval !== null && panel.kind !== "matrix")
      marks.push(
        element(
          `${key}:interval`,
          "line",
          {
            x1: x,
            x2: x,
            y1: py(sample.interval.lower),
            y2: py(sample.interval.upper),
            stroke: "currentColor",
          },
          [element(`${key}:interval-method`, "title", {}, [sample.interval.method])],
        ),
      );
  }
  for (const objectId of paths.keys()) flush(objectId);
  const axis = `${panel.xLabel} (${panel.xUnit}) · ${panel.valueLabel} (${panel.valueUnit})`;
  return {
    label,
    elements: [
      element("title", "title", {}, [panel.title]),
      element("description", "desc", {}, [
        `${axis}; rounded display labels only; histogram marker width has no bin-width meaning`,
      ]),
      ...lines,
      ...marks,
      element("axis", "text", { x: 40, y: 230 }, [axis]),
    ],
  };
}
