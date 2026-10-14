// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — typed original SVG with source-bound pointer and keyboard selection

import { createElement } from "react";
import type { KeyboardEvent, MouseEvent, ReactNode } from "react";
import { projectResultChart } from "../../shared/results/resultChart";
import type { ResultChartElement, ResultSelection } from "../../shared/results/resultChart";
import type { ResultPanel, ResultSample } from "../../shared/results/resultModel";

/** Admitted source chart and original linked selection owner. */
export interface ResultChartProps {
  /** Immutable original panel. */ readonly panel: ResultPanel;
  /** Exact selected source coordinate/object. */ readonly selection: ResultSelection;
  /** Select the original sample without reading source data from DOM attributes. */ readonly onSelect: (
    sample: ResultSample,
  ) => void;
}

function chartNode(
  node: ResultChartElement,
  panel: ResultPanel,
  onSelect: ResultChartProps["onSelect"],
): ReactNode {
  const attributes = Object.fromEntries(
    Object.entries(node.attributes).map(([name, value]) => [
      name === "stroke-width" ? "strokeWidth" : name === "font-size" ? "fontSize" : name,
      value,
    ]),
  );
  const activate = (target: Element) => {
    const index = target.getAttribute("data-index");
    if (
      node.sample !== null &&
      node.sourceIndex !== null &&
      index !== null &&
      Number(index) === node.sourceIndex
    )
      onSelect(node.sample);
  };
  const interactive =
    node.sample === null
      ? {}
      : {
          role: "button",
          tabIndex: 0,
          "aria-label": `Select plotted ${panel.title} sample ${node.sample.coordinate} ${node.sample.objectId}`,
          onClick: (event: MouseEvent<SVGElement>) => activate(event.currentTarget),
          onKeyDown: (event: KeyboardEvent<SVGElement>) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              activate(event.currentTarget);
            }
          },
        };
  return createElement(
    node.tag,
    { ...attributes, ...interactive, key: node.key },
    ...node.children.map((child) =>
      typeof child === "string" ? child : chartNode(child, panel, onSelect),
    ),
  );
}

/** Render escaped SVG text and typed primitives; pointer and keyboard controls bind original samples. */
export function ResultChart({ panel, selection, onSelect }: ResultChartProps) {
  const chart = projectResultChart(panel, selection);
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 600 240"
      width="600"
      height="240"
      role="img"
      aria-label={chart.label}
    >
      {chart.elements.map((node) => chartNode(node, panel, onSelect))}
    </svg>
  );
}
