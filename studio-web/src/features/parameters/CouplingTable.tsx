// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-equivalent sparse coupling table

import { parameterElementText } from "./parameterDraft";
import type { ParameterValue } from "./parameterDraft";

/** A presentation over the admitted original square parameter matrix. */
export interface CouplingTableProps {
  /** Original parameter key shared by matrix, graph and selected form. */
  readonly parameterKey: string;
  /** Source-admitted square row-major coefficient payload. */
  readonly value: ParameterValue;
  /** Original declared coefficient unit; no conversion is inferred. */
  readonly unit: string;
  /** Selected row-major matrix index, or no selection. */
  readonly selectedIndex: number | null;
  /** Select the original index in the shared parameter draft. */
  readonly onSelect: (index: number) => void;
}

/** Show exactly the graph's nonzero directed coefficients with textual identity and selection. */
export function CouplingTable({ parameterKey, value, unit, selectedIndex, onSelect }: CouplingTableProps) {
  const n = Number(value.shape[0]);
  const edges = value.values.flatMap((_, index) => {
    const text = parameterElementText(value, index);
    return text === "0" || text === "-0" ? [] : [{ index, text }];
  });
  return <>
    <div className="qsp-data-scroll" role="region" aria-label={parameterKey + " coupling table scrolling"} tabIndex={0}>
      <table>
        <caption>{parameterKey} coupling edges</caption>
        <thead><tr><th scope="col">From j</th><th scope="col">To i</th><th scope="col">Coefficient ({unit})</th><th scope="col">Select</th></tr></thead>
        <tbody>{edges.map(edge => <tr key={edge.index} data-matrix-index={edge.index}>
          <td>{edge.index % n}</td><td>{Math.floor(edge.index / n)}</td><td>{edge.text}</td>
          <td><button type="button" aria-pressed={selectedIndex === edge.index} onClick={() => onSelect(edge.index)}>Edge {edge.index % n} → {Math.floor(edge.index / n)}: {edge.text} {unit}</button></td>
        </tr>)}</tbody>
      </table>
    </div>
    {edges.length === 0 && <p role="status">No nonzero coupling edges. Zero coefficients remain available in the matrix.</p>}
  </>;
}
