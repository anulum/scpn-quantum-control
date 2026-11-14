// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — symmetry, chimera and topology lenses

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { ResultScrollArea } from "./ResultScrollArea";
import type { TopologyInstrument } from "./topologyInstrument";
import type { ChimeraTemplate, SensitivityCase, SensitivityTemplate } from "./topologyLenses";
import { PersistencePanel, RelabellingPanel, SectorPanel } from "./TopologyPanels";
import { TopologyProjection } from "./TopologyProjection";

function ChimeraPanel({
  chimera,
  boundary,
}: {
  readonly chimera: ChimeraTemplate;
  readonly boundary: string;
}) {
  return (
    <section aria-label="Chimera clusters">
      <h5>Chimera clusters</h5>
      <dl aria-label="Cluster order parameters">
        {chimera.clusters.flatMap((cluster, at) => [
          <dt key={`name:${cluster.nodes.join("-")}`}>
            {`Population ${at + 1} (nodes ${cluster.nodes[0]}–${cluster.nodes.at(-1)})`}
          </dt>,
          <dd key={`value:${cluster.nodes.join("-")}`}>{cluster.order}</dd>,
        ])}
        <dt>Global order parameter (time mean)</dt>
        <dd>{chimera.globalOrder}</dd>
        <dt>Chimera index and metastability</dt>
        <dd>{`${chimera.chimeraIndex}, ${chimera.metastability}`}</dd>
        <dt>Configuration</dt>
        <dd>{`${chimera.regime}, seed ${chimera.seed}`}</dd>
      </dl>
      <p>{chimera.source}.</p>
      <TopologyProjection
        clusters={chimera.clusters.map((cluster) => cluster.nodes)}
        boundary={boundary}
      />
      <p>{chimera.boundary}.</p>
    </section>
  );
}

function Matrix({
  title,
  rows,
}: {
  readonly title: string;
  readonly rows: readonly (readonly number[])[];
}) {
  return (
    <table aria-label={title}>
      <tbody>
        {Array.from(rows.entries()).map(([i, row]) => (
          <tr key={`row:${i}`}>
            {Array.from(row.entries()).map(([j, value]) => (
              <td key={`cell:${i}:${j}`}>{value}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Derivative({ selected }: { readonly selected: SensitivityCase }) {
  if (selected.tangent === null)
    return <p role="status">{`No derivative is shown: ${selected.refusal}.`}</p>;
  return <Matrix title="Local JVP for the fixed active set" rows={selected.tangent} />;
}

function SensitivityPanel({ sensitivity }: { readonly sensitivity: SensitivityTemplate }) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  const selected = (sensitivity.cases[index] ?? sensitivity.cases[0]) as SensitivityCase;
  return (
    <section aria-label="Fixed-active-set sensitivity">
      <h5>Fixed-active-set sensitivity</h5>
      <label htmlFor={selectId}>Primal point</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {sensitivity.cases.map((item, at) => (
          <option key={item.name} value={at}>
            {item.name}
          </option>
        ))}
      </select>
      <ResultScrollArea title="Derivative support">
        <table aria-label="Derivative support rows">
          <thead>
            <tr>
              <th scope="col">Constraint</th>
              <th scope="col">Status</th>
              <th scope="col">Derivative class</th>
              <th scope="col">Owner reason</th>
            </tr>
          </thead>
          <tbody>
            {selected.rows.map((row) => (
              <tr key={row.capability}>
                <th scope="row">{row.capability}</th>
                <td>{row.status}</td>
                <td>{row.differentiability}</td>
                <td>{row.evidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ResultScrollArea>
      <Matrix title="Requested tangent direction" rows={sensitivity.direction} />
      <Derivative selected={selected} />
      <p>{sensitivity.boundary}.</p>
    </section>
  );
}

/**
 * Inspect source symmetry, chimera and topology evidence without recomputing it.
 * @param props - Component inputs.
 * @param props.instrument - Parsed source data or its refusal.
 * @returns Every lens, or a visible refusal of contradictory data.
 */
export function TopologyInspector({
  instrument,
}: {
  readonly instrument: Loaded<TopologyInstrument>;
}) {
  if (!instrument.ok)
    return <Unverifiable surface="topology instrument" reason={instrument.reason} />;
  const data = instrument.value;
  return (
    <section aria-label="Symmetry, chimera and topology lenses" className="qsp-panel">
      <h4>Symmetry, chimera and topology lenses</h4>
      <p>
        Source-owned values from small finite examples: exact spectra and evolutions, a finite
        synthetic chimera and exact persistence of sampled phases. The caller's global random state
        was left unchanged.
      </p>
      <RelabellingPanel relabelling={data.relabelling} />
      <SectorPanel sectors={data.sectors} />
      <ChimeraPanel chimera={data.chimera} boundary={data.layoutBoundary} />
      <PersistencePanel persistence={data.persistence} />
      <SensitivityPanel sensitivity={data.sensitivity} />
    </section>
  );
}
