// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — relabelling, sector and persistence panels

import { ResultScrollArea } from "./ResultScrollArea";
import type { RelabellingTemplate, SectorTemplate } from "./topologyInstrument";
import type { PersistenceDimension, PersistenceTemplate } from "./topologyLenses";

/**
 * Show the order parameter for the original and the relabelled nodes.
 * @param props - Component inputs.
 * @param props.relabelling - Parsed source template.
 * @returns The relabelling panel.
 */
export function RelabellingPanel({ relabelling }: { readonly relabelling: RelabellingTemplate }) {
  return (
    <section aria-label="Order parameter under relabelling">
      <h5>Order parameter under relabelling</h5>
      <dl aria-label="Relabelling">
        <dt>Node moves</dt>
        <dd>{relabelling.permutation.map((to, from) => `${from}→${to}`).join(", ")}</dd>
        <dt>Evolution</dt>
        <dd>{relabelling.evolution}</dd>
      </dl>
      <ResultScrollArea title="Order parameter by labelling">
        <table aria-label="Order parameter by labelling">
          <thead>
            <tr>
              <th scope="col">Labelling</th>
              <th scope="col">R</th>
              <th scope="col">ψ (rad)</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ["Original", relabelling.original],
                ["Relabelled", relabelling.relabelled],
              ] as const
            ).map(([name, reading]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td>{reading.magnitude}</td>
                <td>{reading.phase}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ResultScrollArea>
      <dl aria-label="Relabelling differences">
        <dt>|ΔR|</dt>
        <dd>{relabelling.magnitudeDifference}</dd>
        <dt>|Δψ| (rad)</dt>
        <dd>{relabelling.phaseDifference}</dd>
        <dt>Tolerance</dt>
        <dd>{relabelling.tolerance}</dd>
      </dl>
    </section>
  );
}

const sortedKeys = (row: Readonly<Record<string, number>>) =>
  Object.keys(row).sort((left, right) => Number(left) - Number(right));

/**
 * Show sector spectra against the dense reference and the parity weights.
 * @param props - Component inputs.
 * @param props.sectors - Parsed source template.
 * @returns The sector panel.
 */
export function SectorPanel({ sectors }: { readonly sectors: SectorTemplate }) {
  return (
    <section aria-label="Symmetry sectors">
      <h5>Symmetry sectors</h5>
      <dl aria-label="Sector spectra against the dense reference">
        <dt>Parity sector dimensions</dt>
        <dd>{`even ${sectors.parityDimensions["even"]}, odd ${sectors.parityDimensions["odd"]}`}</dd>
        <dt>Magnetisation sector dimensions</dt>
        <dd>
          {sortedKeys(sectors.magnetisationDimensions)
            .map((key) => `M=${key}: ${sectors.magnetisationDimensions[key]}`)
            .join(", ")}
        </dd>
        <dt>Largest deviation from the dense spectrum (parity, magnetisation)</dt>
        <dd>{`${sectors.parityDeviation}, ${sectors.magnetisationDeviation} (tolerance ${sectors.tolerance})`}</dd>
        <dt>Ground energy and parity</dt>
        <dd>{`${sectors.groundEnergy}, ${sectors.groundParity}`}</dd>
      </dl>
      <ResultScrollArea title="Parity weights">
        <table aria-label="Parity weights before and after the evolution">
          <thead>
            <tr>
              <th scope="col">Initial state</th>
              <th scope="col">Even before</th>
              <th scope="col">Odd before</th>
              <th scope="col">Even after</th>
              <th scope="col">Odd after</th>
              <th scope="col">Leakage after</th>
            </tr>
          </thead>
          <tbody>
            {sectors.weights.map((track) => (
              <tr key={track.state}>
                <th scope="row">{track.state.replaceAll("_", " ")}</th>
                <td>{track.before.even}</td>
                <td>{track.before.odd}</td>
                <td>{track.after.even}</td>
                <td>{track.after.odd}</td>
                <td>{track.after.leakage}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ResultScrollArea>
      <p>{sectors.boundary}.</p>
    </section>
  );
}

function SummaryTable({
  title,
  rows,
}: {
  readonly title: string;
  readonly rows: readonly PersistenceDimension[];
}) {
  return (
    <ResultScrollArea title={title}>
      <table aria-label={title}>
        <thead>
          <tr>
            <th scope="col">Dimension</th>
            <th scope="col">Pairs</th>
            <th scope="col">Essential</th>
            <th scope="col">Persistent finite</th>
            <th scope="col">Longest finite lifetime</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.dimension}>
              <th scope="row">{row.dimension}</th>
              <td>{row.pairs}</td>
              <td>{row.essential}</td>
              <td>{row.persistent}</td>
              <td>{row.longest}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ResultScrollArea>
  );
}

/**
 * Show exact finite persistence summaries.
 * @param props - Component inputs.
 * @param props.persistence - Parsed source template.
 * @returns The persistence panel.
 */
export function PersistencePanel({ persistence }: { readonly persistence: PersistenceTemplate }) {
  return (
    <section aria-label="Persistence summaries">
      <h5>Persistence summaries</h5>
      <p>{persistence.method}.</p>
      <dl aria-label="Exact example">
        <dt>Evenly spaced phases: analytic spacing (rad)</dt>
        <dd>{persistence.spacing}</dd>
      </dl>
      <SummaryTable title="Exact example (evenly spaced phases)" rows={persistence.exact} />
      <SummaryTable title="Chimera final phases" rows={persistence.chimera} />
      <p>{persistence.boundary}.</p>
    </section>
  );
}
