// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned sensing and entropy provenance instrument

import { useId, useState } from "react";
import type { Loaded } from "../../../shared/browser/data";
import { Unverifiable } from "../../../shared/browser/Unverifiable";
import { EvidenceInspector } from "../../../shared/evidence/EvidenceInspector";
import { entropyEvidenceView, exportEntropyMetadata } from "./sensingInstrument";
import type { EntropyMetadata, SensingInstrument } from "./sensingInstrument";

/**
 * Inspect source scores and public entropy metadata without exporting private material.
 * @param props - Component inputs.
 * @param props.instrument - Parsed source metadata or its refusal.
 * @returns Source declarations and a safe metadata download.
 */
export function SensingInspector({
  instrument,
}: {
  readonly instrument: Loaded<SensingInstrument>;
}) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  if (!instrument.ok)
    return <Unverifiable surface="sensing instrument" reason={instrument.reason} />;
  const item = (instrument.value.entropy[index] ?? instrument.value.entropy[0]) as EntropyMetadata;
  const verdict = (passed: boolean) => (passed ? "passed" : "failed");
  return (
    <section aria-label="Sensing and entropy provenance" className="qsp-panel">
      <h4>Sensing and entropy provenance</h4>
      <p>{instrument.value.designBoundary}.</p>
      <p>{instrument.value.model}.</p>
      <section
        className="qsp-table-scroll"
        // biome-ignore lint/a11y/noNoninteractiveTabindex: Keyboard access to the overflow region is required by axe at mobile width.
        tabIndex={0}
        aria-label="Measurement-design table scroll"
      >
        <table aria-label="Measurement-design candidates">
          <thead>
            <tr>
              <th scope="col">Observation</th>
              <th scope="col">Shots</th>
              <th scope="col">Prior variance</th>
              <th scope="col">Sensitivity</th>
              <th scope="col">Noise variance</th>
              <th scope="col">Expected gain (nats)</th>
              <th scope="col">Posterior variance</th>
            </tr>
          </thead>
          <tbody>
            {instrument.value.candidates.map((candidate) => (
              <tr key={candidate.id}>
                <th scope="row">{candidate.id}</th>
                <td>{candidate.shots}</td>
                <td>{candidate.prior}</td>
                <td>{candidate.sensitivity}</td>
                <td>{candidate.noise}</td>
                <td>{candidate.gain}</td>
                <td>{candidate.posterior}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <label htmlFor={selectId}>Entropy record</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {instrument.value.entropy.map((record, at) => (
          <option key={record.id} value={at}>
            {record.id}
          </option>
        ))}
      </select>
      <dl aria-label="Entropy provenance">
        <dt>Declared origin</dt>
        <dd>{item.kind}</dd>
        <dt>Source</dt>
        <dd>{item.source ?? "Not supplied"}</dd>
        <dt>Extractor</dt>
        <dd>{item.extractor ?? "Not supplied"}</dd>
        <dt>Assumptions</dt>
        <dd>{item.assumptions ?? "Not supplied"}</dd>
      </dl>
      {item.diagnostics === null ? (
        <p role="status">No samples or diagnostics supplied; device evidence unavailable.</p>
      ) : (
        <dl aria-label="Recorded diagnostic outcomes">
          <dt>Frequency test</dt>
          <dd>{verdict(item.diagnostics.frequency)}</dd>
          <dt>Runs test</dt>
          <dd>{verdict(item.diagnostics.runs)}</dd>
          <dt>Producer signature check</dt>
          <dd>
            {verdict(item.diagnostics.signature)} — recorded declaration; not independently verified
            in this browser
          </dd>
        </dl>
      )}
      <p role="note">Source entropy: unqualified. {item.boundary}.</p>
      <EvidenceInspector view={entropyEvidenceView(item)} inputRevision={item.id} />
      <a
        href={`data:application/json;charset=utf-8,${encodeURIComponent(exportEntropyMetadata(item))}`}
        download={`entropy-${item.id}.json`}
      >
        Download public entropy metadata
      </a>
    </section>
  );
}
