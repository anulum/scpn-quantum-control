// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — forecasting, QSNN, NQS and free-energy panels

import type { FreeEnergyTemplate } from "./learningDomains";
import type { ForecastTemplate, NqsTemplate, QsnnTemplate } from "./learningInstrument";
import { EnergyHistoryPlot } from "./LearningPlots";
import { ResultScrollArea } from "./ResultScrollArea";

const yesNo = (value: boolean) => (value ? "yes" : "no");

/**
 * Show forecasting data custody, the classical reference and held-out scores.
 * @param props - Component inputs.
 * @param props.forecast - Parsed source template.
 * @returns The forecasting panel.
 */
export function ForecastPanel({ forecast }: { readonly forecast: ForecastTemplate }) {
  return (
    <section aria-label="Multimodal forecasting">
      <h5>Multimodal forecasting</h5>
      <dl aria-label="Forecasting data provenance">
        <dt>Source</dt>
        <dd>{forecast.source}</dd>
        <dt>Licence</dt>
        <dd>{forecast.licence}</dd>
        <dt>Generator seed</dt>
        <dd>{forecast.seed}</dd>
      </dl>
      <ResultScrollArea title="Forecasting splits">
        <table aria-label="Forecasting splits">
          <thead>
            <tr>
              <th scope="col">Split</th>
              <th scope="col">Trajectories</th>
              <th scope="col">Missing fraction</th>
            </tr>
          </thead>
          <tbody>
            {forecast.splits.map((split) => (
              <tr key={split.split}>
                <th scope="row">{split.split}</th>
                <td>{split.samples}</td>
                <td>{split.missing}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ResultScrollArea>
      <dl aria-label="Held-out custody">
        <dt>Training trajectories</dt>
        <dd>{forecast.training}</dd>
        <dt>Held-out trajectories</dt>
        <dd>{forecast.heldOut}</dd>
        <dt>Held-out trajectories in training</dt>
        <dd>{forecast.overlapping}</dd>
        <dt>Owner refusal to fit a held-out batch</dt>
        <dd>{forecast.fitRefusal}</dd>
      </dl>
      <dl aria-label="Classical reference on held-out data">
        <dt>Reference model</dt>
        <dd>
          {forecast.model} (ridge {forecast.ridge})
        </dd>
        <dt>Held-out wrapped MSE</dt>
        <dd>{forecast.mse}</dd>
        <dt>Persistence wrapped MSE</dt>
        <dd>{forecast.persistenceMse}</dd>
        <dt>Lower error than persistence</dt>
        <dd>{yesNo(forecast.beatsPersistence)}</dd>
        <dt>Interval coverage (target)</dt>
        <dd>
          {forecast.sampleCoverage} ({forecast.targetCoverage})
        </dd>
      </dl>
      <p>{forecast.intervalBoundary}.</p>
      <p>{forecast.boundary}</p>
    </section>
  );
}

/**
 * Show seeded QSNN descent, the checkpoint resume and held-out custody.
 * @param props - Component inputs.
 * @param props.qsnn - Parsed source template.
 * @returns The QSNN panel.
 */
export function QsnnPanel({ qsnn }: { readonly qsnn: QsnnTemplate }) {
  return (
    <section aria-label="QSNN parameter-shift training">
      <h5>QSNN parameter-shift training</h5>
      <dl aria-label="QSNN data and optimiser">
        <dt>Data</dt>
        <dd>{qsnn.source}</dd>
        <dt>Licence</dt>
        <dd>{qsnn.licence}</dd>
        <dt>Layer seed</dt>
        <dd>{qsnn.seed}</dd>
        <dt>Optimiser</dt>
        <dd>
          {qsnn.method}, learning rate {qsnn.learningRate}
        </dd>
      </dl>
      <ResultScrollArea title="QSNN descent">
        <table aria-label="Loss and gradient per step">
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">Loss</th>
              <th scope="col">Gradient norm</th>
            </tr>
          </thead>
          <tbody>
            {qsnn.steps.map((step) => (
              <tr key={step.index}>
                <th scope="row">{step.index}</th>
                <td>{step.loss}</td>
                <td>{step.gradientNorm}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ResultScrollArea>
      <dl aria-label="Checkpoint identity">
        <dt>Checkpoint schema</dt>
        <dd>{qsnn.checkpointSchema}</dd>
        <dt>Saved after epochs</dt>
        <dd>{qsnn.savedAfterEpochs}</dd>
        <dt>Resumed run reproduces the uninterrupted run</dt>
        <dd>{yesNo(qsnn.resumedMatches)}</dd>
        <dt>Random state</dt>
        <dd>{qsnn.randomness}</dd>
      </dl>
      <dl aria-label="QSNN held-out evaluation">
        <dt>Held-out samples</dt>
        <dd>{qsnn.heldOutSamples}</dd>
        <dt>Held-out loss</dt>
        <dd>{qsnn.heldOutLoss ?? "no in-distribution sample"}</dd>
        <dt>Owner refusal to score training samples</dt>
        <dd>{qsnn.overlapRefusal}</dd>
      </dl>
      <p>{qsnn.heldOutBoundary}.</p>
    </section>
  );
}

/**
 * Show the committed JAX NQS baseline or its declared absence.
 * @param props - Component inputs.
 * @param props.nqs - Parsed source template.
 * @returns The NQS panel.
 */
export function NqsPanel({ nqs }: { readonly nqs: NqsTemplate }) {
  if (nqs.status === "unavailable")
    return (
      <section aria-label="JAX NQS baseline">
        <h5>JAX NQS baseline</h5>
        <p role="status">No committed JAX NQS evidence; nothing is shown in its place.</p>
      </section>
    );
  return (
    <section aria-label="JAX NQS baseline">
      <h5>JAX NQS baseline</h5>
      <dl aria-label="NQS evidence">
        <dt>Evidence</dt>
        <dd>
          {nqs.evidencePath} (SHA-256 {nqs.sha256})
        </dd>
        <dt>Spins and seed</dt>
        <dd>
          {nqs.qubits} spins, seed {nqs.seed}
        </dd>
        <dt>Environment</dt>
        <dd>
          JAX {nqs.jaxVersion}, {nqs.posture}
        </dd>
        <dt>Exact and variational energy</dt>
        <dd>
          {nqs.exact} and {nqs.variational}
        </dd>
        <dt>Relative error (within tolerance)</dt>
        <dd>
          {nqs.relativeError} ({yesNo(nqs.withinTolerance)})
        </dd>
      </dl>
      <EnergyHistoryPlot history={nqs.history} exact={nqs.exact} />
      <p>{nqs.boundary}.</p>
    </section>
  );
}

/**
 * Show the free-energy workflow metrics and the owner's covariance refusals.
 * @param props - Component inputs.
 * @param props.template - Parsed source template.
 * @returns The free-energy panel.
 */
export function FreeEnergyPanel({ template }: { readonly template: FreeEnergyTemplate }) {
  return (
    <section aria-label="Free-energy workflow">
      <h5>Free-energy workflow</h5>
      <dl aria-label="Free-energy data provenance">
        <dt>Dataset</dt>
        <dd>{template.datasetId}</dd>
        <dt>Licence</dt>
        <dd>{template.licenceNote}</dd>
        <dt>Reference</dt>
        <dd>{template.reference}</dd>
      </dl>
      <dl aria-label="Free-energy metrics">
        {template.metrics.map((metric) => [
          <dt key={`${metric.name}-name`}>{metric.name}</dt>,
          <dd key={`${metric.name}-value`}>{metric.value}</dd>,
        ])}
      </dl>
      <ul aria-label="Owner covariance refusals">
        {template.refusals.map((refusal) => (
          <li key={refusal.case}>
            Refused {refusal.case}: {refusal.message}
          </li>
        ))}
      </ul>
      <p>{template.boundary}.</p>
    </section>
  );
}
