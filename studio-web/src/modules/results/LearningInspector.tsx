// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — learning, forecasting and domain-benchmark instrument

import { useId, useState } from "react";
import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { exportDomainProxy } from "./learningDomains";
import type { DomainProxy } from "./learningDomains";
import type { LearningInstrument } from "./learningInstrument";
import { ForecastPanel, FreeEnergyPanel, NqsPanel, QsnnPanel } from "./LearningPanels";
import { ProxyMetricsPlot } from "./LearningPlots";

function KitList({ label, items }: { readonly label: string; readonly items: readonly string[] }) {
  return (
    <ul aria-label={label}>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

function DomainPanel({ domains }: { readonly domains: readonly DomainProxy[] }) {
  const [index, setIndex] = useState(0);
  const selectId = useId();
  const row = (domains[index] ?? domains[0]) as DomainProxy;
  return (
    <section aria-label="Domain benchmark proxies">
      <h5>Domain benchmark proxies</h5>
      <label htmlFor={selectId}>Domain benchmark</label>
      <select
        id={selectId}
        value={index}
        onChange={(event) => setIndex(Number(event.target.value))}
      >
        {domains.map((domain, at) => (
          <option key={domain.datasetId} value={at}>
            {domain.datasetId}
          </option>
        ))}
      </select>
      <p role="note">
        Proxy metrics of a curated benchmark under the {row.kitId} kit ({row.supportStatus}); no
        measured physical outcome is in custody.
      </p>
      <ProxyMetricsPlot dataset={row.datasetId} metrics={row.metrics} />
      <dl aria-label="Benchmark provenance">
        <dt>Domain</dt>
        <dd>{row.domain}</dd>
        <dt>Licence</dt>
        <dd>{row.licenceNote}</dd>
        <dt>Reference</dt>
        <dd>{row.reference}</dd>
      </dl>
      <KitList label="Allowed uses" items={row.allowedUses} />
      <KitList label="Caveats" items={row.caveats} />
      <KitList label="Claims this benchmark never supports" items={row.forbidden} />
      <a
        href={`data:application/json;charset=utf-8,${encodeURIComponent(exportDomainProxy(row))}`}
        download={`domain-proxy-${row.datasetId}.json`}
      >
        Download proxy metadata
      </a>
    </section>
  );
}

/**
 * Inspect source learning, forecasting and domain-benchmark evidence without recomputing it.
 * @param props - Component inputs.
 * @param props.instrument - Parsed source data or its refusal.
 * @returns Every template, or a visible refusal of contradictory data.
 */
export function LearningInspector({
  instrument,
}: {
  readonly instrument: Loaded<LearningInstrument>;
}) {
  if (!instrument.ok)
    return <Unverifiable surface="learning instrument" reason={instrument.reason} />;
  const data = instrument.value;
  return (
    <section aria-label="Learning, forecasting and domain benchmarks" className="qsp-panel">
      <h4>Learning, forecasting and domain benchmarks</h4>
      <p>
        Source-owned values only: synthetic and curated inputs, seeded local runs and committed
        evidence. The caller's global random state was left unchanged.
      </p>
      <ForecastPanel forecast={data.forecasting} />
      <QsnnPanel qsnn={data.qsnn} />
      <NqsPanel nqs={data.nqs} />
      <DomainPanel domains={data.domains} />
      <FreeEnergyPanel template={data.freeEnergy} />
    </section>
  );
}
