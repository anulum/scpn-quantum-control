// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — attested result-pack verification in the browser

import type { Loaded } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import type { AttestationInstrument } from "./attestationInstrument";
import { CaseTable, EnvelopeFileCheck, OraclePanel } from "./AttestationPanels";

/**
 * Attested result-pack envelopes decided in this browser and by the Python owner.
 *
 * The browser recomputes every digest itself; the recorded Python verdict is
 * shown beside it as an independent reference. VERIFIED states structure and
 * digest integrity of an unsigned envelope, nothing more.
 */
export function AttestationInspector({
  instrument,
}: {
  readonly instrument: Loaded<AttestationInstrument>;
}) {
  if (!instrument.ok)
    return (
      <Unverifiable
        surface="data/studio/attestation_instrument_20261115.json"
        reason={instrument.reason}
      />
    );
  const data = instrument.value;
  return (
    <article className="qsp-panel" aria-label="Attestation verification">
      <h4>Attestation verification</h4>
      <p className="qsp-banner">{data.boundary}.</p>
      <dl aria-label="Sealed source">
        <dt>Source summary</dt>
        <dd>
          <code>{data.source.path}</code>, sha256 <code>{data.source.sha256}</code>
        </dd>
        <dt>Backend and status</dt>
        <dd>{`${data.source.backend}: ${data.source.status}`}</dd>
        <dt>Reference verifier</dt>
        <dd>
          <code>{data.verifier}</code> ({data.envelopeSchema})
        </dd>
      </dl>
      <CaseTable cases={data.cases} whitespace={data.whitespace} />
      <OraclePanel rows={data.oracle} />
      <EnvelopeFileCheck whitespace={data.whitespace} />
      <p>{data.claimBoundary}.</p>
    </article>
  );
}
