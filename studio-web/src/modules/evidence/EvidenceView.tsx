// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the Evidence stage: committed catalogue and attestation checks

import { useEffect, useRef } from "react";
import { AttestationInspector } from "./AttestationInspector";
import { attestationInstrument } from "./attestationInstrument";
import EvidenceCatalogueView from "./EvidenceCatalogueView";

/**
 * The evidence catalogue followed by the attestation instrument.
 *
 * @param props.focusAttestation - Move focus to the attestation instrument after mounting, for its deep link.
 */
export default function EvidenceView({ focusAttestation }: { readonly focusAttestation: boolean }) {
  const attestation = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focusAttestation && attestation.current !== null) attestation.current.focus();
  }, [focusAttestation]);
  return (
    <>
      <EvidenceCatalogueView />
      <div id="/evidence/attestation" tabIndex={-1} ref={attestation}>
        <AttestationInspector instrument={attestationInstrument} />
      </div>
    </>
  );
}
