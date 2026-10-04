// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned devices and operations
import { BackendProfiles } from "../../features/operators/profiles/BackendProfiles";
import { PolicyInspector } from "../../features/operators/policy/PolicyInspector";
import { OperatorDossier } from "../../features/operators/dossiers/OperatorDossier";
/** Reachable offline operator metadata; importing does not confer execution authority. */
export default function OperationsView() {
  return <article className="qsp-panel"><h3>Devices &amp; Operations</h3><BackendProfiles /><PolicyInspector /><OperatorDossier /></article>;
}
