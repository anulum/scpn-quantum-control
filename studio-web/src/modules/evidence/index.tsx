// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered evidence catalogue destination

import { lazy } from "react";
import type { StudioModule } from "../../shared/contracts/module";
import type { NoServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
/** Shell services this module uses; the compiler refuses any other. */
type Services = NoServices;

const EvidenceView = lazy(() => import("./EvidenceView"));
/** Committed capability, support, gradient-plan and scorecard surfaces, attestation checks and benchmarks. */
export const evidenceModule: StudioModule<Services, WorkspaceViewServices, "evidence"> = {
  id: "evidence",
  apiVersion: "2.2",
  requires: { verbs: [], schemas: [] },
  stages: [
    {
      id: "evidence" as const,
      label: "Evidence",
      group: "primary",
      instruments: ["attestation", "benchmarks"],
      render: (_services: Services, instrument: string | null) => (
        <EvidenceView
          focusAttestation={instrument === "attestation"}
          focusBenchmarks={instrument === "benchmarks"}
        />
      ),
    },
  ],
  capabilities: [
    {
      id: "verify-attestations",
      label: "Verify attested result packs in the browser",
      maturity: "stable",
      needs: [],
    },
    {
      id: "verify-benchmarks",
      label: "Verify and read the tier benchmark databank",
      maturity: "stable",
      needs: [],
    },
  ],
  commands: [
    {
      id: "open-evidence",
      label: "Open Evidence",
      action: { kind: "navigate", stage: "evidence", instrument: null },
    },
    {
      id: "open-attestation",
      label: "Verify attested result packs",
      action: { kind: "navigate", stage: "evidence", instrument: "attestation" },
    },
    {
      id: "open-benchmarks",
      label: "Open the benchmark databank",
      action: { kind: "navigate", stage: "evidence", instrument: "benchmarks" },
    },
  ],
};
