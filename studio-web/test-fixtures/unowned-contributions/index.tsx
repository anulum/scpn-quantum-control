// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — compiled negative fixture for unavailable contribution ownership

import type { StudioModule } from "../../src/shared/contracts/module";
import type {
  WorkbenchServices,
  WorkspaceViewServices,
} from "../../src/shared/contracts/workbench";

function refuseInvocation(): never {
  throw new Error("An unowned contribution was invoked");
}
/** Real compiled contributions whose callbacks must remain unreachable after owner lookup fails. */
export const unownedContributionsModule: StudioModule<
  WorkbenchServices,
  WorkspaceViewServices,
  never
> = {
  id: "unowned-contributions",
  apiVersion: "2.0",
  requires: { verbs: [], schemas: [] },
  stages: [],
  explorer: [
    {
      id: "unowned-explorer",
      label: "Unowned explorer",
      kinds: ["quantum_workspace_archive.v1"],
      nodes: refuseInvocation,
    },
  ],
  editors: [
    {
      kind: "stage",
      id: "unowned-stage-editor",
      label: "Unowned stage editor",
      stage: "workspace",
      render: refuseInvocation,
    },
  ],
  inspectors: [
    {
      kind: "document",
      id: "unowned-inspector",
      label: "Unowned inspector",
      schemas: ["quantum_workspace_archive.v1"],
      render: refuseInvocation,
    },
  ],
  settings: [
    {
      kind: "document",
      id: "unowned-settings",
      label: "Unowned settings",
      schemas: ["quantum_workspace_archive.v1"],
      render: refuseInvocation,
    },
  ],
  commands: [
    {
      id: "open-unowned-probe",
      label: "Open unowned probe",
      action: { kind: "navigate", stage: "workspace", instrument: null },
    },
  ],
};
