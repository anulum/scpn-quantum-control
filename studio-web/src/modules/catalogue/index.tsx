// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered capability catalogue

import catalogueJson from "../../../../docs/_generated/studio_manifest.json";
import { studioManifest } from "../../shared/browser/data";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import type { StudioModule } from "../../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../../shared/contracts/workbench";
import { CapabilityCatalogue } from "./CapabilityCatalogue";
import { parseCatalogue } from "./catalogue";
import { useCatalogueRuntimes } from "./useCatalogueRuntimes";

/** Original catalogue admission and runtime projections with context-preserving links. */
export function CatalogueView({
  contextSuffix,
}: {
  /** Validated shell URL context. */ readonly contextSuffix: string;
}) {
  const runtimes = useCatalogueRuntimes();
  const catalogue = parseCatalogue(catalogueJson);
  return catalogue.ok && studioManifest.ok ? (
    <CapabilityCatalogue
      catalogue={catalogue.value}
      manifest={studioManifest.value}
      runtimes={runtimes}
      routeHref={(href) => href + contextSuffix}
    />
  ) : (
    <Unverifiable
      surface="capability catalogue"
      reason="Catalogue or source manifest unavailable"
    />
  );
}

/** Capability catalogue contributed to the original workspace overview position. */
export const catalogueModule: StudioModule<WorkbenchServices, WorkspaceViewServices, never> = {
  id: "catalogue",
  version: "1",
  requires: { verbs: [], schemas: [] },
  stages: [],
  editors: [
    {
      kind: "stage",
      id: "capability-catalogue",
      label: "Capability catalogue",
      stage: "workspace",
      render: (services: WorkbenchServices) => (
        <CatalogueView contextSuffix={services.contextSuffix} />
      ),
    },
  ],
};
