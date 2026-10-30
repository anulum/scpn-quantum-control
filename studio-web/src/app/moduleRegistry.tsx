// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — the modules compiled into the workbench

import type { Loaded, StudioManifestView } from "../shared/browser/data";
import { createModuleRegistry } from "../shared/contracts/module";
import type { ModuleRequirements, StudioModule } from "../shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../shared/contracts/workbench";
export type { WorkbenchServices } from "../shared/contracts/workbench";
import { workspaceModule } from "../modules/workspace";
import { buildModule } from "../modules/build";
import { experimentsModule } from "../modules/experiments";
import { resultsModule } from "../modules/results";
import { atlasModule } from "../modules/atlas";
import { operatorsModule } from "../modules/operators";
import { evidenceModule } from "../modules/evidence";
import { programsModule } from "../modules/programs";
import { compilerModule } from "../modules/compiler";
import { workflowsModule } from "../modules/workflows";
import { compareModule } from "../modules/compare";
import { catalogueModule } from "../modules/catalogue";
import { parametersModule } from "../modules/parameters";
import { settingsModule } from "../modules/settings";

/** Sole build-time list of complete workbench modules; contribution ownership stays in their folders. */
export const workbenchModules = [
  workspaceModule,
  buildModule,
  experimentsModule,
  resultsModule,
  atlasModule,
  operatorsModule,
  evidenceModule,
  programsModule,
  compilerModule,
  workflowsModule,
  compareModule,
  catalogueModule,
  parametersModule,
  settingsModule,
] as const satisfies readonly StudioModule<WorkbenchServices, WorkspaceViewServices>[];

/** Validated registry used by the shell, addresses and standalone composition. */
export const workbenchRegistry = createModuleRegistry<WorkbenchServices, WorkspaceViewServices>(
  workbenchModules,
);

/**
 * Read what an installed manifest declares, for comparison with module requirements.
 *
 * A manifest that failed its guard declares nothing: every module that
 * requires a verb or a schema is then unavailable, never assumed present.
 *
 * @param manifest - The loaded Studio manifest, verified or refused.
 * @returns The declared verb names and evidence schema identifiers.
 */
export function declaredBy(manifest: Loaded<StudioManifestView>): ModuleRequirements {
  if (!manifest.ok) return { verbs: [], schemas: [] };
  return {
    verbs: manifest.value.verbs.map((entry) => entry.verb),
    schemas: manifest.value.evidenceTypes,
  };
}
