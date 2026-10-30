// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered stage editor rendering

import type { ModuleRegistry, ModuleRequirements, ModuleStageEditor } from "../contracts/module";
import { ContributionBoundary } from "./ContributionBoundary";

/** Fixed work-area slot with the build's registry and existing stage services. */
export interface ModuleStageEditorsProps<Services, DocumentServices = Services> {
  /** Validated modules compiled into the application. */
  readonly registry: ModuleRegistry<Services, DocumentServices>;
  /** Stage whose editor contributions are being displayed. */
  readonly stage: string;
  /** Original source and lifecycle owners. */
  readonly services: Services;
  /** Guarded installation declarations. */
  readonly declared: ModuleRequirements;
}

/**
 * Render available stage editors in registration order without calling unavailable modules.
 *
 * Each editor renders inside its own boundary, so one failing or loading editor
 * leaves the others on the stage working.
 */
export function ModuleStageEditors<Services, DocumentServices = Services>({
  registry,
  stage,
  services,
  declared,
}: ModuleStageEditorsProps<Services, DocumentServices>) {
  const editors = registry.editors.filter(
    (editor): editor is ModuleStageEditor<Services> =>
      editor.kind === "stage" && editor.stage === stage,
  );
  const missing = registry.missing(declared);
  return (
    <>
      {editors.map((editor) => {
        const owner = registry.contributionOwner("editors", editor.id);
        const lacking = missing.filter((requirement) => requirement.module === owner?.id);
        return owner === undefined || lacking.length > 0 ? (
          <section key={editor.id} aria-label={`${editor.label} unavailable`}>
            <h3>{editor.label} unavailable</h3>
            {owner === undefined && <p>Registered contribution owner unavailable</p>}
            <ul>
              {lacking.map(({ kind, name }) => (
                <li key={`${kind}:${name}`}>
                  {kind} <code>{name}</code>
                </li>
              ))}
            </ul>
          </section>
        ) : (
          <ContributionBoundary
            key={editor.id}
            label={editor.label}
            owner={owner.id}
            render={() => editor.render(services)}
          />
        );
      })}
    </>
  );
}
