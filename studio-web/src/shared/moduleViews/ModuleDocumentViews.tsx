// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered document contribution rendering

import type {
  ModuleDocument,
  ModuleDocumentView,
  ModuleEditor,
  ModuleRegistry,
  ModuleRequirements,
} from "../contracts/module";
import { ContributionBoundary } from "./ContributionBoundary";

/** Document contribution slots that preserve source bytes and grant no job authority. */
export type DocumentModuleSlot = "editors" | "inspectors" | "settings";

/** Render schema-matched contributions or retain an unavailable document visibly. */
export interface ModuleDocumentViewsProps<Services, DocumentServices = Services> {
  /** The validated registry compiled into this build. */
  readonly registry: ModuleRegistry<Services, DocumentServices>;
  /** Fixed shell slot receiving the views. */
  readonly slot: DocumentModuleSlot;
  /** Original admitted schema, identity and content. */
  readonly document: ModuleDocument;
  /** Existing source and lifecycle owners, never imported from document contents. */
  readonly services: DocumentServices;
  /** Capabilities declared by the installation's guarded manifest. */
  readonly declared: ModuleRequirements;
}

/**
 * Render only views registered for the exact document schema and available in this installation.
 *
 * An unavailable module is named without executing its renderer. Unknown
 * inspector schemas keep a generic read-only view of the original bytes; a
 * missing editor or settings module keeps the document and reports absence.
 * Each view renders inside its own boundary, keyed by the document identity, so
 * a failing view is contained and a newly selected document retries it.
 *
 * @param props - Registered slot, unchanged admitted document and existing shell services.
 * @returns Available projections or the explicit retained-document fallback.
 */
export function ModuleDocumentViews<Services, DocumentServices = Services>({
  registry,
  slot,
  document,
  services,
  declared,
}: ModuleDocumentViewsProps<Services, DocumentServices>) {
  const contributions: readonly ModuleEditor<Services, DocumentServices>[] = registry[slot];
  const views = contributions.filter(
    (view): view is ModuleDocumentView<DocumentServices> =>
      view.kind === "document" && view.schemas.includes(document.schema),
  );
  if (views.length === 0) {
    return slot === "inspectors" ? (
      <section aria-label="Generic evidence inspector">
        <h3>Evidence document</h3>
        <dl>
          <dt>Schema</dt>
          <dd>{document.schema}</dd>
          <dt>Original identity</dt>
          <dd>{document.sha256}</dd>
        </dl>
        <label>
          Original document content
          <textarea readOnly value={document.content} spellCheck={false} />
        </label>
      </section>
    ) : (
      <p role="note">
        No {slot === "editors" ? "editor" : "settings view"} is registered for {document.schema}.
        The original document is retained.
      </p>
    );
  }
  const missing = registry.missing(declared);
  return (
    <>
      {views.map((view) => {
        const owner = registry.contributionOwner(slot, view.id);
        const lacking = missing.filter((requirement) => requirement.module === owner?.id);
        return owner === undefined || lacking.length > 0 ? (
          <section key={view.id} aria-label={`${view.label} unavailable`}>
            <h3>{view.label} unavailable</h3>
            {owner === undefined && <p>Registered contribution owner unavailable</p>}
            <p>The installation does not declare:</p>
            <ul>
              {lacking.map(({ kind, name }) => (
                <li key={`${kind}:${name}`}>
                  {kind} <code>{name}</code>
                </li>
              ))}
            </ul>
            <p>The original document is retained.</p>
          </section>
        ) : (
          <ContributionBoundary
            key={`${view.id}:${document.sha256}`}
            label={view.label}
            owner={owner.id}
            render={() => view.render(services, document)}
          />
        );
      })}
    </>
  );
}
