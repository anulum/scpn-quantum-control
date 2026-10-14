// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered source explorer

import type { ModuleDocument, ModuleRegistry, ModuleRequirements } from "../contracts/module";

/** Registered project-tree contributions with admitted source selection. */
export interface ModuleExplorerViewProps<Services, DocumentServices = Services> {
  /** Build-time module registry. */ readonly registry: ModuleRegistry<Services, DocumentServices>;
  /** Original shell services used to project nodes. */ readonly services: Services;
  /** Guarded installation capabilities. */ readonly declared: ModuleRequirements;
  /** Source records from complete workspace admission. */ readonly documents: readonly ModuleDocument[];
  /** Currently selected original source identity. */ readonly selected: string | null;
  /** Open an existing admitted record; no save or execution occurs. */ readonly select: (
    document: ModuleDocument,
  ) => void;
}

/** Render available declared node kinds and refuse nodes detached from admitted source content. */
export function ModuleExplorerView<Services, DocumentServices = Services>({
  registry,
  services,
  declared,
  documents,
  selected,
  select,
}: ModuleExplorerViewProps<Services, DocumentServices>) {
  const missing = registry.missing(declared);
  return (
    <nav aria-label="Project explorer">
      {registry.explorer.map((explorer) => {
        const owner = registry.contributionOwner("explorer", explorer.id);
        const lacking = missing.filter((requirement) => requirement.module === owner?.id);
        if (owner === undefined || lacking.length > 0)
          return (
            <section key={explorer.id}>
              <h3>{explorer.label} unavailable</h3>
              {owner === undefined && <p>Registered explorer owner unavailable</p>}
              <p>{lacking.map(({ kind, name }) => `${kind} ${name}`).join(" · ")}</p>
            </section>
          );
        const nodes = explorer.nodes(services);
        const counts = new Map<string, number>();
        const occurrences = new Map<string, number>();
        for (const node of nodes) counts.set(node.id, (counts.get(node.id) ?? 0) + 1);
        return (
          <section key={explorer.id}>
            <h3>{explorer.label}</h3>
            {nodes.length === 0 ? (
              <p>No admitted source records</p>
            ) : (
              <ul>
                {nodes.map((node) => {
                  const occurrence = occurrences.get(node.id) ?? 0;
                  occurrences.set(node.id, occurrence + 1);
                  const source = documents.find(
                    (document) =>
                      document.sha256 === node.document.sha256 &&
                      document.schema === node.document.schema &&
                      document.content === node.document.content,
                  );
                  const available =
                    source !== undefined &&
                    counts.get(node.id) === 1 &&
                    explorer.kinds.includes(node.kind);
                  return (
                    <li key={`${explorer.id}:${node.id}:${node.document.sha256}:${occurrence}`}>
                      <button
                        type="button"
                        disabled={!available}
                        aria-pressed={selected === node.document.sha256}
                        onClick={
                          source !== undefined && available ? () => select(source) : undefined
                        }
                      >
                        {node.label}
                      </button>
                      {!available && <span> · Source unavailable</span>}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        );
      })}
    </nav>
  );
}
