// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — registered command search and dispatch

import { useState } from "react";
import type { ModuleCommandAction, ModuleRegistry, ModuleRequirements } from "../contracts/module";

/** Searchable compiled commands with shell-owned navigation and job dispatch. */
export interface ModuleCommandSearchProps<Services, DocumentServices = Services> {
  /** Actual build-time registry. */ readonly registry: ModuleRegistry<Services, DocumentServices>;
  /** Guarded installation capabilities. */ readonly declared: ModuleRequirements;
  /** Existing shell action handler; commands supply no permission callback. */ readonly dispatch: (
    action: ModuleCommandAction,
  ) => void;
}

/** Show matching commands and named missing requirements without executing unavailable modules. */
export function ModuleCommandSearch<Services, DocumentServices = Services>({
  registry,
  declared,
  dispatch,
}: ModuleCommandSearchProps<Services, DocumentServices>) {
  const [query, setQuery] = useState("");
  const missing = registry.missing(declared);
  const commands = registry.commands.filter((command) =>
    command.label.toLocaleLowerCase().includes(query.toLocaleLowerCase()),
  );
  return (
    <nav aria-label="Workbench commands">
      <label>
        Search commands
        <input value={query} maxLength={512} onChange={(event) => setQuery(event.target.value)} />
      </label>
      {commands.length === 0 ? (
        <p>No matching commands</p>
      ) : (
        <ul>
          {commands.map((command) => {
            const owner = registry.contributionOwner("commands", command.id);
            const lacking = missing.filter((requirement) => requirement.module === owner?.id);
            return (
              <li key={command.id}>
                <button
                  type="button"
                  disabled={owner === undefined || lacking.length > 0}
                  onClick={
                    owner !== undefined && lacking.length === 0
                      ? () => dispatch(command.action)
                      : undefined
                  }
                >
                  {command.label}
                </button>
                {owner === undefined && <span> · Registered command owner unavailable</span>}
                {lacking.length > 0 && (
                  <span>
                    {" "}
                    · Unavailable: {lacking.map(({ kind, name }) => `${kind} ${name}`).join(" · ")}
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </nav>
  );
}
