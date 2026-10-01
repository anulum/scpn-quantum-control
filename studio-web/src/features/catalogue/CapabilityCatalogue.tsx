// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — CapabilityCatalogue

import { useState } from "react";

import type { StudioManifestView } from "../../panel/data";
import { catalogueMatchesSource } from "./catalogue";
import type { Catalogue, RuntimeAvailability } from "./catalogue";

/** Read-only search over source declarations and measured browser kernel availability. */
export function CapabilityCatalogue({ catalogue, manifest, runtimes, routeHref }: {
  catalogue: Catalogue;
  manifest: StudioManifestView;
  runtimes: Readonly<Record<string, RuntimeAvailability>>;
  /** Shell-provided context encoding; the catalogue retains original runtime/source admission. */
  routeHref?: (route: string) => string;
}) {
  const [task, setTask] = useState("");
  const [runtime, setRuntime] = useState("");
  const [backend, setBackend] = useState("");
  const current = catalogueMatchesSource(catalogue, manifest);
  const rows = catalogue.rows.filter(row =>
    `${row.verb} ${row.label} ${row.api}`.toLowerCase().includes(task.trim().toLowerCase()) &&
    (runtime === "" || row.runtime === runtime) &&
    (backend === "" || row.backends.includes(backend)),
  );
  const backends = [...new Set(catalogue.rows.flatMap(row => row.backends))].sort();

  return (
    <section className="qsp-catalogue" aria-labelledby="catalogue-heading">
      <h3 id="catalogue-heading">Capability catalogue</h3>
      <p>
        Find a local instrument or a library entry. Listed backends are declarations;
        opening an instrument does not submit a job.
      </p>
      {!current && (
        <p role="alert">
          Source mismatch: this catalogue is stale or unknown for the installed source.
          Browser routes are disabled.
        </p>
      )}
      <div className="qsp-catalogue-filters">
        <label>
          Capability task
          <input type="search" value={task} onChange={event => setTask(event.target.value)} />
        </label>
        <label>
          Capability runtime
          <select value={runtime} onChange={event => setRuntime(event.target.value)}>
            <option value="">All runtimes</option>
            <option value="browser-wasm">Browser WASM</option>
            <option value="local-python">Local Python</option>
          </select>
        </label>
        <label>
          Capability backend
          <select value={backend} onChange={event => setBackend(event.target.value)}>
            <option value="">All declared backends</option>
            {backends.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => { setTask(""); setRuntime(""); setBackend(""); }}>
          Clear filters
        </button>
      </div>
      <p role="status">{rows.length} of {catalogue.rows.length} capabilities</p>
      {rows.length === 0 && <p>No capability matches these filters.</p>}
      <ul className="qsp-catalogue-rows">
        {rows.map(row => {
          const availability = runtimes[row.verb];
          const route = row.route;
          return (
            <li key={row.verb} data-testid={`capability-${row.verb}`}>
              <h4>{row.verb} · {row.label}</h4>
              <p>{row.reason}</p>
              <p>{row.settings}</p>
              <p>
                <code>{row.api}</code> · {row.runtime} · declared backends: {row.backends.join(", ")}
              </p>
              <details>
                <summary>Evidence schemas</summary>
                {row.evidence.map(schema => <p key={schema}><code>{schema}</code></p>)}
              </details>
              {route !== null && (
                current && availability?.available ? (
                  <a href={routeHref === undefined ? route : routeHref(route)} onClick={() => { document.getElementById(route.slice(1))?.focus(); }}>
                    Open {row.label}
                  </a>
                ) : (
                  <p className="qsp-boundary">
                    {!current ? "Unavailable for this source" : availability?.reason ?? "Backend availability unknown"}
                  </p>
                )
              )}
            </li>
          );
        })}
      </ul>
      <p>
        <a href="https://github.com/anulum/scpn-quantum-control/blob/main/docs/studio_federation.md">
          Evidence and federation contracts
        </a>
      </p>
      <p className="qsp-digest">Catalogue identity: <code>{catalogue.identity}</code></p>
    </section>
  );
}
