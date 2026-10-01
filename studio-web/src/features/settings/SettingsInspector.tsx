// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-owned settings inspector

import { writeJson } from "../../shared/contracts/jsonTransport";
import { settingsFromArchive } from "../../shared/contracts/settings";
import type { WorkspaceArchivePreview } from "../../shared/storage/workspaceArchive";

/** Exact admitted portable workspace currently displayed by the original editor. */
export interface SettingsInspectorProps {
  /** Immutable archive preview; no draft text or current provider credentials. */
  readonly preview: WorkspaceArchivePreview;
}

/** Show requested/effective values and original provenance without executing a policy. */
export function SettingsInspector({ preview }: SettingsInspectorProps) {
  const settings = settingsFromArchive(preview);
  if (!settings.ok) return <p role="alert">{settings.message}</p>;
  return <section aria-label="Resolved settings provenance">
    <h4>Resolved settings</h4>
    <p>Recorded source values; this inspector grants no execution or provider authority.</p>
    {settings.value.length === 0 && <p>No resolved settings records in this workspace.</p>}
    {settings.value.map(({ digest, document }) => {
      const requested = document.body["requested"] as Readonly<Record<string, unknown>>;
      const effective = document.body["effective"] as Readonly<Record<string, unknown>>;
      const origins = document.body["origins"] as Readonly<Record<string, unknown>>;
      return <article key={digest} aria-label={`Settings ${digest}`}>
        <p>Settings digest: <code>{digest}</code></p>
        <p>Policy reference: <code>{writeJson(document.body["policy_ref"])}</code></p>
        <p>Environment reference: <code>{writeJson(document.body["environment_ref"])}</code></p>
        <table><caption>Requested and effective settings</caption>
          <thead><tr><th scope="col">Field</th><th scope="col">Requested</th><th scope="col">Effective</th><th scope="col">Origin</th></tr></thead>
          <tbody>{Object.keys(effective).sort().map(key => <tr key={key}>
            <th scope="row">{key}</th><td>{Object.hasOwn(requested, key) ? writeJson(requested[key]) : "Not recorded"}</td>
            <td>{writeJson(effective[key])}</td><td>{writeJson(origins[key])}</td>
          </tr>)}</tbody>
        </table>
        <p>Rejected fields: {writeJson(document.body["rejected_fields"])}</p>
      </article>;
    })}
  </section>;
}
