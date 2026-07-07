// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web schema-A capabilities view

import type { StudioManifestView } from "./data";

/**
 * Verbatim rendering of the committed schema-A capability manifest: the
 * verbs this studio advertises and the evidence schemas they produce.
 */
export function ManifestCapabilities({ manifest }: { manifest: StudioManifestView }) {
  return (
    <section className="qsp-capabilities">
      <h3>Capabilities</h3>
      <p className="qsp-meta">
        <code>{manifest.studio}</code> v{manifest.studioVersion} ·{" "}
        {manifest.verbs.length} verbs · {manifest.evidenceTypes.length} evidence schemas ·
        transport <code>{manifest.transportProfile}</code>
      </p>
      <table>
        <thead>
          <tr>
            <th scope="col">Verb</th>
            <th scope="col">Side effect</th>
            <th scope="col">Safety tier</th>
            <th scope="col">Timing</th>
            <th scope="col">Fidelity</th>
            <th scope="col">Produces</th>
          </tr>
        </thead>
        <tbody>
          {manifest.verbs.map((verb) => (
            <tr key={verb.verb}>
              <td>
                <code>{verb.verb}</code>
              </td>
              <td>{verb.sideEffect}</td>
              <td>{verb.safetyTier}</td>
              <td>{verb.timingClass}</td>
              <td>{verb.fidelity}</td>
              <td>
                {verb.produces.map((schema) => (
                  <code key={schema} className="qsp-schema">
                    {schema}
                  </code>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="qsp-digest">
        Declared-surface digest: <code>{manifest.contentDigest}</code>
      </p>
    </section>
  );
}
