// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel (Module Federation expose)

import "./tokens.css";

import { ManifestCapabilities } from "./panel/ManifestCapabilities";
import { ScorecardTable } from "./panel/ScorecardTable";
import { SupportMatrixGrid } from "./panel/SupportMatrixGrid";
import { Unverifiable } from "./panel/Unverifiable";
import { scorecard, studioManifest, supportMatrix } from "./panel/data";

/**
 * The QUANTUM studio panel the Hub mounts through Module Federation.
 *
 * Phase 0 renders the committed evidence surfaces verbatim: the schema-A
 * capability manifest, the transform-algebra support matrix, and the
 * baseline scorecard — each at its true claim boundary. Empty but alive,
 * never dishonest: nothing here can upgrade a grade, and any surface that
 * fails its guard renders as a loud `unverifiable` block.
 */
export function QuantumStudioPanel() {
  return (
    <article className="qsp-panel">
      <header className="qsp-header">
        <h2>SCPN QUANTUM CONTROL</h2>
        <p className="qsp-banner">
          Committed bounded-model evidence rendered at its boundary. Blocked rows
          are explicit fail-closed boundaries, not failures.
        </p>
      </header>
      {studioManifest.ok ? (
        <ManifestCapabilities manifest={studioManifest.value} />
      ) : (
        <Unverifiable surface="studio_manifest.json" reason={studioManifest.reason} />
      )}
      {supportMatrix.ok ? (
        <SupportMatrixGrid matrix={supportMatrix.value} />
      ) : (
        <Unverifiable
          surface="differentiable_transform_support_matrix_20260708.json"
          reason={supportMatrix.reason}
        />
      )}
      {scorecard.ok ? (
        <ScorecardTable scorecard={scorecard.value} />
      ) : (
        <Unverifiable
          surface="differentiable_baseline_scorecard_20260620.json"
          reason={scorecard.reason}
        />
      )}
    </article>
  );
}

export default QuantumStudioPanel;
