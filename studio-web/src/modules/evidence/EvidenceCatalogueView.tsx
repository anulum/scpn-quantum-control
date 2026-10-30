// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — committed evidence catalogue destination

import { GradientPlanExplanation } from "../../shared/browser/GradientPlanExplanation";
import { ManifestCapabilities } from "./ManifestCapabilities";
import { ScorecardTable } from "../../shared/browser/ScorecardTable";
import { SupportMatrixGrid } from "../../shared/browser/SupportMatrixGrid";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import {
  gradientPlanExplanations,
  scorecard,
  studioManifest,
  supportMatrix,
} from "../../shared/browser/data";

/**
 * The committed evidence surfaces, each rendered verbatim at its claim boundary.
 *
 * The capability manifest, the transform support matrix, the gradient-plan
 * explanation and the baseline scorecard live here rather than on the
 * Workspace landing page. Any surface that fails its guard renders as a loud
 * `unverifiable` block; nothing here can upgrade a grade.
 */
export default function EvidenceCatalogueView() {
  return (
    <article className="qsp-panel" aria-label="Committed evidence catalogue">
      <h3>Evidence catalogue</h3>
      <p className="qsp-banner">
        Committed bounded-model evidence rendered at its boundary. Blocked rows are explicit
        fail-closed boundaries, not failures.
      </p>
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
      {gradientPlanExplanations.ok ? (
        <GradientPlanExplanation plans={gradientPlanExplanations.value} />
      ) : (
        <Unverifiable
          surface="gradient_plan_explanations_20260709.json"
          reason={gradientPlanExplanations.reason}
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
