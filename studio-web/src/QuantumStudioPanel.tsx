// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel (Module Federation expose)

import "./tokens.css";

import { GradientPlanExplanation } from "./panel/GradientPlanExplanation";
import { KuramotoPlayPanel } from "./panel/KuramotoPlayPanel";
import { Lab3DPanel } from "./panel/Lab3DPanel";
import { ManifestCapabilities } from "./panel/ManifestCapabilities";
import { ProgramADReplayCard } from "./panel/ProgramADReplayCard";
import { RecomputeCard } from "./panel/RecomputeCard";
import { ScorecardTable } from "./panel/ScorecardTable";
import { SupportMatrixGrid } from "./panel/SupportMatrixGrid";
import { Unverifiable } from "./panel/Unverifiable";
import {
  gradientPlanExplanations,
  scorecard,
  studioManifest,
  supportMatrix,
} from "./panel/data";
import { committedScenario } from "./panel/kuramoto";
import { programAdUnit } from "./panel/programAd";
import { recomputeUnit } from "./panel/recompute";

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
      {recomputeUnit.ok ? (
        <RecomputeCard unit={recomputeUnit.value} />
      ) : (
        <Unverifiable
          surface="xy_compile_recompute_unit_20260708.json"
          reason={recomputeUnit.reason}
        />
      )}
      {committedScenario.ok ? (
        <KuramotoPlayPanel scenario={committedScenario.value} />
      ) : (
        <Unverifiable
          surface="kuramoto_scenario_meanfield_20260708.json"
          reason={committedScenario.reason}
        />
      )}
      {committedScenario.ok ? (
        <Lab3DPanel scenario={committedScenario.value} />
      ) : (
        <Unverifiable
          surface="kuramoto_scenario_meanfield_20260708.json"
          reason={committedScenario.reason}
        />
      )}
      {programAdUnit.ok ? (
        <ProgramADReplayCard unit={programAdUnit.value} />
      ) : (
        <Unverifiable
          surface="program_ad_replay_rational_20260714.json"
          reason={programAdUnit.reason}
        />
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

export default QuantumStudioPanel;
