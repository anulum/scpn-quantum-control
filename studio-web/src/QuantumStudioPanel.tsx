// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — QuantumStudioPanel (Module Federation expose)

import "./tokens.css";
import type { WorkspacePanelProps } from "./shared/workspace/WorkspaceEditor";
import { Workbench } from "./app/Workbench";
import type { WorkbenchMode } from "./app/Workbench";

import { GradientPlanExplanation } from "./shared/browser/GradientPlanExplanation";
import { KuramotoPlayPanel } from "./shared/browser/KuramotoPlayPanel";
import { Lab3DPanel } from "./shared/browser/Lab3DPanel";
import { ManifestCapabilities } from "./shared/browser/ManifestCapabilities";
import { EvidenceViewer } from "./shared/evidence/EvidenceViewer";
import { ProgramADReplayCard } from "./shared/browser/ProgramADReplayCard";
import { RecomputeCard } from "./shared/browser/RecomputeCard";
import { ScorecardTable } from "./shared/browser/ScorecardTable";
import { SupportMatrixGrid } from "./shared/browser/SupportMatrixGrid";
import { Unverifiable } from "./shared/browser/Unverifiable";
import {
  gradientPlanExplanations,
  scorecard,
  studioManifest,
  supportMatrix,
} from "./shared/browser/data";
import { committedScenario } from "./shared/browser/kuramoto";
import { programAdUnit } from "./shared/browser/programAd";
import { recomputeUnit } from "./shared/browser/recompute";

/** Original host producer contract with an additive layout mode. */
export interface QuantumStudioPanelProps extends Pick<WorkspacePanelProps, "rawCodecs"> {
  /** Standalone by default; embedded adapts the shell to its host's available width. */
  readonly mode?: WorkbenchMode;
}

/**
 * The QUANTUM studio panel the Hub mounts through Module Federation.
 *
 * Phase 0 renders the committed evidence surfaces verbatim: the schema-A
 * capability manifest, the transform-algebra support matrix, and the
 * baseline scorecard — each at its true claim boundary. Empty but alive,
 * never dishonest: nothing here can upgrade a grade, and any surface that
 * fails its guard renders as a loud `unverifiable` block.
 */
export function QuantumStudioPanel({ rawCodecs, mode }: QuantumStudioPanelProps = {}) {
  return (
    <Workbench
      {...(rawCodecs === undefined ? {} : { rawCodecs })}
      {...(mode === undefined ? {} : { mode })}
    >
      {(_context, editors) => (
        <article className="qsp-panel" aria-label="Committed evidence overview">
          <header className="qsp-header">
            <h2>SCPN QUANTUM CONTROL</h2>
            <p className="qsp-banner">
              Committed bounded-model evidence rendered at its boundary. Blocked rows are explicit
              fail-closed boundaries, not failures.
            </p>
          </header>
          {studioManifest.ok ? (
            <ManifestCapabilities manifest={studioManifest.value} />
          ) : (
            <Unverifiable surface="studio_manifest.json" reason={studioManifest.reason} />
          )}
          {editors}
          <div id="/build/compile-recompute" tabIndex={-1}>
            {recomputeUnit.ok ? (
              <RecomputeCard unit={recomputeUnit.value} />
            ) : (
              <Unverifiable
                surface="xy_compile_recompute_unit_20260708.json"
                reason={recomputeUnit.reason}
              />
            )}
          </div>
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
          <div id="/results/program-ad-replay" tabIndex={-1}>
            {programAdUnit.ok ? (
              <ProgramADReplayCard unit={programAdUnit.value} />
            ) : (
              <Unverifiable
                surface="program_ad_replay_rational_20260714.json"
                reason={programAdUnit.reason}
              />
            )}
          </div>
          <EvidenceViewer />
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
      )}
    </Workbench>
  );
}

export default QuantumStudioPanel;
