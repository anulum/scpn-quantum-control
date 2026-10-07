// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — lazy original evidence surfaces

import { useEffect, useRef } from "react";
import { ResultLoader } from "../../features/results/ResultLoader";
import type { ResultLoaderProps } from "../../features/results/ResultLoader";
import { EvidenceViewer } from "../../shared/evidence/EvidenceViewer";
import { ProgramADReplayCard } from "../../panel/ProgramADReplayCard";
import { SupportMatrixGrid } from "../../panel/SupportMatrixGrid";
import { GradientPlanExplanation } from "../../panel/GradientPlanExplanation";
import { ScorecardTable } from "../../panel/ScorecardTable";
import { Unverifiable } from "../../panel/Unverifiable";
import { programAdUnit } from "../../panel/programAd";
import { RunComparison } from "../../features/compare/RunComparison";
import type { RunComparisonProps } from "../../features/compare/RunComparison";
import { gradientPlanExplanations, scorecard, supportMatrix } from "../../panel/data";

/** Original evidence and replay owners retain their source identities and claim boundaries. */
export default function ResultsView({
  focusInstrument = false,
  plan = null,
  outcome = null,
  sourceJson,
  rawCodecs,
}: ResultLoaderProps &
  RunComparisonProps & {
    /** Focus the original replay target only after this lazy module has mounted. */
    focusInstrument?: boolean;
  }) {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focusInstrument && target.current !== null) target.current.focus();
  }, [focusInstrument]);
  return (
    <article className="qsp-panel">
      <h3>Results</h3>
      <p>
        Inspect source-bound evidence or replay the committed browser fixture. These results are not
        attached to a requested revision by navigation.
      </p>
      <ResultLoader plan={plan} outcome={outcome} />
      <RunComparison
        {...(sourceJson === undefined ? {} : { sourceJson })}
        {...(rawCodecs === undefined ? {} : { rawCodecs })}
      />
      <div id="/results/program-ad-replay" tabIndex={-1} ref={target}>
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
  );
}
