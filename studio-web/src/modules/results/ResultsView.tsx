// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — lazy original evidence surfaces

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { ResultLoader } from "./ResultLoader";
import type { ResultLoaderProps } from "./ResultLoader";
import { EvidenceViewer } from "../../shared/evidence/EvidenceViewer";
import { ProgramADReplayCard } from "../../shared/browser/ProgramADReplayCard";
import { SupportMatrixGrid } from "../../shared/browser/SupportMatrixGrid";
import { GradientPlanExplanation } from "../../shared/browser/GradientPlanExplanation";
import { ScorecardTable } from "../../shared/browser/ScorecardTable";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { programAdUnit } from "../../shared/browser/programAd";
import { gradientPlanExplanations, scorecard, supportMatrix } from "../../shared/browser/data";

/** Original evidence and replay owners retain their source identities and claim boundaries. */
export default function ResultsView({
  focusInstrument = false,
  plan = null,
  outcome = null,
  editors,
}: ResultLoaderProps & {
  /** Focus the original replay target only after this lazy module has mounted. */
  focusInstrument?: boolean;
  /** Registered comparison and inspection controls supplied by the shell slot. */
  editors?: ReactNode;
}) {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focusInstrument && target.current !== null) target.current.focus();
  }, [focusInstrument]);
  return (
    <article className="qsp-panel">
      <h3>Analyse</h3>
      <p>
        Inspect source-bound evidence or replay the committed browser fixture. These results are not
        attached to a requested revision by navigation.
      </p>
      <ResultLoader plan={plan} outcome={outcome} />
      {editors}
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
