// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — lazy source-owned browser instruments

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { KuramotoPlayPanel } from "../../shared/browser/KuramotoPlayPanel";
import { Lab3DPanel } from "../../shared/browser/Lab3DPanel";
import { RecomputeCard } from "../../shared/browser/RecomputeCard";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { committedScenario } from "../../shared/browser/kuramoto";
import { recomputeUnit } from "../../shared/browser/recompute";

/** Reuse original guarded instruments; no new solver, quota or provider policy. */
export default function BuildView({
  focusInstrument = false,
  editors,
}: {
  /** Focus the original compile target after its lazy module has actually mounted. */
  focusInstrument?: boolean;
  /** Registered source editors supplied by the shell slot. */
  editors?: ReactNode;
}) {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focusInstrument && target.current !== null) target.current.focus();
  }, [focusInstrument]);
  return (
    <article className="qsp-panel qsp-build">
      <h3>Build</h3>
      <p className="qsp-meta">
        Browser instruments use their committed inputs and original resource admission. A requested
        workspace revision is not substituted into these fixtures.
      </p>
      <div className="qsp-build-grid">
        <div className="qsp-build-main">
          <div className="qsp-card">{editors}</div>
          {committedScenario.ok && (
            <div className="qsp-card">
              <Lab3DPanel scenario={committedScenario.value} />
            </div>
          )}
        </div>
        <div className="qsp-build-side">
          <div className="qsp-card">
            {committedScenario.ok ? (
              <KuramotoPlayPanel scenario={committedScenario.value} />
            ) : (
              <Unverifiable
                surface="kuramoto_scenario_meanfield_20260708.json"
                reason={committedScenario.reason}
              />
            )}
          </div>
          <div className="qsp-card" id="/build/compile-recompute" tabIndex={-1} ref={target}>
            {recomputeUnit.ok ? (
              <RecomputeCard unit={recomputeUnit.value} />
            ) : (
              <Unverifiable
                surface="xy_compile_recompute_unit_20260708.json"
                reason={recomputeUnit.reason}
              />
            )}
          </div>
        </div>
      </div>
    </article>
  );
}
