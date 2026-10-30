// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — lazy source-owned browser instruments

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { Lab3DPanel } from "../../shared/browser/Lab3DPanel";
import { RecomputeCard } from "../../shared/browser/RecomputeCard";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import { committedScenario } from "../../shared/browser/kuramoto";
import { recomputeUnit } from "../../shared/browser/recompute";
import { BuildWorkbench } from "./BuildWorkbench";

/** Reuse original guarded instruments; no new solver, quota or provider policy. */
export default function BuildView({
  focusInstrument = false,
  editors,
  compareHref = "#/results",
}: {
  /** Focus the original compile target after its lazy module has actually mounted. */
  focusInstrument?: boolean;
  /** Registered source editors supplied by the shell slot. */
  editors?: ReactNode;
  /** Results address with the current URL context, for comparing runs. */
  compareHref?: string;
}) {
  const target = useRef<HTMLElement>(null);
  useEffect(() => {
    if (focusInstrument && target.current !== null) target.current.focus();
  }, [focusInstrument]);
  const evidence = (
    <section
      className="qsp-card qsp-evidence-card"
      id="/build/compile-recompute"
      tabIndex={-1}
      ref={target}
      aria-label="Compile recomputation evidence"
    >
      {recomputeUnit.ok ? (
        <RecomputeCard unit={recomputeUnit.value} />
      ) : (
        <Unverifiable
          surface="xy_compile_recompute_unit_20260708.json"
          reason={recomputeUnit.reason}
        />
      )}
    </section>
  );
  return (
    <article className="qsp-panel qsp-build">
      <h3>Build</h3>
      <p className="qsp-meta">
        Browser instruments use their committed inputs and original resource admission. A requested
        workspace revision is not substituted into these fixtures.
      </p>
      {committedScenario.ok ? (
        <>
          <BuildWorkbench
            scenario={committedScenario.value}
            editors={editors}
            evidence={evidence}
            compareHref={compareHref}
          />
          <div className="qsp-card qsp-lab-card">
            <Lab3DPanel scenario={committedScenario.value} />
          </div>
        </>
      ) : (
        <div className="qsp-build-grid">
          <div className="qsp-build-main">
            <div className="qsp-card">{editors}</div>
          </div>
          <div className="qsp-build-side">
            <div className="qsp-card">
              <Unverifiable
                surface="kuramoto_scenario_meanfield_20260708.json"
                reason={committedScenario.reason}
              />
            </div>
            {evidence}
          </div>
        </div>
      )}
    </article>
  );
}
