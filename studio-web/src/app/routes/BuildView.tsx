// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — lazy source-owned browser instruments

import { useEffect, useRef } from "react";
import { KuramotoPlayPanel } from "../../panel/KuramotoPlayPanel";
import { Lab3DPanel } from "../../panel/Lab3DPanel";
import { RecomputeCard } from "../../panel/RecomputeCard";
import { Unverifiable } from "../../panel/Unverifiable";
import { committedScenario } from "../../panel/kuramoto";
import { recomputeUnit } from "../../panel/recompute";
import { ProgramEditor } from "../../features/programs/ProgramEditor";

/** Reuse original guarded instruments; no new solver, quota or provider policy. */
export default function BuildView({ focusInstrument = false }: {
  /** Focus the original compile target after its lazy module has actually mounted. */
  focusInstrument?: boolean;
}) {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => { if (focusInstrument && target.current !== null) target.current.focus(); }, [focusInstrument]);
  return <article className="qsp-panel">
    <h3>Build</h3>
    <ProgramEditor />
    <p>Browser instruments use their committed inputs and original resource admission. A requested workspace revision is not substituted into these fixtures.</p>
    <div id="/build/compile-recompute" tabIndex={-1} ref={target}>
      {recomputeUnit.ok ? <RecomputeCard unit={recomputeUnit.value} /> : <Unverifiable surface="xy_compile_recompute_unit_20260708.json" reason={recomputeUnit.reason} />}
    </div>
    {committedScenario.ok ? <><KuramotoPlayPanel scenario={committedScenario.value} /><Lab3DPanel scenario={committedScenario.value} /></> : <Unverifiable surface="kuramoto_scenario_meanfield_20260708.json" reason={committedScenario.reason} />}
  </article>;
}
