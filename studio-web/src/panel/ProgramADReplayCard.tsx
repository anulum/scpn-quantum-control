// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web program-AD gradient replay card

import type { KernelReplay, ProgramAdUnit, ReplayVerdict } from "./programAd";
import { fetchProgramAd, verifyProgramAdUnit } from "./programAd";
import { useUnitBoundRun } from "./useUnitBoundRun";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
export type ReplayLoader = () => Promise<KernelReplay>;

const DISPLAY_LABEL: Record<ReplayVerdict["display"], string> = {
  match: "recomputed value + gradient match the committed claim",
  mismatch: "recomputed value or gradient does NOT match the committed claim",
  unverifiable: "unverifiable",
};

/**
 * Identity of the unit a verdict belongs to.
 *
 * Include the complete verification contract, not only the artifact and input
 * digest. A changed expected result, schema or displayed boundary invalidates
 * an earlier verdict even when the artifact identifier has not changed.
 *
 * @param unit - The unit currently displayed.
 * @returns A stable identity string for that unit.
 */
function unitIdentity(unit: ProgramAdUnit): string {
  return JSON.stringify([
    unit.schema,
    unit.artifactId,
    unit.claimBoundary,
    unit.inputHex,
    unit.inputSha256,
    unit.expectedValue,
    unit.expectedGradient,
    unit.parameterTargets,
  ]);
}

/**
 * The program-AD gradient replay card. Pressing replay loads the standalone
 * program-AD WASM kernel, recomputes the committed rational program's gradient
 * in the browser, and reports the verdict at its true class — a gradient that
 * disagrees with the claim reads `mismatch`, a wrong schema or kernel rejection
 * reads `unverifiable`, never a silent pass. The bounded claim boundary is
 * shown verbatim.
 *
 * The verdict is bound to the unit's identity. Replacing the `unit` prop clears
 * a displayed verdict before the new claim is painted, and a replay already in
 * flight for the previous unit is discarded when it resolves rather than shown
 * beside the new one.
 */
export function ProgramADReplayCard({
  unit,
  loadKernel = fetchProgramAd,
}: {
  unit: ProgramAdUnit;
  loadKernel?: ReplayLoader;
}) {
  const { state, run } = useUnitBoundRun<ReplayVerdict>(unitIdentity(unit));

  const replay = async (): Promise<void> => {
    await run(async () => verifyProgramAdUnit(unit, await loadKernel()), "kernel load failed");
  };

  return (
    <section className="qsp-program-ad">
      <h3>Program-AD gradient replay</h3>
      <p className="qsp-meta">
        Reverse-mode gradient of the committed rational program, recomputed in
        your browser through the shipped Rust replay. Claimed value{" "}
        <strong>{unit.expectedValue}</strong>, gradient{" "}
        <code>[{unit.expectedGradient.join(", ")}]</code> over{" "}
        <code>{unit.parameterTargets.join(", ")}</code>.
      </p>
      <p className="qsp-meta qsp-boundary">{unit.claimBoundary}</p>
      <button type="button" onClick={replay} disabled={state.phase === "running"}>
        {state.phase === "running" ? "Recomputing…" : "Recompute gradient in browser"}
      </button>
      {state.phase === "done" && (
        <p
          className={`qsp-badge qsp-badge-${state.verdict.display === "match" ? "boundary" : "unverifiable"}`}
          role="status"
        >
          {DISPLAY_LABEL[state.verdict.display]}
          {state.verdict.recomputed
            ? ` — value ${state.verdict.recomputed.value}, gradient [${state.verdict.recomputed.gradient.join(", ")}]`
            : ""}
          {state.verdict.reason ? ` (${state.verdict.reason})` : ""}
        </p>
      )}
      {state.phase === "error" && (
        <p className="qsp-badge qsp-badge-unverifiable" role="alert">
          unverifiable — {state.reason}
        </p>
      )}
    </section>
  );
}
