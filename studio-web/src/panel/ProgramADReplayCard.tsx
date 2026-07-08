// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web program-AD gradient replay card (ST-12)

import { useState } from "react";

import type { KernelReplay, ProgramAdUnit, ReplayVerdict } from "./programAd";
import { fetchProgramAd, verifyProgramAdUnit } from "./programAd";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
export type ReplayLoader = () => Promise<KernelReplay>;

type CardState =
  | { readonly phase: "idle" }
  | { readonly phase: "running" }
  | { readonly phase: "done"; readonly verdict: ReplayVerdict }
  | { readonly phase: "error"; readonly reason: string };

const DISPLAY_LABEL: Record<ReplayVerdict["display"], string> = {
  match: "recomputed value + gradient match the committed claim",
  mismatch: "recomputed gradient does NOT match — claim forged",
  unverifiable: "unverifiable",
};

/**
 * The program-AD gradient replay card. Pressing replay loads the standalone
 * program-AD WASM kernel, recomputes the committed rational program's gradient
 * in the browser, and reports the verdict at its true class — a forged gradient
 * reads `mismatch`, a wrong schema or kernel rejection reads `unverifiable`,
 * never a silent pass. The bounded claim boundary is shown verbatim.
 */
export function ProgramADReplayCard({
  unit,
  loadKernel = fetchProgramAd,
}: {
  unit: ProgramAdUnit;
  loadKernel?: ReplayLoader;
}) {
  const [state, setState] = useState<CardState>({ phase: "idle" });

  const run = async (): Promise<void> => {
    setState({ phase: "running" });
    try {
      const kernel = await loadKernel();
      setState({ phase: "done", verdict: verifyProgramAdUnit(unit, kernel) });
    } catch (error) {
      const reason = error instanceof Error ? error.message : "kernel load failed";
      setState({ phase: "error", reason });
    }
  };

  return (
    <section className="qsp-program-ad">
      <h3>Program-AD gradient replay (ST-12)</h3>
      <p className="qsp-meta">
        Reverse-mode gradient of the committed rational program, recomputed in
        your browser through the shipped Rust replay. Claimed value{" "}
        <strong>{unit.expectedValue}</strong>, gradient{" "}
        <code>[{unit.expectedGradient.join(", ")}]</code> over{" "}
        <code>{unit.parameterTargets.join(", ")}</code>.
      </p>
      <p className="qsp-meta qsp-boundary">{unit.claimBoundary}</p>
      <button type="button" onClick={run} disabled={state.phase === "running"}>
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
