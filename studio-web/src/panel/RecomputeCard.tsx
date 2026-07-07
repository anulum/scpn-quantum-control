// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web XY-compile recompute card (WS-1)

import { useState } from "react";

import type { KernelRecompute, RecomputeUnit, RecomputeVerdict } from "./recompute";
import { fetchKernel, verifyRecomputeUnit } from "./recompute";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
export type KernelLoader = () => Promise<KernelRecompute>;

type CardState =
  | { readonly phase: "idle" }
  | { readonly phase: "running" }
  | { readonly phase: "done"; readonly verdict: RecomputeVerdict }
  | { readonly phase: "error"; readonly reason: string };

const DISPLAY_LABEL: Record<RecomputeVerdict["display"], string> = {
  match: "recomputed digest matches the signed claim",
  mismatch: "recomputed digest does NOT match — claim forged",
  unverifiable: "unverifiable",
};

/**
 * The XY-compile recompute card. Pressing recompute loads the WASM kernel,
 * replays the committed unit's input in the browser, and reports the verdict
 * at its true class — a forged digest reads `mismatch`, a stripped grade or
 * kernel rejection reads `unverifiable`, never a silent pass.
 */
export function RecomputeCard({
  unit,
  loadKernel = fetchKernel,
}: {
  unit: RecomputeUnit;
  loadKernel?: KernelLoader;
}) {
  const [state, setState] = useState<CardState>({ phase: "idle" });

  const run = async (): Promise<void> => {
    setState({ phase: "running" });
    try {
      const kernel = await loadKernel();
      setState({ phase: "done", verdict: verifyRecomputeUnit(unit, kernel) });
    } catch (error) {
      const reason = error instanceof Error ? error.message : "kernel load failed";
      setState({ phase: "error", reason });
    }
  };

  return (
    <section className="qsp-recompute">
      <h3>Compile recompute (WS-1)</h3>
      <p className="qsp-meta">
        Bit-exact <code>{unit.exactnessClass}</code> compile digest, replayed in
        your browser through the WASM kernel. Signed claim:{" "}
        <code className="qsp-digest">{unit.claimedDigest}</code>
      </p>
      <button type="button" onClick={run} disabled={state.phase === "running"}>
        {state.phase === "running" ? "Recomputing…" : "Recompute in browser"}
      </button>
      {state.phase === "done" && (
        <p
          className={`qsp-badge qsp-badge-${verdictClass(state.verdict.display)}`}
          role="status"
        >
          {DISPLAY_LABEL[state.verdict.display]}
          {state.verdict.reason ? ` — ${state.verdict.reason}` : ""}
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

function verdictClass(display: RecomputeVerdict["display"]): string {
  return display === "match" ? "boundary" : display === "mismatch" ? "unverifiable" : "unverifiable";
}
