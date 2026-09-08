// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web XY-compile recompute card

import type { KernelRecompute, RecomputeUnit, RecomputeVerdict } from "./recompute";
import { fetchKernel, verifyRecomputeUnit } from "./recompute";
import { useUnitBoundRun } from "./useUnitBoundRun";

/** Loader for the WASM kernel; overridable so tests inject a built kernel. */
export type KernelLoader = () => Promise<KernelRecompute>;

const DISPLAY_LABEL: Record<RecomputeVerdict["display"], string> = {
  match: "recomputed digest matches the signed claim",
  mismatch: "recomputed digest does NOT match the signed claim",
  unverifiable: "unverifiable",
};

/**
 * Identity of the unit a verdict belongs to.
 *
 * Include schema, verification mode and exactness alongside digest and payload.
 * A claim with a changed verification contract cannot reuse a previous verdict.
 *
 * @param unit - The unit currently displayed.
 * @returns A stable identity string for that unit.
 */
function unitIdentity(unit: RecomputeUnit): string {
  return JSON.stringify([
    unit.schema,
    unit.verifiabilityMode,
    unit.exactnessClass,
    unit.claimedDigest,
    unit.inputHex,
  ]);
}

/**
 * The XY-compile recompute card. Pressing recompute loads the WASM kernel,
 * replays the committed unit's input in the browser, and reports the verdict
 * at its true class — a digest that disagrees with the claim reads `mismatch`,
 * a stripped grade or kernel rejection reads `unverifiable`, never a silent
 * pass.
 *
 * The verdict is bound to the unit's identity. Replacing the `unit` prop clears
 * a displayed verdict before the new claim is painted, and a recompute already
 * in flight for the previous unit is discarded when it resolves rather than
 * shown beside the new one.
 */
export function RecomputeCard({
  unit,
  loadKernel = fetchKernel,
}: {
  unit: RecomputeUnit;
  loadKernel?: KernelLoader;
}) {
  const { state, run } = useUnitBoundRun<RecomputeVerdict>(unitIdentity(unit));

  const recompute = async (): Promise<void> => {
    await run(async () => verifyRecomputeUnit(unit, await loadKernel()), "kernel load failed");
  };

  return (
    <section className="qsp-recompute">
      <h3>Compile recompute</h3>
      <p className="qsp-meta">
        Bit-exact <code>{unit.exactnessClass}</code> compile digest, replayed in
        your browser through the WASM kernel. Signed claim:{" "}
        <code className="qsp-digest">{unit.claimedDigest}</code>
      </p>
      <button type="button" onClick={recompute} disabled={state.phase === "running"}>
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
