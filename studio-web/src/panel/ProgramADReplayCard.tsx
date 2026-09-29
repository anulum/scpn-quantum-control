// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web program-AD gradient replay card


import type { KernelReplay, ProgramAdUnit, ReplayVerdict } from "./programAd";
import { fetchProgramAd, verifyProgramAdUnit } from "./programAd";
import { EvidenceInspector } from "../shared/evidence/EvidenceInspector";
import type { EvidenceVerification } from "../shared/evidence/EvidenceInspector";
import { projectProgramAdEvidence } from "../shared/evidence/evidence";

/** Loader for the original WASM replay kernel. */
export type ReplayLoader = () => Promise<KernelReplay>;

const DISPLAY_LABEL: Record<ReplayVerdict["display"], string> = {
  match: "recomputed value + gradient match the committed claim",
  mismatch: "recomputed value or gradient does NOT match the committed claim",
  unverifiable: "unverifiable",
};

/** Original replay source and its optional runtime/input-revision context. */
export interface ProgramADReplayCardProps {
  /** Complete producer-owned verification contract. */ readonly unit: ProgramAdUnit;
  /** Loader for the actual bounded WASM implementation. */ readonly loadKernel?: ReplayLoader;
  /** Current editor revision; defaults to the producer's input digest. */ readonly inputRevision?: string;
}

/** Recompute the original bounded replay and inspect its source-owned evidence. */
export function ProgramADReplayCard({
  unit,
  loadKernel = fetchProgramAd,
  inputRevision = unit.inputSha256,
}: ProgramADReplayCardProps) {
  const snapshot: ProgramAdUnit = { ...unit, expectedGradient: [...unit.expectedGradient], parameterTargets: [...unit.parameterTargets] };
  const replay = async (): Promise<EvidenceVerification> => {
    const verdict = await verifyProgramAdUnit(snapshot, await loadKernel());
    return { display: verdict.display, detail: DISPLAY_LABEL[verdict.display]
      + (verdict.recomputed ? ` — value ${verdict.recomputed.value}, gradient [${verdict.recomputed.gradient.join(", ")}]` : "")
      + (verdict.reason ? ` (${verdict.reason})` : "") };
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
      <EvidenceInspector view={projectProgramAdEvidence(snapshot)} inputRevision={inputRevision}
        verify={replay} actionLabel="Recompute gradient in browser" runningLabel="Recomputing…" failureReason="kernel load failed" />
    </section>
  );
}
