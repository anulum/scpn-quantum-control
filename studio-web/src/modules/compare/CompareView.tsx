// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Compare stage of the research loop

import type { ReactNode } from "react";

/**
 * The Compare stage: registered comparison instruments, each bound to its own sources.
 *
 * @param props - Comparison controls contributed to this stage through the shell slot.
 * @returns The stage.
 */
export default function CompareView({
  editors,
}: {
  /** Registered comparison controls supplied by the shell slot. */
  readonly editors: ReactNode;
}) {
  return (
    <article className="qsp-panel">
      <h3>Compare</h3>
      <p>
        Compare immutable runs of one source, or methods across a committed case population. Each
        comparison names its sources; nothing here runs a new job.
      </p>
      {editors}
    </article>
  );
}
