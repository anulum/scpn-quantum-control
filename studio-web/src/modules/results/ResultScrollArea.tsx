// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — keyboard entry into an independently scrollable raw table

import { ScrollRegion } from "../../shared/browser/ScrollRegion";
import type { ReactNode } from "react";

/** Original raw table and its source-specific scrolling name. */
export interface ResultScrollAreaProps {
  /** Original source panel title. */ readonly title: string;
  /** Original admitted raw table. */ readonly children: ReactNode;
}

/** A native keyboard button focuses the scrolling region without adding noninteractive tab stops. */
export function ResultScrollArea({ title, children }: ResultScrollAreaProps) {
  return (
    <ScrollRegion
      className="qsp-data-scroll"
      name={`${title} raw table scrolling`}
      action={`Scroll ${title} raw values`}
    >
      {children}
    </ScrollRegion>
  );
}
