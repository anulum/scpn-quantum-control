// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — native keyboard entry into named scrolling content

import { useRef } from "react";
import type { ReactNode } from "react";

/** A source-specific scrolling region with an explicit native keyboard entry action. */
export interface ScrollRegionProps {
  /** Existing accessible region name. */ readonly name: string;
  /** Meaningful action name for the native focus button. */ readonly action: string;
  /** Existing overflow layout class. */ readonly className: string;
  /** Original content, without source conversion. */ readonly children: ReactNode;
}

/** Focus the real scrolling section through a native keyboard control and retain its original content. */
export function ScrollRegion({ name, action, className, children }: ScrollRegionProps) {
  const region = useRef<HTMLElement>(null);
  return (
    <section className={className} aria-label={name} ref={region} tabIndex={-1}>
      <button type="button" onClick={() => region.current?.focus()}>
        {action}
      </button>
      {children}
    </section>
  );
}
