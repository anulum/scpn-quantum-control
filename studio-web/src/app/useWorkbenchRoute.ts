// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — browser history subscription

import { useSyncExternalStore } from "react";
import { parseWorkbenchRoute } from "./routing";
import type { WorkbenchLocation } from "./routing";

function subscribe(changed: () => void): () => void {
  window.addEventListener("hashchange", changed);
  window.addEventListener("popstate", changed);
  return () => {
    window.removeEventListener("hashchange", changed);
    window.removeEventListener("popstate", changed);
  };
}

function currentHash(): string { return window.location.hash; }

/** Observe actual hash/history changes and release both listeners at panel disposal. */
export function useWorkbenchRoute(): WorkbenchLocation {
  return parseWorkbenchRoute(useSyncExternalStore(subscribe, currentHash));
}
