// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web panel state

import { create } from "zustand";

/** Panel state: the lane filter applied to the support-matrix grid. */
export interface PanelState {
  readonly laneFilter: string;
  readonly setLaneFilter: (lane: string) => void;
}

/** All lanes are shown until the user narrows the grid. */
export const ALL_LANES = "all";

export const usePanelStore = create<PanelState>((set) => ({
  laneFilter: ALL_LANES,
  setLaneFilter: (lane: string) => set({ laneFilter: lane }),
}));
