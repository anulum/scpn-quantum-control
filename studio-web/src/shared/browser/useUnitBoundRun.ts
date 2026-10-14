// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — studio-web verdict lifecycle bound to unit identity

import { useRef, useState } from "react";

/** Lifecycle of one verification, parameterised by the verdict it produces. */
export type UnitBoundState<Verdict> =
  | {
      /** Nothing has been run for the current unit. */
      readonly phase: "idle";
    }
  | {
      /** A run is in flight for the current unit. */
      readonly phase: "running";
    }
  | {
      /** A run completed for the current unit. */
      readonly phase: "done";
      /** The verdict it produced. */
      readonly verdict: Verdict;
    }
  | {
      /** A run failed for the current unit. */
      readonly phase: "error";
      /** Why it failed, shown rather than swallowed. */
      readonly reason: string;
    };

/**
 * Bind a verification result to the identity of the unit it was computed for.
 *
 * A verdict card keeps its result in state while its `unit` prop can change
 * underneath it. Two things go wrong without this hook. A verdict completed for
 * unit A stays on screen once the parent renders B, so A's answer is displayed
 * beside B's claim. And a kernel load already in flight for A resolves after
 * the swap and writes A's verdict into the card now showing B.
 *
 * Both are closed here. Changing `identity` clears the displayed state before
 * the new unit is painted, and every run carries a ticket that is compared
 * against the current generation when it resolves, so a stale result is
 * discarded rather than shown. The ticket also orders repeated runs on the same
 * unit: if a caller starts a second run before the first resolves, the first is
 * discarded rather than allowed to overwrite the second.
 *
 * `identity` must be derived from immutable, content-bound fields of the unit.
 * A value that changes on every render would reset the card continuously; one
 * that fails to change between two different units would not reset it at all.
 *
 * @param identity - Stable identity of the unit currently displayed.
 * @returns The current lifecycle state and a `run` that owns the guard.
 */
export function useUnitBoundRun<Verdict>(identity: string): {
  /** Lifecycle state for the unit currently identified. */
  readonly state: UnitBoundState<Verdict>;
  /** Start a run whose result is discarded if the unit changes underneath it. */
  readonly run: (task: () => Promise<Verdict>, fallbackReason: string) => Promise<void>;
} {
  const [trackedIdentity, setTrackedIdentity] = useState(identity);
  const [state, setState] = useState<UnitBoundState<Verdict>>({ phase: "idle" });
  const generation = useRef(0);

  if (trackedIdentity !== identity) {
    // Adjusting state during render is React's supported way to derive state
    // from props. It matters here that the reset happens before paint: the
    // alternative, an effect, would show A's verdict beside B for one frame.
    generation.current += 1;
    setTrackedIdentity(identity);
    setState({ phase: "idle" });
  }

  const run = async (task: () => Promise<Verdict>, fallbackReason: string): Promise<void> => {
    generation.current += 1;
    const ticket = generation.current;
    setState({ phase: "running" });
    try {
      const verdict = await task();
      if (generation.current !== ticket) {
        return;
      }
      setState({ phase: "done", verdict });
    } catch (error) {
      if (generation.current !== ticket) {
        return;
      }
      setState({
        phase: "error",
        reason: error instanceof Error ? error.message : fallbackReason,
      });
    }
  };

  return { state, run };
}
