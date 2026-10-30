// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — live bench view models: order vector, axes, run statistics and history

/** Number of cyclic phase hues the bench colours oscillators with. */
export const PHASE_BINS = 8;

/** Number of run series colours before they repeat. */
export const SERIES_COLOURS = 6;

/** Kuramoto order vector R·e^{iψ} of a set of phases. */
export interface OrderVector {
  /** Coherence R in [0, 1]. */
  readonly r: number;
  /** Mean phase ψ in (−π, π]. */
  readonly psi: number;
}

/**
 * Order vector of phases: the mean of e^{iθ}.
 *
 * @param phases - Oscillator phases in radians; at least one.
 * @returns Coherence and mean phase.
 */
export function orderVector(phases: ArrayLike<number>): OrderVector {
  if (phases.length === 0) throw new RangeError("order vector needs at least one phase");
  let re = 0;
  let im = 0;
  for (let index = 0; index < phases.length; index += 1) {
    const theta = phases[index] as number;
    re += Math.cos(theta);
    im += Math.sin(theta);
  }
  re /= phases.length;
  im /= phases.length;
  return { r: Math.hypot(re, im), psi: Math.atan2(im, re) };
}

/**
 * Cyclic hue bin of a phase.
 *
 * @param theta - Phase in radians, any real value.
 * @returns Bin index 0 … PHASE_BINS − 1, bin 0 centred on phase 0.
 */
export function phaseBin(theta: number): number {
  const turn = 2 * Math.PI;
  const wrapped = ((theta % turn) + turn) % turn;
  return Math.round((wrapped / turn) * PHASE_BINS) % PHASE_BINS;
}

/**
 * Axis ticks on a 1–2–5 ladder covering an interval.
 *
 * @param min - Lower end of the data.
 * @param max - Upper end of the data.
 * @param target - Approximate number of intervals wanted.
 * @returns Ascending tick values from at most `min` to at least `max`.
 */
export function niceTicks(min: number, max: number, target = 5): readonly number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || target < 1)
    throw new RangeError("ticks need a finite interval and a positive target");
  if (max <= min) return [min];
  const raw = (max - min) / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 5].map((factor) => factor * magnitude).find((candidate) => candidate >= raw) ??
    10 * magnitude;
  const first = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let k = 0; ; k += 1) {
    const value = Number((first + k * step).toPrecision(12));
    ticks.push(value);
    if (value >= max) break;
  }
  return ticks;
}

/**
 * First time the order parameter reaches a fraction of its final value.
 *
 * @param series - Order-parameter samples, index 0 at time 0.
 * @param dt - Time between samples.
 * @param fraction - Fraction of the final value; above 1 it is never reached.
 * @returns The time of the first sample at or above the threshold, or `null`
 *   when the final value is not positive.
 */
export function timeToFraction(
  series: ArrayLike<number>,
  dt: number,
  fraction: number,
): number | null {
  if (series.length === 0) return null;
  const final = series[series.length - 1] as number;
  if (!(final > 0)) return null;
  const threshold = fraction * final;
  for (let index = 0; index < series.length; index += 1)
    if ((series[index] as number) >= threshold) return index * dt;
  return null;
}

/** One finished run kept for comparison during this session; nothing is stored. */
export interface BenchRun {
  /** Session-local run number, starting at 1. */
  readonly number: number;
  /** Series colour index, stable for the run. */
  readonly colour: number;
  /** Oscillator count, coupling, frequency spread, steps and mode of the run. */
  readonly controls: {
    /** Oscillator count. */
    readonly n: number;
    /** Coupling strength K. */
    readonly coupling: number;
    /** Half-width of the natural-frequency spread. */
    readonly spread: number;
    /** Integration steps. */
    readonly steps: number;
    /** Kernel mode. */
    readonly mode: string;
  };
  /** Order-parameter samples of the run. */
  readonly orderParameter: Float64Array;
  /** Time between samples. */
  readonly dt: number;
  /** Browser wall time from request to result, in milliseconds. */
  readonly wallMs: number;
  /** Whether the run is drawn over the current result. */
  readonly overlaid: boolean;
}

/** Changes to the session run history. */
export type RunHistoryAction =
  | {
      /** Record a finished run. */
      readonly type: "record";
      /** The finished run; the history assigns its number, colour and overlay. */
      readonly run: Omit<BenchRun, "number" | "colour" | "overlaid">;
    }
  | {
      /** Draw or stop drawing a run over the current result. */
      readonly type: "toggle";
      /** Session number of the run. */
      readonly number: number;
    }
  | {
      /** Forget every recorded run. */
      readonly type: "clear";
    };

/** Most recent runs kept for comparison. */
export const RUN_HISTORY_LIMIT = 12;

/**
 * Session run history, newest first, each run with a stable colour.
 *
 * @param runs - Current history.
 * @param action - Change to apply.
 * @returns The new history.
 */
export function runHistory(
  runs: readonly BenchRun[],
  action: RunHistoryAction,
): readonly BenchRun[] {
  switch (action.type) {
    case "record": {
      const number = (runs[0]?.number ?? 0) + 1;
      const run: BenchRun = {
        ...action.run,
        number,
        colour: (number - 1) % SERIES_COLOURS,
        overlaid: false,
      };
      return [run, ...runs].slice(0, RUN_HISTORY_LIMIT);
    }
    case "toggle":
      return runs.map((run) =>
        run.number === action.number ? { ...run, overlaid: !run.overlaid } : run,
      );
    case "clear":
      return [];
  }
}
