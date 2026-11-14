// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — symmetry, chimera and topology data projection

import committed from "../../../../data/studio/topology_instrument_20261114.json";
import type { Loaded } from "../../shared/browser/data";
import { agrees, count, finite, list, record, text } from "../../shared/browser/jsonFields";
import { parseChimera, parsePersistence, parseSensitivity } from "./topologyLenses";
import type { ChimeraTemplate, PersistenceTemplate, SensitivityTemplate } from "./topologyLenses";

/** The owner's order parameter for one labelling of the nodes. */
export interface OrderReading {
  /** Order-parameter magnitude R. */ readonly magnitude: number;
  /** Mean phase ψ in radians. */ readonly phase: number;
}
/** Order parameter before and after a consistent relabelling of the nodes. */
export interface RelabellingTemplate {
  /** Source owner. */ readonly owner: string;
  /** New position of each node. */ readonly permutation: readonly number[];
  /** Evolution method and time. */ readonly evolution: string;
  /** Reading with the original labels. */ readonly original: OrderReading;
  /** Reading with the permuted labels. */ readonly relabelled: OrderReading;
  /** Absolute magnitude difference. */ readonly magnitudeDifference: number;
  /** Absolute wrapped phase difference. */ readonly phaseDifference: number;
  /** Recorded tolerance. */ readonly tolerance: number;
}
/** Parity weights of one state. */
export interface ParityWeights {
  /** Weight in the even sector. */ readonly even: number;
  /** Weight in the odd sector. */ readonly odd: number;
  /** Leakage out of the even sector, as the owner evaluates it. */ readonly leakage: number;
}
/** Parity weights of one initial state before and after the evolution. */
export interface WeightTrack {
  /** State label. */ readonly state: string;
  /** Weights of the initial state. */ readonly before: ParityWeights;
  /** Weights after the parity-conserving evolution. */ readonly after: ParityWeights;
}
/** Sector spectra against the dense reference, and parity weights. */
export interface SectorTemplate {
  /** Source owners. */ readonly owners: readonly string[];
  /** Parity sector dimensions. */ readonly parityDimensions: Readonly<Record<string, number>>;
  /** Magnetisation sector dimensions keyed by magnetisation. */ readonly magnetisationDimensions: Readonly<
    Record<string, number>
  >;
  /** Largest parity-sector eigenvalue deviation. */ readonly parityDeviation: number;
  /** Largest magnetisation-sector eigenvalue deviation. */ readonly magnetisationDeviation: number;
  /** Recorded tolerance. */ readonly tolerance: number;
  /** Ground energy. */ readonly groundEnergy: number;
  /** Ground-state parity. */ readonly groundParity: string;
  /** Weight tracks in source order. */ readonly weights: readonly WeightTrack[];
  /** Source claim boundary. */ readonly boundary: string;
}
/** All topology templates, parsed once at the boundary. */
export interface TopologyInstrument {
  /** Source artefact identity. */ readonly id: string;
  /** Statement that browser positions carry no physics. */ readonly layoutBoundary: string;
  /** Relabelling template. */ readonly relabelling: RelabellingTemplate;
  /** Sector template. */ readonly sectors: SectorTemplate;
  /** Chimera template. */ readonly chimera: ChimeraTemplate;
  /** Persistence template. */ readonly persistence: PersistenceTemplate;
  /** Sensitivity template. */ readonly sensitivity: SensitivityTemplate;
}

function reading(raw: unknown, field: string): OrderReading {
  const row = record(raw, field);
  return { magnitude: finite(row["magnitude"], "magnitude"), phase: finite(row["phase"], "phase") };
}

function parseRelabelling(raw: unknown): RelabellingTemplate {
  const row = record(raw, "relabelling");
  const permutation = list(row["permutation"], "permutation").map((at) => count(at, "position"));
  const sorted = [...permutation].sort((a, b) => a - b);
  if (sorted.some((node, at) => node !== at))
    throw new Error("The relabelling must be a permutation of the nodes");
  const original = reading(row["original"], "original");
  const relabelled = reading(row["relabelled"], "relabelled");
  const tolerance = finite(row["tolerance"], "tolerance");
  const magnitudeDifference = finite(row["magnitude_difference"], "magnitude_difference");
  agrees(magnitudeDifference, Math.abs(relabelled.magnitude - original.magnitude), "|ΔR|");
  if (magnitudeDifference > tolerance) throw new Error("A relabelling changed the order parameter");
  const evolution = record(row["evolution"], "evolution");
  return {
    owner: text(row["owner"], "owner"),
    permutation,
    evolution: `${text(evolution["method"], "method")}, t = ${finite(evolution["time"], "time")}`,
    original,
    relabelled,
    magnitudeDifference,
    phaseDifference: finite(row["phase_difference"], "phase_difference"),
    tolerance,
  };
}

function weights(raw: unknown, field: string): ParityWeights {
  const row = record(raw, field);
  const even = finite(row["even"], "even");
  const odd = finite(row["odd"], "odd");
  agrees(even + odd, 1, "Parity weights");
  return { even, odd, leakage: finite(row["leakage_from_even"], "leakage_from_even") };
}

function dimensions(raw: unknown, field: string): Record<string, number> {
  const row = record(raw, field);
  return Object.fromEntries(Object.keys(row).map((key) => [key, count(row[key], key)]));
}

function parseSectors(raw: unknown): SectorTemplate {
  const row = record(raw, "sectors");
  const spectra = record(row["spectra"], "spectra");
  const tolerance = finite(row["tolerance"], "tolerance");
  const parityDeviation = finite(spectra["parity_max_deviation"], "parity_max_deviation");
  const magnetisationDeviation = finite(spectra["magnetisation_max_deviation"], "deviation");
  if (parityDeviation > tolerance || magnetisationDeviation > tolerance)
    throw new Error("Sector spectra must agree with the dense reference");
  const tracks = record(row["weights"], "weights");
  const weightTracks = Object.keys(tracks).map((state) => {
    const track = record(tracks[state], state);
    const before = weights(track["before"], "before");
    const after = weights(track["after"], "after");
    if (Math.abs(after.odd - before.odd) > 1e-9)
      throw new Error("Parity weights changed under a parity-conserving evolution");
    return { state, before, after };
  });
  return {
    owners: list(row["owners"], "owners").map((owner) => text(owner, "owner")),
    parityDimensions: dimensions(spectra["parity_dimensions"], "parity_dimensions"),
    magnetisationDimensions: dimensions(spectra["magnetisation_dimensions"], "magnetisation"),
    parityDeviation,
    magnetisationDeviation,
    tolerance,
    groundEnergy: finite(spectra["ground_energy"], "ground_energy"),
    groundParity: text(spectra["ground_parity"], "ground_parity"),
    weights: weightTracks,
    boundary: text(row["claim_boundary"], "claim_boundary"),
  };
}

/**
 * Parse source topology data, refusing unsupported schemas and contradictory evidence.
 * @param raw - Decoded source JSON; unlisted fields are discarded.
 * @returns The parsed templates or a visible refusal.
 */
export function parseTopologyInstrument(raw: unknown): Loaded<TopologyInstrument> {
  try {
    const row = record(raw, "instrument");
    if (row["schema"] !== committed.schema)
      throw new Error("Unsupported topology instrument schema");
    if (row["global_random_state_unchanged"] !== true)
      throw new Error("The build must leave the caller's global random state unchanged");
    const templates = record(row["templates"], "templates");
    return {
      ok: true,
      value: {
        id: text(row["artifact_id"], "artifact_id"),
        layoutBoundary: text(row["layout_boundary"], "layout_boundary"),
        relabelling: parseRelabelling(templates["relabelling"]),
        sectors: parseSectors(templates["sectors"]),
        chimera: parseChimera(templates["chimera"]),
        persistence: parsePersistence(templates["persistence"]),
        sensitivity: parseSensitivity(templates["sensitivity"]),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** Source-owned, committed topology data parsed once at the boundary. */
export const topologyInstrument = parseTopologyInstrument(committed);
