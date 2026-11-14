// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — learning, forecasting and domain-benchmark data projection

import committed from "../../../../data/studio/learning_instrument_20261114.json";
import type { Loaded } from "../../shared/browser/data";
import {
  agrees,
  count,
  finite,
  flag,
  list,
  numbers,
  record,
  text,
} from "../../shared/browser/jsonFields";
import { parseDomainProxies, parseFreeEnergy } from "./learningDomains";
import type { DomainProxy, FreeEnergyTemplate } from "./learningDomains";

/** One dataset split exactly as the forecasting evidence records it. */
export interface ForecastSplit {
  /** Split name, in train, calibration, test order. */ readonly split: string;
  /** Independent trajectories in the split. */ readonly samples: number;
  /** Source content digest of the split. */ readonly digest: string;
  /** Fraction of missing inputs. */ readonly missing: number;
}
/** Forecasting evidence: data custody, the classical reference model and held-out scores. */
export interface ForecastTemplate {
  /** Source owner. */ readonly owner: string;
  /** Source claim boundary. */ readonly boundary: string;
  /** Dataset origin. */ readonly source: string;
  /** Dataset licence statement. */ readonly licence: string;
  /** Root seed of the generator. */ readonly seed: number;
  /** Splits in source order. */ readonly splits: readonly ForecastSplit[];
  /** Training trajectories. */ readonly training: number;
  /** Calibration and test trajectories together. */ readonly heldOut: number;
  /** Held-out trajectories found among the training ones; always 0 once parsed. */ readonly overlapping: number;
  /** The owner's refusal to fit a held-out batch. */ readonly fitRefusal: string;
  /** Reference model kind. */ readonly model: string;
  /** Ridge penalty of the reference model. */ readonly ridge: number;
  /** Held-out test trajectories. */ readonly testSamples: number;
  /** Held-out wrapped mean squared error. */ readonly mse: number;
  /** Persistence baseline on the same samples. */ readonly persistenceMse: number;
  /** Whether the model beats persistence, as the source decided. */ readonly beatsPersistence: boolean;
  /** Target interval coverage. */ readonly targetCoverage: number;
  /** Observed sample coverage on the test split. */ readonly sampleCoverage: number;
  /** Interval claim boundary. */ readonly intervalBoundary: string;
}
/** One parameter-shift descent step. */
export interface QsnnStep {
  /** Step number from one. */ readonly index: number;
  /** Training loss after the step. */ readonly loss: number;
  /** Gradient norm at the step. */ readonly gradientNorm: number;
}
/** Seeded QSNN training, checkpoint resume and held-out evaluation. */
export interface QsnnTemplate {
  /** Source owner. */ readonly owner: string;
  /** Layer seed; the trainer itself consumes no random state. */ readonly seed: number;
  /** Dataset origin. */ readonly source: string;
  /** Dataset licence statement. */ readonly licence: string;
  /** Optimiser method. */ readonly method: string;
  /** Learning rate of the optimiser. */ readonly learningRate: number;
  /** Descent steps in order. */ readonly steps: readonly QsnnStep[];
  /** Checkpoint schema. */ readonly checkpointSchema: string;
  /** The owner's statement on random state. */ readonly randomness: string;
  /** Epochs before the checkpoint was saved. */ readonly savedAfterEpochs: number;
  /** Whether the resumed run reproduced the uninterrupted one; always true once parsed. */ readonly resumedMatches: boolean;
  /** Held-out samples scored. */ readonly heldOutSamples: number;
  /** Held-out loss, or null when no in-distribution sample exists. */ readonly heldOutLoss:
    | number
    | null;
  /** Held-out claim boundary. */ readonly heldOutBoundary: string;
  /** The owner's refusal to score training samples as held out. */ readonly overlapRefusal: string;
}
/** The committed JAX NQS baseline, or its declared absence. */
export type NqsTemplate =
  | {
      /** Discriminant. */ readonly status: "unavailable";
    }
  | {
      /** Discriminant. */ readonly status: "available";
      /** Committed evidence path. */ readonly evidencePath: string;
      /** Evidence digest verified by the owner. */ readonly sha256: string;
      /** Source claim boundary. */ readonly boundary: string;
      /** Visible spins. */ readonly qubits: number;
      /** Optimiser seed. */ readonly seed: number;
      /** JAX version that produced the evidence. */ readonly jaxVersion: string;
      /** Numeric posture of that run. */ readonly posture: string;
      /** Exact ground energy. */ readonly exact: number;
      /** Final variational energy. */ readonly variational: number;
      /** Relative error against the exact reference. */ readonly relativeError: number;
      /** Whether the declared tolerance held. */ readonly withinTolerance: boolean;
      /** Energy after each iteration. */ readonly history: readonly number[];
    };
/** All learning templates, parsed once at the boundary. */
export interface LearningInstrument {
  /** Source artefact identity. */ readonly id: string;
  /** Forecasting template. */ readonly forecasting: ForecastTemplate;
  /** QSNN template. */ readonly qsnn: QsnnTemplate;
  /** NQS template. */ readonly nqs: NqsTemplate;
  /** Domain proxies in source order. */ readonly domains: readonly DomainProxy[];
  /** Free-energy workflow. */ readonly freeEnergy: FreeEnergyTemplate;
}

function parseForecast(raw: unknown): ForecastTemplate {
  const row = record(raw, "forecasting");
  const dataset = record(row["dataset"], "forecasting.dataset");
  const splits = list(dataset["splits"], "forecasting.dataset.splits").map((item) => {
    const split = record(item, "split");
    return {
      split: text(split["split"], "split"),
      samples: count(split["samples"], "samples"),
      digest: text(split["content_digest"], "content_digest"),
      missing: finite(split["missing_fraction"], "missing_fraction"),
    };
  });
  if (splits.map((item) => item.split).join() !== "train,calibration,test")
    throw new Error("Forecasting splits must be train, calibration and test in order");
  const custody = record(row["held_out"], "forecasting.held_out");
  const overlapping = count(custody["overlapping_samples"], "overlapping_samples");
  if (overlapping !== 0) throw new Error("Held-out records cannot enter training");
  if (
    text(custody["model_training_batch_digest"], "model_training_batch_digest") !==
    splits[0]?.digest
  )
    throw new Error("The forecasting model must be fitted on the training split");
  const model = record(row["model"], "forecasting.model");
  const test = record(row["test"], "forecasting.test");
  const interval = record(row["interval"], "forecasting.interval");
  return {
    owner: text(row["owner"], "owner"),
    boundary: text(row["claim_boundary"], "claim_boundary"),
    source: text(dataset["source"], "source"),
    licence: text(dataset["licence"], "licence"),
    seed: count(dataset["seed"], "seed"),
    splits,
    training: count(custody["training_samples"], "training_samples"),
    heldOut: count(custody["held_out_samples"], "held_out_samples"),
    overlapping,
    fitRefusal: text(custody["fit_refusal"], "fit_refusal"),
    model: text(model["kind"], "kind"),
    ridge: finite(model["ridge"], "ridge"),
    testSamples: count(test["samples"], "samples"),
    mse: finite(test["wrapped_mse"], "wrapped_mse"),
    persistenceMse: finite(test["persistence_wrapped_mse"], "persistence_wrapped_mse"),
    beatsPersistence: flag(test["lower_mse_than_persistence"], "lower_mse_than_persistence"),
    targetCoverage: finite(interval["target_coverage"], "target_coverage"),
    sampleCoverage: finite(interval["sample_coverage"], "sample_coverage"),
    intervalBoundary: text(interval["claim_boundary"], "interval.claim_boundary"),
  };
}

function parseQsnn(raw: unknown): QsnnTemplate {
  const row = record(raw, "qsnn");
  const dataset = record(row["dataset"], "qsnn.dataset");
  const optimiser = record(row["optimiser"], "qsnn.optimiser");
  const checkpoint = record(row["checkpoint"], "qsnn.checkpoint");
  const heldOut = record(row["held_out"], "qsnn.held_out");
  const steps = list(row["steps"], "qsnn.steps").map((item, at) => {
    const step = record(item, "step");
    if (count(step["index"], "index") !== at + 1)
      throw new Error("Descent steps must be numbered from one");
    return {
      index: at + 1,
      loss: finite(step["loss"], "loss"),
      gradientNorm: finite(step["gradient_norm"], "gradient_norm"),
    };
  });
  const learningRate = finite(optimiser["learning_rate"], "learning_rate");
  agrees(
    finite(checkpoint["learning_rate"], "checkpoint.learning_rate"),
    learningRate,
    "checkpoint.learning_rate",
  );
  const resumedMatches = flag(
    checkpoint["resumed_matches_uninterrupted"],
    "resumed_matches_uninterrupted",
  );
  if (!resumedMatches) throw new Error("A resumed checkpoint must reproduce the uninterrupted run");
  const loss = heldOut["in_distribution_loss"];
  return {
    owner: text(row["owner"], "owner"),
    seed: count(record(row["layer"], "qsnn.layer")["seed"], "seed"),
    source: text(dataset["source"], "source"),
    licence: text(dataset["licence"], "licence"),
    method: text(optimiser["method"], "method"),
    learningRate,
    steps,
    checkpointSchema: text(checkpoint["schema"], "schema"),
    randomness: text(checkpoint["randomness"], "randomness"),
    savedAfterEpochs: count(checkpoint["saved_after_epochs"], "saved_after_epochs"),
    resumedMatches,
    heldOutSamples: count(heldOut["samples"], "samples"),
    heldOutLoss: loss === null ? null : finite(loss, "in_distribution_loss"),
    heldOutBoundary: text(heldOut["claim_boundary"], "held_out.claim_boundary"),
    overlapRefusal: text(heldOut["overlap_refusal"], "overlap_refusal"),
  };
}

function parseNqs(raw: unknown): NqsTemplate {
  const row = record(raw, "nqs");
  if (row["status"] === "unavailable") return { status: "unavailable" };
  if (row["status"] !== "available") throw new Error("NQS status must be available or unavailable");
  const sha256 = text(row["evidence_sha256"], "evidence_sha256");
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error("Invalid NQS evidence digest");
  const request = record(row["request"], "nqs.request");
  const environment = record(row["environment"], "nqs.environment");
  const comparison = record(row["comparison"], "nqs.comparison");
  const history = numbers(comparison["energy_history"], "energy_history");
  const variational = finite(comparison["variational_energy"], "variational_energy");
  if (history.length === 0) throw new Error("NQS energy history must not be empty");
  agrees(history[history.length - 1] as number, variational, "energy_history end");
  return {
    status: "available",
    evidencePath: text(row["evidence_path"], "evidence_path"),
    sha256,
    boundary: text(row["claim_boundary"], "claim_boundary"),
    qubits: count(request["qubits"], "qubits"),
    seed: count(request["seed"], "seed"),
    jaxVersion: text(environment["jax_version"], "jax_version"),
    posture: text(environment["numeric_posture"], "numeric_posture"),
    exact: finite(comparison["exact_ground_energy"], "exact_ground_energy"),
    variational,
    relativeError: finite(comparison["relative_error"], "relative_error"),
    withinTolerance: flag(comparison["within_declared_tolerance"], "within_declared_tolerance"),
    history,
  };
}

/**
 * Parse source learning data, refusing unsupported schemas and contradictory evidence.
 * @param raw - Decoded source JSON; unlisted fields are discarded.
 * @returns The parsed templates or a visible refusal.
 */
export function parseLearningInstrument(raw: unknown): Loaded<LearningInstrument> {
  try {
    const row = record(raw, "instrument");
    if (row["schema"] !== committed.schema)
      throw new Error("Unsupported learning instrument schema");
    if (row["global_random_state_unchanged"] !== true)
      throw new Error("The build must leave the caller's global random state unchanged");
    const templates = record(row["templates"], "templates");
    return {
      ok: true,
      value: {
        id: text(row["artifact_id"], "artifact_id"),
        forecasting: parseForecast(templates["forecasting"]),
        qsnn: parseQsnn(templates["qsnn"]),
        nqs: parseNqs(templates["nqs"]),
        domains: parseDomainProxies(templates["domain_proxies"]),
        freeEnergy: parseFreeEnergy(templates["free_energy"]),
      },
    };
  } catch (cause: unknown) {
    return { ok: false, reason: (cause as Error).message };
  }
}

/** Source-owned, committed learning data parsed once at the boundary. */
export const learningInstrument = parseLearningInstrument(committed);
