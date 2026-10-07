// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original immutable archive comparison projection

import { canonicalDigest, writeJson } from "../../shared/contracts";
import type { LocalRunRecord, RawCodec, ResolvedSettings } from "../../shared/contracts";
import { admitWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import type {
  AdmittedWorkspaceArchive,
  WorkspaceArchiveMember,
} from "../../shared/storage/workspaceArchive";
import { encodeKuramotoInput } from "../../panel/kuramoto";
import { readLocalExperiment, validateExperimentLifecycle } from "../experiments/experimentArchive";
import type {
  ArchivedExperimentEvent,
  LocalExperimentSource,
} from "../experiments/experimentArchive";
import {
  artifactBytesDigest,
  artifactContent,
  decodeFloat64,
  ExperimentRefusal,
  experimentSchemas,
  readExperimentArtifact,
} from "../experiments/kuramotoArtifacts";
import type { ComparisonKey, ComparisonObservation, ComparisonSnapshot } from "./comparisonModel";

/** Explicit refusal for an absent selection or incompatible original source binding. */
export class ComparisonSourceRefusal extends Error {}
/** Original run selection remains separate from its revision. */
export interface ComparisonRunReference {
  /** Original run document hash. */ readonly hash: string;
  /** Original revision bound by the run. */ readonly revisionHash: string;
  /** Original attempt UUID. */ readonly attemptId: string;
  /** Last recorded event kind, or explicitly absent terminal evidence. */ readonly state: string;
}
/** One completely admitted archive; selecting a revision never changes its saved draft. */
export interface ComparisonArchive {
  /** Unchanged original archive and producer-verified members. */ readonly archive: AdmittedWorkspaceArchive;
  /** Original trusted verifier registry captured before asynchronous admission. */ readonly rawCodecs: ReadonlyMap<
    string,
    RawCodec
  >;
  /** Original listed revision choices, preserving manifest order. */ readonly revisions: readonly {
    /** Immutable identity. */ readonly hash: string;
  }[];
  /** Original explicitly indexed run choices; orphan members are never promoted. */ readonly runs: readonly ComparisonRunReference[];
}

function member(
  archive: AdmittedWorkspaceArchive,
  hash: string,
  schema: string,
): WorkspaceArchiveMember {
  const found = archive.members.find((value) => value.sha256 === hash && value.schema === schema);
  if (found === undefined)
    throw new ComparisonSourceRefusal("Original comparison source member is absent or unsupported");
  return found;
}
function runRecord(archive: AdmittedWorkspaceArchive, hash: string): LocalRunRecord {
  // Complete original graph admission has already validated/frozen this exact document.
  return archive.documents[member(archive, hash, "local_run_record.v1").sha256] as LocalRunRecord;
}

/** Admit all original references/versions/digests before offering read-only revision and run choices. */
export async function readComparisonArchive(
  json: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
): Promise<ComparisonArchive> {
  const codecs = new Map(rawCodecs),
    archive = await admitWorkspaceArchive(json, codecs);
  const references = archive.manifest.body["revision_refs"] as readonly {
    readonly sha256: string;
  }[];
  const artifacts = archive.manifest.body["artefact_refs"] as readonly {
    readonly sha256: string;
    readonly schema: string;
  }[];
  const runs = artifacts
    .filter((reference) => reference.schema === "local_run_record.v1")
    .map((reference) => {
      const record = runRecord(archive, reference.sha256);
      const events = record.body["events"] as readonly ArchivedExperimentEvent[];
      return Object.freeze({
        hash: reference.sha256,
        revisionHash: record.body["revision_hash"] as string,
        attemptId: record.body["attempt_id"] as string,
        state: events.at(-1)?.kind ?? "no terminal event",
      });
    });
  return Object.freeze({
    archive,
    rawCodecs: codecs,
    revisions: Object.freeze(
      references.map((reference) => Object.freeze({ hash: reference.sha256 })),
    ),
    runs: Object.freeze(runs),
  });
}

/** Read the recorded output through the original artifact/lifecycle owners, without instantiating or executing imported WASM. */
async function recordedObservations(
  source: LocalExperimentSource,
  record: LocalRunRecord,
): Promise<readonly ComparisonObservation[]> {
  const body = record.body,
    planHash = body["plan_hash"] as string;
  const payload = await readExperimentArtifact(
    "plan",
    artifactContent(member(source.archive, planHash, experimentSchemas.plan)),
  );
  const encoded = encodeKuramotoInput(source.request) as Uint8Array<ArrayBuffer>;
  const shape = payload["shape"] as {
    readonly n: number;
    readonly steps: number;
    readonly mode: string;
  };
  if (
    payload["revision_hash"] !== source.revisionHash ||
    payload["kernel_sha256"] !== source.kernelHash ||
    payload["environment_sha256"] !== source.environmentMember.sha256 ||
    payload["input_sha256"] !== (await artifactBytesDigest(encoded)) ||
    payload["binary_bytes"] !== source.kernelBytes.length ||
    shape.n !== source.request.omega.length ||
    shape.steps !== source.request.steps ||
    shape.mode !== source.request.mode ||
    (await canonicalDigest("studio.comparison-bounds.v1", payload["bounds"])) !==
      (await canonicalDigest("studio.comparison-bounds.v1", source.environment["bounds"]))
  )
    throw new ComparisonSourceRefusal(
      "Recorded numerical plan differs from the original immutable source",
    );
  const policy = await readExperimentArtifact(
    "policy",
    artifactContent(
      member(source.archive, payload["policy_sha256"] as string, experimentSchemas.policy),
    ),
  );
  if (
    (await canonicalDigest("studio.comparison-policy.v1", policy)) !==
    (await canonicalDigest("studio.comparison-policy.v1", payload["policy"]))
  )
    throw new ComparisonSourceRefusal(
      "Recorded numerical policy differs from its original artifact",
    );
  // Also require the original bytes whose independently computed digest binds effective input.
  member(source.archive, payload["input_sha256"] as string, experimentSchemas.input);
  validateExperimentLifecycle(
    body["events"] as readonly ArchivedExperimentEvent[],
    body["run_id"] as string,
    source.revisionHash,
    planHash,
    source.kernelHash,
    "result",
  );
  const outputs = body["output_refs"] as readonly {
    readonly sha256: string;
    readonly schema: string;
  }[];
  const [outputReference] = outputs;
  if (
    outputs.length !== 1 ||
    outputReference === undefined ||
    outputReference.schema !== experimentSchemas.output
  )
    throw new ComparisonSourceRefusal("One original recorded numerical output required");
  const output = await readExperimentArtifact(
    "output",
    artifactContent(member(source.archive, outputReference.sha256, experimentSchemas.output)),
  );
  if (
    output["revision_hash"] !== source.revisionHash ||
    output["plan_hash"] !== planHash ||
    output["kernel_sha256"] !== source.kernelHash
  )
    throw new ComparisonSourceRefusal(
      "Recorded output belongs to another immutable source or plan",
    );
  const order = output["order_parameter"] as readonly string[],
    phases = output["theta_final"] as readonly string[];
  if (order.length !== source.request.steps + 1 || phases.length !== source.request.omega.length)
    throw new ComparisonSourceRefusal("Recorded numerical output shape differs from its source");
  const finalTime = source.request.steps * source.request.dt;
  return Object.freeze([
    ...order.map((bits, index) =>
      Object.freeze({
        key: "order-parameter",
        time: index * source.request.dt,
        value: decodeFloat64(bits),
      }),
    ),
    ...phases.map((bits, index) =>
      Object.freeze({ key: `oscillator/${index}`, time: finalTime, value: decodeFloat64(bits) }),
    ),
  ]);
}

/** Project exact semantic metadata and original completed output from explicitly selected immutable identities. */
export async function projectComparisonSource(
  selection: ComparisonArchive,
  revisionHash: string,
  runHash: string | null,
): Promise<ComparisonSnapshot> {
  const archive = selection.archive,
    revision = archive.revisions[revisionHash];
  if (
    !selection.revisions.some((reference) => reference.hash === revisionHash) ||
    revision === undefined
  )
    throw new ComparisonSourceRefusal("Selected immutable revision is absent");
  const settingsRef = revision.body["semantic_settings_ref"] as { readonly sha256: string };
  const settings = (archive.documents[settingsRef.sha256] as ResolvedSettings).body;
  const effective = settings["effective"] as Readonly<Record<string, unknown>>;
  const parameters = revision.body["parameters"] as Readonly<Record<string, unknown>>;
  const semantics: Record<string, unknown> = {
    problem: revision.body["problem_ref"],
    program: revision.body["program_ref"],
    evidence: revision.body["input_refs"],
    revision_extensions: revision.extensions,
    settings_reference: revision.body["semantic_settings_ref"],
    policy: settings["policy_ref"],
    environment: settings["environment_ref"],
    rejected_fields: settings["rejected_fields"],
  };
  for (const [key, value] of Object.entries(parameters)) {
    semantics[`parameters.${key}`] = value;
    semantics[`units.${key}`] = archive.parameterUnits[key];
  }
  for (const kind of ["requested", "effective", "origins"] as const)
    for (const [key, value] of Object.entries(settings[kind] as Readonly<Record<string, unknown>>))
      semantics[`${kind === "origins" ? "setting_origins" : `${kind}_settings`}.${key}`] = value;
  let record: LocalRunRecord | null = null;
  if (runHash !== null) {
    const reference = selection.runs.find((value) => value.hash === runHash);
    if (reference === undefined)
      throw new ComparisonSourceRefusal("Selected recorded run is absent");
    if (reference.revisionHash !== revisionHash)
      throw new ComparisonSourceRefusal("Selected run belongs to another immutable revision");
    record = runRecord(archive, runHash);
    semantics["output_evidence"] = record.body["output_refs"];
  }
  const extension = revision.extensions["local_experiment"] as
    | Readonly<Record<string, unknown>>
    | undefined;
  const dataset = await canonicalDigest("studio.comparison-dataset.v1", {
    problem: revision.body["problem_ref"],
    inputs: revision.body["input_refs"],
    initial_parameters: {
      omega: parameters["omega"] ?? null,
      theta0: parameters["theta0"] ?? null,
    },
  });
  let key: ComparisonKey = Object.freeze({
    estimand: "Original output not admitted",
    unit: "Original output units unavailable",
    model: writeJson(extension ?? null),
    dataset,
    backend:
      typeof effective["backend"] === "string"
        ? effective["backend"]
        : writeJson(effective["backend"] ?? null),
    precision:
      typeof effective["precision"] === "string"
        ? effective["precision"]
        : writeJson(effective["precision"] ?? null),
    shotProtocol: writeJson({
      shots: effective["shots"] ?? null,
      protocol: effective["shot_protocol"] ?? null,
    }),
    calibration: "Original calibration meaning unavailable",
    uncertainty: "Original uncertainty meaning unavailable",
  });
  let unavailableReason: string | null = "No run selected",
    observations: readonly ComparisonObservation[] = Object.freeze([]);
  if (record !== null) {
    const terminal = (record.body["events"] as readonly ArchivedExperimentEvent[]).at(-1)?.kind;
    if (terminal !== "result")
      unavailableReason = `Recorded run is ${terminal ?? "missing a terminal event"}; no completed output`;
    else {
      try {
        const source = await readLocalExperiment(
          archive.preview.json,
          selection.rawCodecs,
          revisionHash,
        );
        observations = await recordedObservations(source, record);
        key = Object.freeze({
          estimand: "Order parameter R and per-oscillator final phase on requested i×dt grid",
          unit: "R:1; phase:rad; time:model-time",
          model: `classical Kuramoto / ${source.request.mode} / program:${source.kernelHash}`,
          dataset,
          backend: source.environment["backend"] as string,
          precision: source.environment["dtype"] as string,
          shotProtocol: "None: deterministic local integration",
          calibration: "Not applicable: local classical model",
          uncertainty: "Not estimated by original output producer",
        });
        unavailableReason = null;
      } catch (cause: unknown) {
        if (!(cause instanceof ExperimentRefusal) && !(cause instanceof ComparisonSourceRefusal))
          throw cause;
        unavailableReason = cause.message;
      }
    }
  }
  return Object.freeze({
    revisionHash,
    runHash,
    semantics: Object.freeze(semantics),
    key,
    observations,
    unavailableReason,
  });
}
