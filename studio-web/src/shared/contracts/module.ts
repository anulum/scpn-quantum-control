// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workbench module contract and build-time registry

import type { ReactNode } from "react";

/**
 * One destination a module adds to the workbench.
 *
 * The shell owns layout, routing and authority. A stage only renders inside the
 * work area, from the services the shell hands it.
 */
export interface StudioStage<Services> {
  /** Stable identifier; it is the first segment of the stage address. */
  readonly id: string;
  /** Label shown in navigation and in the breadcrumb. */
  readonly label: string;
  /** Navigation group: the primary stages of the work, or a context destination. */
  readonly group: "primary" | "context";
  /** Deep-link instruments of the stage; each is a second address segment. */
  readonly instruments: readonly string[];
  /** Render the stage from the shell services and the requested instrument. */
  readonly render: (services: Services, instrument: string | null) => ReactNode;
}

/** What a module needs from the installed package before it can work. */
export interface ModuleRequirements {
  /** Manifest verbs the module calls. */
  readonly verbs: readonly string[];
  /** Evidence schema identifiers the module reads. */
  readonly schemas: readonly string[];
}

/**
 * A unit of the workbench that is added or removed as a whole.
 *
 * Modules are compiled into the application and listed in one registry; none is
 * loaded at run time. A module contributes to fixed shell slots and decides no
 * authority: whether something may run stays with the shell.
 */
export interface StudioModule<Services> {
  /** Stable identifier of the module. */
  readonly id: string;
  /** Version of the module's own contract with the shell. */
  readonly version: string;
  /** Verbs and schemas the installed manifest must declare. */
  readonly requires: ModuleRequirements;
  /** Stages the module adds, in navigation order. */
  readonly stages: readonly StudioStage<Services>[];
}

/** Destination of one admitted address path. */
export interface StageDestination {
  /** Identifier of the registered stage. */
  readonly view: string;
  /** Instrument of that stage, or its landing page. */
  readonly instrument: string | null;
}

/** A requirement of a module that the installed manifest does not declare. */
export interface MissingRequirement {
  /** Module whose requirement is not met. */
  readonly module: string;
  /** Kind of the missing declaration. */
  readonly kind: "verb" | "schema";
  /** Name of the missing verb or schema. */
  readonly name: string;
}

/** Validated, immutable view of the registered modules. */
export interface ModuleRegistry<Services> {
  /** Registered modules in registration order. */
  readonly modules: readonly StudioModule<Services>[];
  /** Every stage in registration order. */
  readonly stages: readonly StudioStage<Services>[];
  /** Admitted address paths; nothing outside this table is routable. */
  readonly paths: ReadonlyMap<string, StageDestination>;
  /** Find a stage by identifier. */
  readonly stage: (id: string) => StudioStage<Services> | undefined;
  /** Find the module that registered a stage. */
  readonly owner: (stage: string) => StudioModule<Services> | undefined;
  /** List what the registered modules require and the manifest does not declare. */
  readonly missing: (declared: ModuleRequirements) => readonly MissingRequirement[];
}

const identifier = /^[a-z][a-z0-9-]{0,39}$/u;

function requireIdentifier(kind: string, value: string): void {
  if (!identifier.test(value)) throw new Error(`Invalid ${kind} identifier`);
}

function requireDistinct(kind: string, values: readonly string[]): void {
  if (new Set(values).size !== values.length) throw new Error(`Repeated ${kind} identifier`);
}

/**
 * Validate modules and derive the stage list and the closed address table.
 *
 * Identifiers are lower-case words of at most forty characters. A repeated
 * module, stage or instrument identifier is refused, so that one address can
 * never name two destinations.
 *
 * @param modules - Modules in the order their stages appear in navigation.
 * @returns The immutable registry.
 * @throws Error When an identifier is malformed or repeated.
 */
export function createModuleRegistry<Services>(
  modules: readonly StudioModule<Services>[],
): ModuleRegistry<Services> {
  for (const module of modules) requireIdentifier("module", module.id);
  requireDistinct(
    "module",
    modules.map((module) => module.id),
  );
  const stages = modules.flatMap((module) => module.stages);
  requireDistinct(
    "stage",
    stages.map((stage) => stage.id),
  );
  const paths = new Map<string, StageDestination>();
  for (const stage of stages) {
    requireIdentifier("stage", stage.id);
    requireDistinct("instrument", stage.instruments);
    paths.set(`/${stage.id}`, Object.freeze({ view: stage.id, instrument: null }));
    for (const instrument of stage.instruments) {
      requireIdentifier("instrument", instrument);
      paths.set(`/${stage.id}/${instrument}`, Object.freeze({ view: stage.id, instrument }));
    }
  }
  const byIdentifier = new Map(stages.map((stage) => [stage.id, stage]));
  const owners = new Map(
    modules.flatMap((module) => module.stages.map((stage) => [stage.id, module] as const)),
  );
  return Object.freeze({
    modules: Object.freeze([...modules]),
    stages: Object.freeze(stages),
    paths,
    stage: (id: string) => byIdentifier.get(id),
    owner: (stage: string) => owners.get(stage),
    missing: (declared: ModuleRequirements) => {
      const verbs = new Set(declared.verbs);
      const schemas = new Set(declared.schemas);
      return modules.flatMap((module) => [
        ...module.requires.verbs
          .filter((name) => !verbs.has(name))
          .map((name) => ({ module: module.id, kind: "verb" as const, name })),
        ...module.requires.schemas
          .filter((name) => !schemas.has(name))
          .map((name) => ({ module: module.id, kind: "schema" as const, name })),
      ]);
    },
  });
}
