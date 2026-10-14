// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — workbench module contract and build-time registry

import type { ReactNode } from "react";
import type { ResourcePlan } from "../resources/admission";

/** Exact document bytes and identity supplied by the workspace admission owner. */
export interface ModuleDocument {
  /** Original document or evidence schema, without a compatibility conversion. */
  readonly schema: string;
  /** Admission owner digest; workspace archives retain their domain-specific archive digest. */
  readonly sha256: string;
  /** Exact admitted content; rendering never replaces the stored document. */
  readonly content: string;
}

/** One admitted document shown by a module in the explorer. */
export interface ModuleExplorerNode {
  /** Local node identity, distinct from the document digest. */
  readonly id: string;
  /** Declared node kind used to select an editor. */
  readonly kind: string;
  /** Display text supplied by the owning module. */
  readonly label: string;
  /** Source document retained unchanged when its module is removed. */
  readonly document: ModuleDocument;
}

/** A project-tree contribution derived from the admitted shell services. */
export interface ModuleExplorer<Services> {
  /** Unique contribution identifier within the explorer slot. */
  readonly id: string;
  /** Heading for this part of the project tree. */
  readonly label: string;
  /** Node kinds this contribution may emit. */
  readonly kinds: readonly string[];
  /** Read the admitted documents; this operation neither saves nor executes them. */
  readonly nodes: (services: Services) => readonly ModuleExplorerNode[];
}

/** An editor, inspector section or settings projection for declared schemas. */
export interface ModuleDocumentView<Services> {
  /** Schema-bound views consume admitted documents rather than initial stage controls. */
  readonly kind: "document";
  /** Unique identifier within its shell slot. */
  readonly id: string;
  /** Heading identifying the document projection. */
  readonly label: string;
  /** Exact original schema identifiers the view understands. */
  readonly schemas: readonly string[];
  /** Render only inside the slot from the unchanged admitted document. */
  readonly render: (services: Services, document: ModuleDocument) => ReactNode;
}

/** Initial authoring or inspection controls contributed to one registered work-area stage. */
export interface ModuleStageEditor<Services> {
  /** Stage editors use the shell's stage services without manufacturing a document. */
  readonly kind: "stage";
  /** Unique identifier in the editor slot. */
  readonly id: string;
  /** Heading identifying the work-area controls. */
  readonly label: string;
  /** Registered stage where these controls appear. */
  readonly stage: string;
  /** Render within the editor slot using the original source and lifecycle owners. */
  readonly render: (services: Services) => ReactNode;
}

/** Work-area contributions distinguish authored stage controls from admitted document projections. */
export type ModuleEditor<Services, DocumentServices = Services> =
  | ModuleStageEditor<Services>
  | ModuleDocumentView<DocumentServices>;

/** Declarative commands are routed by the shell, rather than granting authority. */
export type ModuleCommandAction =
  | {
      /** Navigate to one build-time registered stage. */
      readonly kind: "navigate";
      /** Existing registered stage identifier. */
      readonly stage: string;
      /** Registered deep-link instrument, or the stage landing page. */
      readonly instrument: string | null;
    }
  | {
      /** Request a registered job through the existing policy and lifecycle owners. */
      readonly kind: "job";
      /** Registered job kind; a command supplies no admission verdict. */
      readonly job: string;
    };

/** One menu or command-search entry, without an executable authorisation callback. */
export interface ModuleCommand {
  /** Unique command identifier. */
  readonly id: string;
  /** Searchable menu label. */
  readonly label: string;
  /** Declarative target admitted and dispatched by the shell. */
  readonly action: ModuleCommandAction;
}

/** Source-owned local execution with admission kept outside the module. */
export type ModuleJobEstimate =
  | {
      /** Complete existing declared single-run resource plan. */ readonly kind: "declared-resources";
      /** Original buffers and work declarations, without an admission verdict. */ readonly plan: ResourcePlan;
    }
  | {
      /** Existing workflow owner admits every numerical stage before worker allocation. */ readonly kind: "workflow-runtime";
      /** Exact original selected graph identity. */ readonly workflowDigest: string;
      /** Original immutable baseline used by the graph. */ readonly baselineRevision: string;
      /** Exact original source snapshot identity. */ readonly archiveDigest: string;
      /** Actual admitted graph stage count. */ readonly stages: number;
      /** Actual admitted sweep cell count. */ readonly cells: number;
      /** Whole-graph retained memory is not estimated by this projection. */ readonly aggregateBytes: null;
    };

/** Source-owned local execution with admission kept outside the module. */
export interface ModuleJob<Services> {
  /** Unique source job kind. */
  readonly id: string;
  /** Label for the job dock and command search. */
  readonly label: string;
  /** Transport; local service remains unavailable until its separate integration exists. */
  readonly runner: "worker" | "local-service";
  /** Stage whose existing runtime and controls own this job. */
  readonly stage: string;
  /** Complete resource declarations; the policy owner computes their admission. */
  readonly estimate: (services: Services, document: ModuleDocument) => Promise<ModuleJobEstimate>;
  /** Existing runtime re-admits its own source and observes cancellation and disposal. */
  readonly run: (
    services: Services,
    document: ModuleDocument,
    signal: AbortSignal,
  ) => Promise<void>;
}

/** Contribution slots whose ownership the shell can inspect. */
export type ModuleSlot = "explorer" | "editors" | "inspectors" | "commands" | "jobs" | "settings";

/**
 * One destination a module adds to the workbench.
 *
 * The shell owns layout, routing and authority. A stage only renders inside the
 * work area, from the services the shell hands it.
 */
export interface StudioStage<Services, StageId extends string = string> {
  /** Stable identifier; it is the first segment of the stage address. */
  readonly id: StageId;
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
export interface StudioModule<
  Services,
  DocumentServices = Services,
  StageId extends string = string,
> {
  /** Stable identifier of the module. */
  readonly id: string;
  /** Version of the module's own contract with the shell. */
  readonly version: string;
  /** Verbs and schemas the installed manifest must declare. */
  readonly requires: ModuleRequirements;
  /** Stages the module adds, in navigation order. */
  readonly stages: readonly StudioStage<Services, StageId>[];
  /** Project-tree node providers, without a second persistence owner. */
  readonly explorer?: readonly ModuleExplorer<Services>[];
  /** Work-area views selected by original document schema. */
  readonly editors?: readonly ModuleEditor<Services, DocumentServices>[];
  /** Evidence inspector sections selected by original schema. */
  readonly inspectors?: readonly ModuleDocumentView<DocumentServices>[];
  /** Declarative command-search and menu entries. */
  readonly commands?: readonly ModuleCommand[];
  /** Job kinds whose execution retains the existing policy and worker owners. */
  readonly jobs?: readonly ModuleJob<Services>[];
  /** Settings projections; resolution and authority remain with the settings owner. */
  readonly settings?: readonly ModuleDocumentView<DocumentServices>[];
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
export interface ModuleRegistry<Services, DocumentServices = Services> {
  /** Registered modules in registration order. */
  readonly modules: readonly StudioModule<Services, DocumentServices>[];
  /** Every stage in registration order. */
  readonly stages: readonly StudioStage<Services>[];
  /** Admitted address paths; nothing outside this table is routable. */
  readonly paths: ReadonlyMap<string, StageDestination>;
  /** Find a stage by identifier. */
  readonly stage: (id: string) => StudioStage<Services> | undefined;
  /** Find the module that registered a stage. */
  readonly owner: (stage: string) => StudioModule<Services, DocumentServices> | undefined;
  /** List what the registered modules require and the manifest does not declare. */
  readonly missing: (declared: ModuleRequirements) => readonly MissingRequirement[];
  /** Registered explorer providers in module order. */
  readonly explorer: readonly ModuleExplorer<Services>[];
  /** Registered document editors in module order. */
  readonly editors: readonly ModuleEditor<Services, DocumentServices>[];
  /** Registered evidence inspector sections in module order. */
  readonly inspectors: readonly ModuleDocumentView<DocumentServices>[];
  /** Registered menu commands in module order. */
  readonly commands: readonly ModuleCommand[];
  /** Registered local job kinds in module order. */
  readonly jobs: readonly ModuleJob<Services>[];
  /** Registered settings projections in module order. */
  readonly settings: readonly ModuleDocumentView<DocumentServices>[];
  /** Find the module responsible for a contribution without transferring authority. */
  readonly contributionOwner: (
    slot: ModuleSlot,
    id: string,
  ) => StudioModule<Services, DocumentServices> | undefined;
}

const contributionSlots: readonly ModuleSlot[] = [
  "explorer",
  "editors",
  "inspectors",
  "commands",
  "jobs",
  "settings",
];

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
export function createModuleRegistry<Services, DocumentServices = Services>(
  modules: readonly StudioModule<Services, DocumentServices>[],
): ModuleRegistry<Services, DocumentServices> {
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
  const explorer = Object.freeze(modules.flatMap((module) => module.explorer ?? []));
  const editors = Object.freeze(modules.flatMap((module) => module.editors ?? []));
  const inspectors = Object.freeze(modules.flatMap((module) => module.inspectors ?? []));
  const commands = Object.freeze(modules.flatMap((module) => module.commands ?? []));
  const jobs = Object.freeze(modules.flatMap((module) => module.jobs ?? []));
  const settings = Object.freeze(modules.flatMap((module) => module.settings ?? []));
  const contributionOwners = new Map<
    ModuleSlot,
    ReadonlyMap<string, StudioModule<Services, DocumentServices>>
  >();
  for (const slot of contributionSlots) {
    const entries = modules.flatMap((module) =>
      (module[slot] ?? []).map((entry) => [entry.id, module] as const),
    );
    for (const [id] of entries) requireIdentifier(slot, id);
    requireDistinct(
      slot,
      entries.map(([id]) => id),
    );
    contributionOwners.set(slot, new Map(entries));
  }
  for (const command of commands) {
    const action = command.action;
    if (action.kind === "navigate") {
      const path = `/${action.stage}${action.instrument === null ? "" : `/${action.instrument}`}`;
      if (!paths.has(path)) throw new Error("Command targets an unregistered address");
    } else if (!jobs.some((job) => job.id === action.job)) {
      throw new Error("Command targets an unregistered job");
    }
  }
  return Object.freeze({
    modules: Object.freeze([...modules]),
    stages: Object.freeze(stages),
    paths,
    explorer,
    editors,
    inspectors,
    commands,
    jobs,
    settings,
    contributionOwner: (slot: ModuleSlot, id: string) => contributionOwners.get(slot)?.get(id),
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
