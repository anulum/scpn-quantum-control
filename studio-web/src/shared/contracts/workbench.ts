// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — shared workbench service contract

import type { ReactNode } from "react";
import type { ExperimentRunController } from "../experiments/experimentRunService";
import type { WorkflowRunController } from "../workflows/workflowRunService";
import type { WorkspaceController } from "../workspace/workspaceService";
import type { RawCodec } from "./index";
import type { ModuleDocument } from "./module";

/** Admitted documents and persistence supplied to schema-bound views. */
export interface WorkspaceViewServices {
  /** The existing workspace controller; views do not open another store. */
  readonly workspace: WorkspaceController;
  /** Existing trusted verifiers used at the source admission boundary. */
  readonly codecs: ReadonlyMap<string, RawCodec>;
}

/** Services of a contribution that uses no shell service. */
export type NoServices = object;

/** Admitted sources: the workspace, its verifiers and the records it released. */
export interface SourceServices extends WorkspaceViewServices {
  /** Exact source records released only by complete workspace admission. */
  readonly documents: readonly ModuleDocument[];
}

/** Addresses that keep the requested navigation context. */
export interface NavigationServices {
  /** Encoded opaque navigation context, already validated by the shell. */
  readonly contextSuffix: string;
  /** Address of the Workspace stage carrying the requested context. */
  readonly workspaceHref: string;
}

/** What the shell composes into a stage from other modules' contributions. */
export interface StageCompositionServices {
  /** Registered work-area controls for a stage; no feature is imported by the shell. */
  readonly renderEditors: (stage: string) => ReactNode;
  /** Compatibility overview of the host, rendered on the Workspace stage. */
  readonly overview: (editors: ReactNode) => ReactNode;
}

/** The local experiment run the shell owns. */
export interface ExperimentServices {
  /** Local experiment run owned by the shell. */
  readonly experiment: ExperimentRunController;
}

/** The local workflow run the shell owns. */
export interface WorkflowServices {
  /** Local workflow run owned by the shell. */
  readonly workflow: WorkflowRunController;
}

/**
 * Everything the shell hands a stage; a stage reaches nothing else.
 *
 * A module types its contributions with only the service interfaces it uses,
 * so the compiler refuses any other access; the shell still passes this whole
 * set.
 */
export interface WorkbenchServices
  extends SourceServices,
    NavigationServices,
    StageCompositionServices,
    ExperimentServices,
    WorkflowServices {}
