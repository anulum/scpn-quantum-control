// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — shared workbench service contract

import type { ReactNode } from "react";
import type { ExperimentRunController } from "../experiments/useExperimentRun";
import type { WorkflowRunController } from "../workflows/useWorkflowRun";
import type { WorkspaceController } from "../workspace/useWorkspace";
import type { RawCodec } from "./index";
import type { ModuleDocument } from "./module";

/** Admitted documents and persistence supplied to schema-bound views. */
export interface WorkspaceViewServices {
  /** The existing workspace controller; views do not open another store. */
  readonly workspace: WorkspaceController;
  /** Existing trusted verifiers used at the source admission boundary. */
  readonly codecs: ReadonlyMap<string, RawCodec>;
}

/** What the shell hands every stage; a stage reaches nothing else. */
export interface WorkbenchServices extends WorkspaceViewServices {
  /** Exact source records released only by complete workspace admission. */
  readonly documents: readonly ModuleDocument[];
  /** Registered work-area controls for a stage; no feature is imported by the shell. */
  readonly renderEditors: (stage: string) => ReactNode;
  /** Compatibility overview of the host, rendered on the Workspace stage. */
  readonly overview: (editors: ReactNode) => ReactNode;
  /** Encoded opaque navigation context, already validated by the shell. */
  readonly contextSuffix: string;
  /** Local experiment run owned by the shell. */
  readonly experiment: ExperimentRunController;
  /** Local workflow run owned by the shell. */
  readonly workflow: WorkflowRunController;
  /** Address of the Workspace stage carrying the requested context. */
  readonly workspaceHref: string;
}
