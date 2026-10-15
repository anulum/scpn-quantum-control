// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — historical public workflow executor types

import type {
  WorkflowExecutionOptions,
  WorkflowExecutionResult,
} from "../../src/shared/workflows/workflowExecution";

export declare function runLocalWorkflow(
  options: WorkflowExecutionOptions,
): Promise<WorkflowExecutionResult>;
