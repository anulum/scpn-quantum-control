// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — genuine compiled workflow estimate with an explicit host barrier

import type { StudioModule } from "../src/shared/contracts/module";
import type { WorkbenchServices, WorkspaceViewServices } from "../src/shared/contracts/workbench";
import { workbenchRegistry } from "../src/app/moduleRegistry";
import { requirePresent } from "../test-support/requirePresent";

/** A compiled host job that releases its real original estimate only after an explicit barrier. */
export interface HeldWorkflowEstimate {
  /** Actual static module, never supplied by an archive or a URL. */ readonly module: StudioModule<
    WorkbenchServices,
    WorkspaceViewServices
  >;
  /** Resolves after the original graph owner has actually produced its estimate. */ readonly entered: Promise<void>;
  /** Release that unchanged original estimate. */ release(): void;
  /** Number of real estimator invocations. */ readonly estimates: number;
  /** Number of calls to the original execution owner. */ readonly runs: number;
}

/** Hold the public estimator's actual result to exercise host lifecycle changes without faking data. */
export function heldWorkflowEstimate(): HeldWorkflowEstimate {
  const original = requirePresent(
    workbenchRegistry.jobs.find((job) => job.id === "local-workflow"),
  );
  let markEntered: (() => void) | undefined;
  const entered = new Promise<void>((resolve) => {
    markEntered = resolve;
  });
  let release: (() => void) | undefined;
  const barrier = new Promise<void>((resolve) => {
    release = resolve;
  });
  let estimates = 0,
    runs = 0;
  return {
    entered,
    release: () => requirePresent(release)(),
    get estimates() {
      return estimates;
    },
    get runs() {
      return runs;
    },
    module: {
      id: "held-workflow",
      apiVersion: "2.0",
      requires: { verbs: [], schemas: [] },
      stages: [],
      jobs: [
        {
          ...original,
          id: "held-workflow",
          label: "Run held original workflow",
          estimate: async (services, document) => {
            estimates++;
            const estimate = await original.estimate(services, document);
            requirePresent(markEntered)();
            await barrier;
            return estimate;
          },
          run: async (services, document, signal) => {
            runs++;
            await original.run(services, document, signal);
          },
        },
      ],
    },
  };
}
