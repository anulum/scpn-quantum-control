// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — test-only compiled source notes module

import type { StudioModule } from "../../src/shared/contracts/module";
import type {
  WorkbenchServices,
  WorkspaceViewServices,
} from "../../src/shared/contracts/workbench";

/** Test-only module that consumes actual admitted review-fixture bytes without changing them. */
export const sourceNotesModule = {
  id: "source-notes",
  apiVersion: "2.0",
  requires: { verbs: ["compile"], schemas: [] },
  stages: [
    {
      id: "source-notes" as const,
      label: "Source notes",
      group: "context",
      instruments: [],
      render: () => (
        <section aria-label="Source notes">
          <h3>Source notes</h3>
          <p>Test-only source projection; no execution or evidence promotion.</p>
        </section>
      ),
    },
  ],
  commands: [
    {
      id: "open-source-notes",
      label: "Open source notes",
      action: { kind: "navigate", stage: "source-notes", instrument: null },
    },
  ],
  editors: [
    {
      kind: "document",
      id: "source-notes-editor",
      label: "Source notes",
      schemas: ["review_fixture.v1"],
      render: (_services: WorkspaceViewServices, document) => (
        <label>
          Source notes editor
          <textarea readOnly value={document.content} />
        </label>
      ),
    },
  ],
  inspectors: [
    {
      kind: "document",
      id: "source-notes-inspector",
      label: "Source notes inspector",
      schemas: ["review_fixture.v1"],
      render: (_services: WorkspaceViewServices, document) => (
        <label>
          Source notes inspector
          <textarea readOnly value={document.content} />
        </label>
      ),
    },
  ],
} satisfies StudioModule<WorkbenchServices, WorkspaceViewServices>;
