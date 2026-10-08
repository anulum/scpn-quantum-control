// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — linked public workflow graph editor tests

import { readFileSync } from "node:fs";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { readJson, writeJson } from "../../shared/contracts";
import { WorkflowEditor } from "./WorkflowEditor";
import { parseWorkflow, workflowDocument, WorkflowRefusal } from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});
function graph() {
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  return parseWorkflow(corpus["workflow"]);
}
it("graph and JSON keep the same original stages, ports and six cells before the real save callback", async () => {
  const definition = graph(),
    saved: WorkflowDefinition[] = [];
  render(
    <WorkflowEditor
      definition={definition}
      disabled={false}
      onCompose={async () => definition}
      onSave={async (original) => {
        saved.push(original);
      }}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
  await waitFor(() => expect(screen.getByText(/6 exact cells/)).toBeTruthy());
  expect(screen.getByRole("list", { name: "Original workflow dependency graph" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
  await waitFor(() => expect(saved).toHaveLength(1));
  expect(writeJson(workflowDocument(saved[0] as WorkflowDefinition))).toBe(
    writeJson(workflowDocument(definition)),
  );
});
it("cycle refuses before saving and leaves the original supplied graph unchanged", async () => {
  const definition = graph(),
    before = writeJson(workflowDocument(definition));
  let writes = 0;
  render(
    <WorkflowEditor
      definition={definition}
      disabled={false}
      onCompose={async () => definition}
      onSave={async () => {
        writes++;
      }}
    />,
  );
  const doc = readJson(before) as Record<string, unknown>,
    body = doc["body"] as Record<string, unknown>;
  const stages = body["stages"] as Record<string, unknown>[];
  (stages[1] as Record<string, unknown>)["depends_on"] = ["trace"];
  fireEvent.change(screen.getByRole("textbox", { name: "Workflow JSON" }), {
    target: { value: writeJson(doc) },
  });
  fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
  await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(/cycle/));
  expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(writes).toBe(0);
  expect(writeJson(workflowDocument(definition))).toBe(before);
});

it("retains a replacement source when an earlier real graph composition completes", async () => {
  const original = graph();
  const wire = readJson(writeJson(workflowDocument(original))) as Record<string, unknown>;
  (wire["body"] as Record<string, unknown>)["workflow_id"] = "replacement-compile-source";
  const replacement = parseWorkflow(wire);
  let release: (() => void) | undefined;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const compose = async () => {
    await gate;
    return original;
  };
  const saved: WorkflowDefinition[] = [];
  const save = async (definition: WorkflowDefinition) => {
    saved.push(definition);
  };
  const view = render(
    <WorkflowEditor definition={original} disabled={false} onCompose={compose} onSave={save} />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Compose local workflow" }));
  view.rerender(
    <WorkflowEditor definition={replacement} disabled={false} onCompose={compose} onSave={save} />,
  );
  await act(async () => release?.());
  expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty(
    "value",
    writeJson(workflowDocument(replacement)),
  );
  expect(screen.getByText(/Workflow replacement-compile-source/)).toBeTruthy();
  expect(saved).toHaveLength(0);
});

it("composes the exact original graph from an empty editor without executing or saving", async () => {
  const definition = graph();
  const saved: WorkflowDefinition[] = [];
  render(
    <WorkflowEditor
      definition={null}
      disabled={false}
      onCompose={async () => definition}
      onSave={async (value) => {
        saved.push(value);
      }}
    />,
  );
  expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty("value", "");
  fireEvent.click(screen.getByRole("button", { name: "Compose local workflow" }));
  await waitFor(() => expect(screen.getByText(/6 exact cells/)).toBeTruthy());
  expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty(
    "value",
    writeJson(workflowDocument(definition)),
  );
  expect(saved).toHaveLength(0);
});

it("retains the latest edited text when an earlier real preview settles", async () => {
  const definition = graph();
  render(
    <WorkflowEditor
      definition={definition}
      disabled={false}
      onCompose={async () => definition}
      onSave={async () => {
        throw new Error("unexpected save");
      }}
    />,
  );
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Workflow JSON" }), {
      target: { value: "new unsaved source" },
    });
  });
  expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty(
    "value",
    "new unsaved source",
  );
  expect(screen.queryByRole("list", { name: "Original workflow dependency graph" })).toBeNull();
  expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
    "disabled",
    true,
  );
});

it.each(["complete", "refused"])(
  "does not label a replaced graph with a prior save that %s",
  async (outcome) => {
    const definition = graph();
    const document = readJson(writeJson(workflowDocument(definition))) as Record<string, unknown>;
    (document["body"] as Record<string, unknown>)["workflow_id"] = "replacement-save-source";
    const replacement = parseWorkflow(document);
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const save = async () => {
      await gate;
      if (outcome === "refused") throw new WorkflowRefusal("old source transaction refused");
    };
    const compose = async () => definition;
    const view = render(
      <WorkflowEditor definition={definition} disabled={false} onCompose={compose} onSave={save} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
    view.rerender(
      <WorkflowEditor
        definition={replacement}
        disabled={false}
        onCompose={compose}
        onSave={save}
      />,
    );
    await act(async () => release?.());
    expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty(
      "value",
      writeJson(workflowDocument(replacement)),
    );
    expect(screen.getByRole("status").textContent).toBe("Preview the original graph before saving");
  },
);

it.each(["compose", "save"])(
  "ignores the prior %s completion after actual unmount",
  async (operation) => {
    const definition = graph();
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const compose = async () => {
      await gate;
      return definition;
    };
    const save = async () => {
      await gate;
    };
    const view = render(
      <WorkflowEditor definition={definition} disabled={false} onCompose={compose} onSave={save} />,
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: operation === "compose" ? "Compose local workflow" : "Save workflow graph",
      }),
    );
    view.unmount();
    await act(async () => release?.());
    expect(screen.queryByRole("region", { name: "Workflow graph editor" })).toBeNull();
  },
);

it.each(["source-refusal", "storage-failure"])(
  "keeps original graph visible on %s",
  async (fault) => {
    const definition = graph();
    const before = writeJson(workflowDocument(definition));
    render(
      <WorkflowEditor
        definition={definition}
        disabled={false}
        onCompose={async () => definition}
        onSave={async () => {
          throw fault === "source-refusal"
            ? new WorkflowRefusal("source transaction changed")
            : new Error("original storage unavailable");
        }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Save workflow graph" }));
    await waitFor(() =>
      expect(screen.getByRole("status").textContent).toBe(
        fault === "source-refusal"
          ? "source transaction changed"
          : "Original workflow transaction refused; prior saved data retained",
      ),
    );
    expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty("value", before);
    expect(writeJson(workflowDocument(definition))).toBe(before);
  },
);

it("refuses malformed transport without replacing the original saved definition", async () => {
  const definition = graph();
  const saved: WorkflowDefinition[] = [];
  render(
    <WorkflowEditor
      definition={definition}
      disabled={false}
      onCompose={async () => definition}
      onSave={async (value) => {
        saved.push(value);
      }}
    />,
  );
  fireEvent.change(screen.getByRole("textbox", { name: "Workflow JSON" }), {
    target: { value: "{broken" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
  await waitFor(() =>
    expect(screen.getByRole("status").textContent).toBe(
      "Original graph JSON refused; prior saved data retained",
    ),
  );
  expect(saved).toHaveLength(0);
  expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
    "disabled",
    true,
  );
});

it("starts one workspace save when two activation events arrive before the pending render", async () => {
  const definition = graph();
  let release: (() => void) | undefined;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const started: WorkflowDefinition[] = [];
  render(
    <WorkflowEditor
      definition={definition}
      disabled={false}
      onCompose={async () => definition}
      onSave={async (value) => {
        started.push(value);
        await gate;
      }}
    />,
  );
  const save = screen.getByRole("button", { name: "Save workflow graph" });
  await act(async () => {
    fireEvent.click(save);
    fireEvent.click(save);
  });
  try {
    expect(started).toHaveLength(1);
    expect(save).toHaveProperty("disabled", true);
  } finally {
    await act(async () => release?.());
  }
  expect(save).toHaveProperty("disabled", false);
});

it.each(["current", "edited", "unmounted"])(
  "retains original data when unavailable WebCrypto rejects a %s preview",
  async (state) => {
    const definition = graph();
    const before = writeJson(workflowDocument(definition));
    const writes: WorkflowDefinition[] = [];
    const view = render(
      <WorkflowEditor
        definition={definition}
        disabled={false}
        onCompose={async () => definition}
        onSave={async (value) => {
          writes.push(value);
        }}
      />,
    );
    vi.stubGlobal("crypto", undefined);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
      if (state === "edited")
        fireEvent.change(screen.getByRole("textbox", { name: "Workflow JSON" }), {
          target: { value: "new unavailable-host source" },
        });
      else if (state === "unmounted") view.unmount();
    });
    expect(writes).toHaveLength(0);
    expect(writeJson(workflowDocument(definition))).toBe(before);
    if (state === "unmounted")
      expect(screen.queryByRole("region", { name: "Workflow graph editor" })).toBeNull();
    else if (state === "edited") {
      expect(screen.getByRole("textbox", { name: "Workflow JSON" })).toHaveProperty(
        "value",
        "new unavailable-host source",
      );
      expect(screen.getByRole("status").textContent).toBe(
        "Preview the original graph before saving",
      );
    } else {
      expect(screen.getByRole("status").textContent).toBe(
        "Original graph JSON refused; prior saved data retained",
      );
      expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
        "disabled",
        true,
      );
    }
  },
);

it("previews an original graph while omitted host actions keep composition and persistence unavailable", async () => {
  const definition = graph();
  render(<WorkflowEditor definition={definition} disabled={false} />);
  expect(screen.getByRole("button", { name: "Compose local workflow" })).toHaveProperty(
    "disabled",
    true,
  );
  expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
    "disabled",
    true,
  );
  fireEvent.click(screen.getByRole("button", { name: "Preview workflow graph" }));
  await waitFor(() => expect(screen.getByText(/6 exact cells/)).toBeTruthy());
  expect(
    parseWorkflow(
      readJson(
        (screen.getByRole("textbox", { name: "Workflow JSON" }) as HTMLTextAreaElement).value,
      ),
    ),
  ).toEqual(definition);
  expect(screen.getByRole("button", { name: "Save workflow graph" })).toHaveProperty(
    "disabled",
    true,
  );
});
