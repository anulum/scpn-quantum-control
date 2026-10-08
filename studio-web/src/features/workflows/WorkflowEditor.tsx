// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — linked original workflow graph and JSON editor

import { useEffect, useRef, useState } from "react";
import { readJson, writeJson } from "../../shared/contracts";
import {
  parseWorkflow,
  topologicalOrder,
  WorkflowRefusal,
  workflowDocument,
} from "./workflowModel";
import type { WorkflowDefinition } from "./workflowModel";
import { buildWorkflowCells } from "./workflowSweep";

/** Original selected graph and workspace transaction owner supplied by the caller. */
export interface WorkflowEditorProps {
  /** Saved immutable graph, or null before composing. */ readonly definition: WorkflowDefinition | null;
  /** Actual operation/source ownership disables edits and saves. */ readonly disabled: boolean;
  /** Available original save action; omission keeps graph persistence unavailable. */ readonly onSave?:
    | ((definition: WorkflowDefinition) => Promise<void>)
    | undefined;
  /** Available original graph template, with no numerical parameter defaults. */ readonly onCompose?:
    | (() => Promise<WorkflowDefinition>)
    | undefined;
}

/** Edit one exact JSON source and display its admitted typed dependency graph from the same document. */
export function WorkflowEditor({ definition, disabled, onSave, onCompose }: WorkflowEditorProps) {
  const [text, setText] = useState(() =>
    definition === null ? "" : writeJson(workflowDocument(definition)),
  );
  const [admitted, setAdmitted] = useState<WorkflowDefinition | null>(definition);
  const [cellCount, setCellCount] = useState<number | null>(null);
  const [message, setMessage] = useState("Preview the original graph before saving");
  const [pending, setPending] = useState(false);
  const generation = useRef(0),
    live = useRef(true),
    operationPending = useRef(false);
  useEffect(() => {
    ++generation.current;
    setText(definition === null ? "" : writeJson(workflowDocument(definition)));
    setAdmitted(definition);
    setCellCount(null);
  }, [definition]);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      ++generation.current;
    };
  }, []);
  const preview = async (json: string) => {
    const epoch = ++generation.current;
    try {
      const original = parseWorkflow(readJson(json));
      const cells = await buildWorkflowCells(original);
      if (!live.current || epoch !== generation.current) return;
      setAdmitted(original);
      setCellCount(cells.length);
      setMessage("Original graph admitted; preview does not execute or save it");
    } catch (cause: unknown) {
      if (!live.current || epoch !== generation.current) return;
      setAdmitted(null);
      setCellCount(null);
      setMessage(
        cause instanceof WorkflowRefusal
          ? cause.message
          : "Original graph JSON refused; prior saved data retained",
      );
    }
  };
  const perform = async (operation: () => Promise<void>) => {
    if (disabled || operationPending.current) return;
    const epoch = generation.current;
    operationPending.current = true;
    setPending(true);
    try {
      await operation();
    } catch (cause: unknown) {
      if (live.current && epoch === generation.current)
        setMessage(
          cause instanceof WorkflowRefusal
            ? cause.message
            : "Original workflow transaction refused; prior saved data retained",
        );
    } finally {
      operationPending.current = false;
      if (live.current) setPending(false);
    }
  };
  return (
    <section aria-label="Workflow graph editor">
      <h4>Workflow graph and JSON</h4>
      <p>
        Declared ports retain their original schema, dtype, shape and unit. A matching declaration
        does not prove model equivalence.
      </p>
      <button
        type="button"
        disabled={disabled || pending || onCompose === undefined}
        onClick={
          onCompose === undefined
            ? undefined
            : () => {
                const epoch = generation.current;
                void perform(async () => {
                  const original = await onCompose();
                  if (!live.current || epoch !== generation.current) return;
                  const json = writeJson(workflowDocument(original));
                  setText(json);
                  await preview(json);
                });
              }
        }
      >
        Compose local workflow
      </button>
      <label>
        Workflow JSON
        <textarea
          aria-label="Workflow JSON"
          rows={12}
          value={text}
          disabled={disabled || pending}
          onChange={(event) => {
            ++generation.current;
            setText(event.currentTarget.value);
            setAdmitted(null);
            setCellCount(null);
          }}
        />
      </label>
      <button
        type="button"
        disabled={disabled || pending || text.length === 0}
        onClick={() => {
          void preview(text);
        }}
      >
        Preview workflow graph
      </button>
      <button
        type="button"
        disabled={disabled || pending || admitted === null || onSave === undefined}
        onClick={
          admitted === null || onSave === undefined
            ? undefined
            : () => {
                const epoch = generation.current;
                void perform(async () => {
                  await onSave(admitted);
                  if (live.current && epoch === generation.current)
                    setMessage(
                      "Original workflow graph saved; prior journals and evidence retained",
                    );
                });
              }
        }
      >
        Save workflow graph
      </button>
      <p role="status">{message}</p>
      {admitted !== null && (
        <div>
          <p>
            Workflow {admitted.workflow_id} · {admitted.stages.length} stages
            {cellCount === null ? "" : ` · ${cellCount} exact cells`} · evaluation budget{" "}
            {admitted.sweep.evaluation_budget.toString()}
          </p>
          <ol aria-label="Original workflow dependency graph">
            {topologicalOrder(admitted).map((id) => {
              const stage = admitted.stages.find(
                (item) => item.id === id,
              ) as WorkflowDefinition["stages"][number];
              return (
                <li key={id}>
                  <strong>{id}</strong> · {stage.adapter} / {stage.verb} / {stage.backend}
                  <p>
                    Parents:{" "}
                    {[
                      ...new Set([
                        ...stage.depends_on,
                        ...stage.inputs.map((input) => input.source_stage),
                      ]),
                    ].join(", ") || "none"}
                  </p>
                  <ul aria-label={`Ports for ${id}`}>
                    {stage.inputs.map((input) => (
                      <li key={input.parameter}>
                        Input {input.parameter} ← {input.source_stage}.{input.source_port} ·{" "}
                        {input.type.schema} · {input.type.dtype} · [
                        {input.type.shape.map(String).join(",")}] · {input.type.unit}
                      </li>
                    ))}
                    {stage.outputs.map((output) => (
                      <li key={output.name}>
                        Output {output.name} · {output.path.join(".")} · {output.type.schema} ·{" "}
                        {output.type.dtype} · [{output.type.shape.map(String).join(",")}] ·{" "}
                        {output.type.unit}
                      </li>
                    ))}
                  </ul>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </section>
  );
}
