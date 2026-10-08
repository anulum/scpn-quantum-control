// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — original workflow checkpoint public tests

import { webcrypto } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { instantiateKuramoto } from "../../panel/kuramoto";
import { canonicalBytes, canonicalDigest, readJson, writeJson } from "../../shared/contracts";
import { createLocalExperiment, readLocalExperiment } from "../experiments/experimentArchive";
import { localExperimentCodecs } from "../experiments/kuramotoArtifacts";
import { runLocalWorkflow } from "./workflowExecution";
import {
  createWorkflowJournal,
  maxWorkflowJournalBytes,
  parseWorkflowJournal,
  workflowJournalDocument,
} from "./workflowJournal";
import { parseWorkflow } from "./workflowModel";
import { buildWorkflowCells } from "./workflowSweep";

function original() {
  const corpus = readJson(
    readFileSync("../tests/data/studio_workflow/contract_cases.json", "utf8"),
  ) as Record<string, unknown>;
  return {
    definition: parseWorkflow(corpus["workflow"]),
    wire: corpus["original_checkpoint"] as Record<string, unknown>,
  };
}
function body(wire: Record<string, unknown>): Record<string, unknown> {
  return wire["body"] as Record<string, unknown>;
}
function rows(wire: Record<string, unknown>): Record<string, unknown>[] {
  return body(wire)["entries"] as Record<string, unknown>[];
}

describe("original workflow checkpoint admission", () => {
  it("refuses completion when a real retried parent changes a retained child's dependency", async () => {
    vi.stubGlobal("crypto", webcrypto);
    try {
      const wasmPath = process.env["STUDIO_EXPERIMENT_WASM_PATH"];
      if (wasmPath === undefined) throw new Error("Supply the actual shipped workflow WASM");
      const kernel = await instantiateKuramoto(new Uint8Array(readFileSync(wasmPath)));
      let current = await createLocalExperiment(
        kernel,
        {
          mode: "mean-field",
          omega: [0.2, 0.2],
          theta0: [0, 0.8],
          coupling: 1.4,
          dt: 0.01,
          steps: 4,
        },
        "original parent-plan checkpoint",
        localExperimentCodecs,
        "native journal dependency test",
      );
      const source = await readLocalExperiment(current.json, localExperimentCodecs);
      const stage = {
        adapter: "local-kuramoto",
        verb: "validate",
        backend: "shipped-kuramoto-wasm-float64",
        parameters: {},
        inputs: [],
        outputs: [],
      };
      const definition = parseWorkflow({
        schema: "experiment_workflow.v1",
        body: {
          workflow_id: "changed-parent-plan",
          stages: [
            { ...stage, id: "parent", depends_on: [] },
            { ...stage, id: "child", depends_on: ["parent"] },
          ],
          sweep: { axes: [], seeds: ["0"], seed_binding: null, evaluation_budget: 3n },
        },
        extensions: {},
      });
      const options = {
        baseRevision: source.revisionHash,
        definition,
        kernel,
        rawCodecs: localExperimentCodecs,
        signal: new AbortController().signal,
        save: async (candidate: typeof current, prior: string) => {
          expect(prior).toBe(current.json);
          current = candidate;
        },
      };
      const first = await runLocalWorkflow({ ...options, sourceJson: current.json });
      expect(first.journal.state).toBe("complete");
      const retried = await runLocalWorkflow({
        ...options,
        sourceJson: current.json,
        planOptions: { memoryBudget: "2097152" },
      });
      expect(retried.journal.state).toBe("partial");
      expect(retried.journal.evaluations).toBe(3n);
      expect(retried.journal.entries).toHaveLength(3);
      expect(retried.journal.entries.slice(0, 2)).toEqual(first.journal.entries);
      expect(retried.journal.entries[2]?.output_digest).not.toBe(
        first.journal.entries[0]?.output_digest,
      );
      const claimed = readJson(writeJson(workflowJournalDocument(retried.journal))) as Record<
        string,
        unknown
      >;
      body(claimed)["state"] = "complete";
      await expect(parseWorkflowJournal(claimed, definition)).rejects.toThrow(
        "completed journal dependency has changed",
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("refuses an oversized UTF-8 journal even when its text character count fits", async () => {
    const { definition, wire } = original();
    const diagnostic = "😀".repeat(maxWorkflowJournalBytes / 4);
    expect(diagnostic.length).toBeLessThan(maxWorkflowJournalBytes);
    (wire["extensions"] as Record<string, unknown>)["source_diagnostic"] = diagnostic;
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow(
      "workflow journal exceeds byte bound",
    );
  });

  it("roundtrips the actual Python compiler record while retaining explicit partial state", async () => {
    const { definition, wire } = original();
    const journal = await parseWorkflowJournal(wire, definition);
    expect(journal.state).toBe("partial");
    expect(journal.evaluations).toBe(1n);
    expect(journal.entries).toHaveLength(1);
    const entry = journal.entries[0];
    expect(entry?.status).toBe("complete");
    const output = entry?.output as Record<string, unknown>;
    const result = output["result"] as Record<string, unknown>;
    expect((result["outputs"] as Record<string, unknown>)["execution_status"]).toBe(
      "emitted_not_executed",
    );
    expect(canonicalBytes("journal-fixture.v1", workflowJournalDocument(journal))).toEqual(
      canonicalBytes("journal-fixture.v1", wire),
    );
    const recovered = await parseWorkflowJournal(
      readJson(writeJson(workflowJournalDocument(journal))),
      definition,
    );
    expect(canonicalBytes("journal-fixture.v1", workflowJournalDocument(recovered))).toEqual(
      canonicalBytes("journal-fixture.v1", wire),
    );
    expect(Reflect.set(output, "digest", "changed")).toBe(false);
  });

  it("creates an empty partial history with exact original source and runtime identities", async () => {
    const { definition, wire } = original();
    const journal = await createWorkflowJournal(
      definition,
      body(wire)["source_fingerprint"] as string,
      body(wire)["runtime_fingerprint"] as string,
    );
    expect(journal.entries).toEqual([]);
    expect(journal.state).toBe("partial");
    expect(journal.evaluations).toBe(0n);
    expect(journal.source_fingerprint).toBe(body(wire)["source_fingerprint"]);
    expect(journal.runtime_fingerprint).toBe(body(wire)["runtime_fingerprint"]);
    expect(journal.workflow_digest).toBe(body(wire)["workflow_digest"]);
  });

  it.each([
    "output",
    "premature-complete",
    "duplicate",
    "count",
    "version",
    "state",
    "source",
    "runtime",
    "fields",
    "rows",
    "cell",
    "stage",
    "parents",
    "status",
    "reason",
    "evaluated",
    "output-absent",
  ])("refuses %s on an original producer checkpoint", async (fault) => {
    const { definition, wire } = original();
    const row = rows(wire)[0] as Record<string, unknown>;
    if (fault === "output") (row["output"] as Record<string, unknown>)["digest"] = "changed";
    else if (fault === "premature-complete") body(wire)["state"] = "complete";
    else if (fault === "duplicate") {
      rows(wire).push(row);
      body(wire)["evaluations"] = 2n;
    } else if (fault === "count") body(wire)["evaluations"] = 0n;
    else if (fault === "version") wire["schema"] = "experiment_workflow_journal.v2";
    else if (fault === "state") body(wire)["state"] = {};
    else if (fault === "source") body(wire)["source_fingerprint"] = "unavailable";
    else if (fault === "runtime") body(wire)["runtime_fingerprint"] = "unavailable";
    else if (fault === "fields") row["approved"] = true;
    else if (fault === "rows") body(wire)["entries"] = {};
    else if (fault === "cell") row["cell_id"] = "absent";
    else if (fault === "stage") row["stage_id"] = 0n;
    else if (fault === "parents") row["dependencies"] = { missing: null };
    else if (fault === "status") row["status"] = "succeeded";
    else if (fault === "reason") row["reason"] = "failed despite completed";
    else if (fault === "evaluated") row["evaluated"] = false;
    else row["output"] = null;
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow();
  });

  it("retains failed parent and blocked child without claiming a successful cell", async () => {
    const { definition, wire } = original();
    const row = rows(wire)[0] as Record<string, unknown>;
    row["status"] = "failed";
    row["reason"] = "Original compiler attempt failed";
    rows(wire).push({
      ...row,
      stage_id: "trace",
      dependencies: { source: null },
      status: "blocked",
      reason: "Original parent did not complete",
      output: null,
      output_digest: null,
      evaluated: false,
    });
    const journal = await parseWorkflowJournal(wire, definition);
    expect(journal.entries.map((entry) => entry.status)).toEqual(["failed", "blocked"]);
    expect(journal.evaluations).toBe(1n);
    expect(journal.state).toBe("partial");
    body(wire)["state"] = "complete";
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow("incomplete");
  });

  it("preserves cancellation, partial producer diagnostics and opaque lossless metadata", async () => {
    const { definition, wire } = original();
    const row = rows(wire)[0] as Record<string, unknown>;
    row["status"] = "cancelled";
    row["reason"] = "Original work cancelled";
    body(wire)["state"] = "cancelled";
    wire["extensions"] = {
      exact: 9007199254740993n,
      signed_zero: -0,
      nested: [{ note: "partial" }],
    };
    const journal = await parseWorkflowJournal(wire, definition);
    expect(journal.state).toBe("cancelled");
    expect(journal.entries[0]?.output).toEqual(row["output"]);
    expect(Object.is(journal.extensions["signed_zero"], -0)).toBe(true);
    expect(canonicalBytes("journal-fixture.v1", workflowJournalDocument(journal))).toEqual(
      canonicalBytes("journal-fixture.v1", wire),
    );
  });

  it("refuses child completion without the original completed parent and matching digest", async () => {
    const { definition, wire } = original();
    const source = rows(wire)[0] as Record<string, unknown>;
    const child = { ...source, stage_id: "trace", dependencies: { source: null as string | null } };
    rows(wire).push(child);
    body(wire)["evaluations"] = 2n;
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow("dependency");
    child.dependencies.source = source["output_digest"] as string;
    source["status"] = "failed";
    source["reason"] = "Original failure";
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow("dependency");
  });

  it("refuses history imported against a different current graph", async () => {
    const { definition, wire } = original();
    const changed = { ...definition, workflow_id: "other-workflow" };
    await expect(parseWorkflowJournal(wire, changed)).rejects.toThrow("another workflow");
    expect((await buildWorkflowCells(definition)).map((cell) => cell.id)).toHaveLength(6);
  });

  it("validates the original output digest even for partial diagnostics", async () => {
    const { definition, wire } = original();
    const row = rows(wire)[0] as Record<string, unknown>;
    row["status"] = "interrupted";
    row["reason"] = "Terminal proof was not received";
    row["output"] = { events: [{ status: "running" }] };
    row["output_digest"] = await canonicalDigest("studio.workflow-output.v1", row["output"]);
    const journal = await parseWorkflowJournal(wire, definition);
    expect(journal.entries[0]?.status).toBe("interrupted");
    row["output_digest"] = null;
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow("SHA-256");
  });
});

it.each(["extensions-object", "empty-reason", "oversized-reason", "absent-output-digest"])(
  "refuses original journal corruption: %s",
  async (fault) => {
    const { definition, wire } = original();
    const row = rows(wire)[0] as Record<string, unknown>;
    if (fault === "extensions-object") wire["extensions"] = [];
    else {
      row["status"] = "failed";
      row["reason"] = "Original incomplete diagnostic";
      if (fault === "empty-reason") row["reason"] = "";
      else if (fault === "oversized-reason") row["reason"] = "x".repeat(2049);
      else row["output"] = null;
    }
    await expect(parseWorkflowJournal(wire, definition)).rejects.toThrow();
  },
);
