// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — snapshot comparison view tests

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SnapshotDiffView } from "./SnapshotDiffView";
import { sealedRecord } from "../../../test-support/snapshotDiffRecord";

afterEach(cleanup);

function choose(file: unknown): void {
  fireEvent.change(screen.getByLabelText("Open a snapshot comparison"), {
    target: { files: [file] },
  });
}
function record(text: string, name = "diff.json"): File {
  return new File([text], name, { type: "application/json" });
}

describe("Snapshot comparison view", () => {
  it("shows a verified comparison, its potential impact and why it is incomplete", async () => {
    render(<SnapshotDiffView open={"c".repeat(64)} />);
    choose(record(await sealedRecord()));
    const summary = await screen.findByRole("status", { name: "Comparison summary" });
    expect(summary.textContent).toBe(
      "0 added, 0 removed, 1 tombstoned, 1 changed; 3 unchanged; 1 relations differ; 1 nodes potentially impacted.",
    );
    expect(screen.getByText("The later snapshot is the one open here.")).toBeTruthy();
    expect(screen.getByRole("note").textContent).toBe(
      "The impact analysis is incomplete, so it is not exhaustive: declared relations were followed, not observed ones.",
    );
    const objects = screen.getByRole("table", { name: "Changed objects" });
    expect(objects.textContent).toContain("deleted (removed at 2026-11-23T00:00:00Z)");
    expect(objects.textContent).toContain("source_digest");
    expect(screen.getByRole("table", { name: "Changed relations" }).textContent).toContain(
      "removed",
    );
    expect(
      within(screen.getByRole("table", { name: "Potential impact" })).getByText(
        "definition:repo/left.py derives_from definition:repo/top.py",
      ),
    ).toBeTruthy();
  });

  it("states a complete analysis for observed relations, without claiming proof", async () => {
    render(<SnapshotDiffView open={null} />);
    choose(
      record(
        await sealedRecord((body) => {
          const diff = body["diff"] as Record<string, unknown>;
          diff["complete"] = true;
          diff["incomplete_reasons"] = [];
        }),
      ),
    );
    expect((await screen.findByRole("note")).textContent).toBe(
      "The impact follows only observed relations of complete extractions. It is potential impact, not a proof.",
    );
    expect(screen.getByText("The later snapshot is not the one open here.")).toBeTruthy();
  });

  it("reports a refused or unreadable record and keeps only the latest choice", async () => {
    render(<SnapshotDiffView open={null} />);
    choose(record("{not json"));
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The snapshot comparison was refused: it is not JSON.",
    );
    choose({ name: "broken.json", text: () => Promise.reject(new Error("unreadable")) });
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe(
        "The snapshot comparison could not be read.",
      ),
    );
    let release: (text: string) => void = () => {};
    const slow = {
      name: "slow.json",
      text: () => new Promise<string>((resolve) => (release = resolve)),
    };
    choose(slow);
    choose(record(await sealedRecord()));
    await screen.findByRole("status", { name: "Comparison summary" });
    release("{not json");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.change(screen.getByLabelText("Open a snapshot comparison"), {
      target: { files: [] },
    });
    expect(screen.getByRole("status", { name: "Comparison summary" })).toBeTruthy();
  });
});
