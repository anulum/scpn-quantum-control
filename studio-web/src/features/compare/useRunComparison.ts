// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — read-only comparison controller

import { useEffect, useRef, useState } from "react";
import type { RawCodec } from "../../shared/contracts";
import { maxArchiveBytes } from "../../shared/storage/workspaceArchive";
import { ExperimentRefusal } from "../experiments/kuramotoArtifacts";
import { compareImmutableRuns } from "./comparisonModel";
import type { ComparisonSnapshot, ImmutableRunComparison } from "./comparisonModel";
import {
  ComparisonSourceRefusal,
  projectComparisonSource,
  readComparisonArchive,
} from "./comparisonSources";
import type { ComparisonArchive } from "./comparisonSources";

/** Explicit baseline/candidate position; neither is a mutable saved workspace head. */
export type ComparisonSideIndex = 0 | 1;
/** Current input and independently admitted immutable selection. */
export interface ComparisonSide {
  /** Exact user input text; changing it does not overwrite an admitted archive. */ readonly json: string;
  /** Last completely admitted archive, or none. */ readonly admitted: ComparisonArchive | null;
  /** Explicit original immutable revision identity. */ readonly revisionHash: string;
  /** Explicit recorded run identity, or revision-only comparison. */ readonly runHash:
    | string
    | null;
}
/** One read-only comparison bound to exact original source archives and selections. */
export interface SelectedRunComparison {
  /** Original baseline projection. */ readonly baseline: ComparisonSnapshot;
  /** Original candidate projection. */ readonly candidate: ComparisonSnapshot;
  /** Complete semantic/key/time comparison. */ readonly result: ImmutableRunComparison;
  /** Exact baseline portable archive identity. */ readonly baselineArchiveDigest: string;
  /** Exact candidate portable archive identity. */ readonly candidateArchiveDigest: string;
}
/** Explicit read-only import/selection/compare operations; no storage writer or executor. */
export interface RunComparisonController {
  /** Baseline then candidate source. */ readonly sides: readonly [ComparisonSide, ComparisonSide];
  /** Previous successful comparison, retained after refused or stale input. */ readonly comparison: SelectedRunComparison | null;
  /** One bounded read/projection is in progress. */ readonly busy: boolean;
  /** Both current exact inputs and immutable selections are admitted. */ readonly canCompare: boolean;
  /** Source admission/refusal or retention message. */ readonly message: string;
  /** Edit one local source input, invalidating in-flight completions. */ edit(
    side: ComparisonSideIndex,
    json: string,
  ): void;
  /** Read bounded exact UTF-8 bytes without importing or writing storage. */ read(
    side: ComparisonSideIndex,
    file: File,
  ): Promise<void>;
  /** Admit all references before offering selections. */ inspect(
    side: ComparisonSideIndex,
  ): Promise<void>;
  /** Select a listed immutable revision without rebinding a run. */ selectRevision(
    side: ComparisonSideIndex,
    hash: string,
  ): void;
  /** Select a recorded run bound to the current revision, or metadata only. */ selectRun(
    side: ComparisonSideIndex,
    hash: string | null,
  ): void;
  /** Produce a new comparison without replacing any source bytes or saved data. */ compare(): Promise<void>;
}

function initial(json: string): ComparisonSide {
  return { json, admitted: null, revisionHash: "", runHash: null };
}
function defaultRun(
  admitted: ComparisonArchive,
  revisionHash: string,
  side: ComparisonSideIndex,
): string | null {
  const available = admitted.runs.filter((run) => run.revisionHash === revisionHash);
  return (side === 0 ? available.at(0) : available.at(-1))?.hash ?? null;
}

/** Keep original sources/previous comparison while ignoring every stale asynchronous completion. */
export function useRunComparison(
  sourceJson: string,
  rawCodecs: ReadonlyMap<string, RawCodec>,
): RunComparisonController {
  const [sides, setSides] = useState<readonly [ComparisonSide, ComparisonSide]>(() => [
    initial(sourceJson),
    initial(sourceJson),
  ]);
  const [comparison, setComparison] = useState<SelectedRunComparison | null>(null);
  const [message, setMessage] = useState(
    "Read both original archives before comparing. Saved revisions and results remain unchanged.",
  );
  const [busy, setBusy] = useState(false);
  const generation = useRef(0),
    live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      ++generation.current;
    };
  }, []);
  const update = (side: ComparisonSideIndex, action: (current: ComparisonSide) => ComparisonSide) =>
    setSides((prior) => (side === 0 ? [action(prior[0]), prior[1]] : [prior[0], action(prior[1])]));
  const invalidate = () => {
    ++generation.current;
    setBusy(false);
  };
  const current = (epoch: number) => live.current && generation.current === epoch;
  const operation = async (action: (epoch: number) => Promise<void>) => {
    const epoch = ++generation.current;
    setBusy(true);
    try {
      await action(epoch);
    } catch (cause: unknown) {
      if (current(epoch))
        setMessage(
          cause instanceof ComparisonSourceRefusal || cause instanceof ExperimentRefusal
            ? cause.message
            : "Comparison refused; previous admitted comparison and saved workspace retained.",
        );
    } finally {
      if (current(epoch)) setBusy(false);
    }
  };
  const edit = (side: ComparisonSideIndex, json: string) => {
    invalidate();
    update(side, (prior) => ({ ...prior, json }));
    setMessage(
      "Source input changed. Read it before comparing; the previous admitted comparison and saved workspace are retained.",
    );
  };
  return {
    sides,
    comparison,
    busy,
    message,
    canCompare: sides.every(
      (side) => side.admitted?.archive.preview.json === side.json && side.revisionHash !== "",
    ),
    edit,
    async read(side, file) {
      await operation(async (epoch) => {
        if (file.size > maxArchiveBytes)
          throw new ComparisonSourceRefusal("Comparison archive exceeds the 64 MiB import bound");
        const bytes = await file.arrayBuffer();
        if (!current(epoch)) return;
        let json: string;
        try {
          json = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
        } catch (cause: unknown) {
          throw new ComparisonSourceRefusal("Comparison archive file must be valid UTF-8", {
            cause,
          });
        }
        update(side, (prior) => ({ ...prior, json }));
        setMessage(
          "Original source bytes read locally. Complete archive admission is required; saved data unchanged.",
        );
      });
    },
    async inspect(side) {
      await operation(async (epoch) => {
        const admitted = await readComparisonArchive(sides[side].json, rawCodecs);
        if (!current(epoch)) return;
        const draft = admitted.archive.manifest.body["draft_ref"] as {
          readonly sha256: string;
        } | null;
        const revisionHash = draft?.sha256 ?? admitted.revisions.at(-1)?.hash ?? "";
        update(side, (prior) => ({
          ...prior,
          admitted,
          revisionHash,
          runHash: defaultRun(admitted, revisionHash, side),
        }));
        setMessage(
          revisionHash === ""
            ? "Archive admitted; no immutable revisions to compare."
            : "Original archive admitted. Select immutable revisions and recorded runs; saved data unchanged.",
        );
      });
    },
    selectRevision(side, hash) {
      invalidate();
      const admitted = sides[side].admitted;
      if (admitted === null || !admitted.revisions.some((revision) => revision.hash === hash)) {
        setMessage("Selected immutable revision is absent; prior selection retained.");
        return;
      }
      update(side, (prior) => ({
        ...prior,
        revisionHash: hash,
        runHash: defaultRun(admitted, hash, side),
      }));
      setMessage(
        "Immutable revision selected; compare again to replace the previous comparison. Saved data unchanged.",
      );
    },
    selectRun(side, hash) {
      invalidate();
      const selected = sides[side];
      if (
        selected.admitted === null ||
        (hash !== null &&
          !selected.admitted.runs.some(
            (run) => run.hash === hash && run.revisionHash === selected.revisionHash,
          ))
      ) {
        setMessage(
          "Recorded run is absent or belongs to another revision; prior selection retained.",
        );
        return;
      }
      update(side, (prior) => ({ ...prior, runHash: hash }));
      setMessage(
        "Recorded run selected; compare again to replace the previous comparison. Saved data unchanged.",
      );
    },
    async compare() {
      await operation(async (epoch) => {
        const [left, right] = sides;
        if (
          left.admitted === null ||
          right.admitted === null ||
          left.admitted.archive.preview.json !== left.json ||
          right.admitted.archive.preview.json !== right.json
        )
          throw new ComparisonSourceRefusal("Read both current original archives before comparing");
        const baseline = await projectComparisonSource(
          left.admitted,
          left.revisionHash,
          left.runHash,
        );
        const candidate = await projectComparisonSource(
          right.admitted,
          right.revisionHash,
          right.runHash,
        );
        const result = compareImmutableRuns(baseline, candidate);
        if (!current(epoch)) return;
        setComparison(
          Object.freeze({
            baseline,
            candidate,
            result,
            baselineArchiveDigest: left.admitted.archive.preview.archiveDigest,
            candidateArchiveDigest: right.admitted.archive.preview.archiveDigest,
          }),
        );
        setMessage(
          "Read-only comparison complete. Original revisions, raw values and saved workspace unchanged.",
        );
      });
    },
  };
}
