// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — read-only native comparison controller

import { useEffect, useRef, useState } from "react";
import { resultLimits } from "../../shared/results/resultModel";
import { MAX_COMPARISON_BYTES, parseMethodComparison } from "./methodComparisonArtifact";
import type { MethodComparisonArtifact } from "./methodComparisonArtifact";

/** Native File/Blob byte interface; importing content never installs executable code. */
export interface MethodComparisonFile {
  /** Original byte length, checked before reading. */ readonly size: number;
  /** Original browser or native File byte operation. */ arrayBuffer(): Promise<ArrayBuffer>;
}

/** Decode bounded original UTF-8 bytes, retaining BOM and refusing malformed encoding. */
async function readOriginalFile(file: MethodComparisonFile): Promise<string> {
  const bytes = await file.arrayBuffer();
  if (bytes.byteLength > MAX_COMPARISON_BYTES)
    throw new Error("Method comparison file refused: actual byte limit exceeded.");
  return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
}
/** Independent read-only controller; no workspace storage, numerical solver or provider is called. */
export interface MethodComparisonController {
  /** Unsaved local import text. */ readonly draft: string;
  /** Last completely admitted source; malformed replacement preserves it. */ readonly admitted: MethodComparisonArtifact | null;
  /** Current original read/digest is pending. */ readonly pending: boolean;
  /** Last observable replacement refusal. */ readonly error: string | null;
  /** Display-only selected repetition, empty for aggregate view. */ readonly selected: string;
  /** Bounded visible page, independent of source population. */ readonly page: number;
  /** Edit local unsaved text, invalidating any late prior read. */ edit(text: string): void;
  /** Admit original native text/File with replacement and unmount fences. */ inspect(
    source: string | MethodComparisonFile,
  ): Promise<void>;
  /** Inspect an original repetition without changing an aggregate. */ select(
    identity: string,
  ): void;
  /** Advance only the visible original-row page. */ nextPage(): void;
  /** Return to the prior visible page. */ previousPage(): void;
}

/** Retain prior evidence on refusal and discard late File/digest completions after replacement or unmount. */
export function useMethodComparison(): MethodComparisonController {
  const [draft, setDraft] = useState(""),
    [admitted, setAdmitted] = useState<MethodComparisonArtifact | null>(null),
    [pending, setPending] = useState(false),
    [error, setError] = useState<string | null>(null),
    [selected, setSelected] = useState(""),
    [page, setPage] = useState(0);
  const mounted = useRef(true),
    ticket = useRef(0);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      ticket.current++;
    };
  }, []);
  function edit(text: string): void {
    ticket.current++;
    setDraft(text);
    setError(null);
    setPending(false);
  }
  async function inspect(source: string | MethodComparisonFile): Promise<void> {
    const current = ++ticket.current;
    const active = () => mounted.current && current === ticket.current;
    setPending(true);
    setError(null);
    try {
      if (typeof source !== "string" && source.size > MAX_COMPARISON_BYTES)
        throw new Error("Method comparison file refused: four mebibyte input limit exceeded.");
      const original = typeof source === "string" ? source : await readOriginalFile(source);
      if (!active()) return;
      setDraft(original);
      const result = await parseMethodComparison(original);
      if (!active()) return;
      if (!result.ok) {
        setError(result.message);
        return;
      }
      setAdmitted(result.value);
      setSelected(result.value.body.comparison.selected_run_id ?? "");
      setPage(0);
    } catch {
      if (active())
        setError(
          "Method comparison file refused; previous evidence and workspace remain unchanged.",
        );
    } finally {
      if (active()) setPending(false);
    }
  }
  return {
    draft,
    admitted,
    pending,
    error,
    selected,
    page,
    edit,
    inspect,
    select: setSelected,
    nextPage: () =>
      setPage((current) =>
        Math.min(
          current + 1,
          Math.max(
            0,
            Math.ceil((admitted?.body.comparison.runs.length ?? 0) / resultLimits.tableRows) - 1,
          ),
        ),
      ),
    previousPage: () => setPage((current) => Math.max(0, current - 1)),
  };
}
