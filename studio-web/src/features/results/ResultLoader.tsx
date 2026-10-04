// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — explicit source result import and read-only projection

import { useEffect, useRef, useState } from "react";
import type { OwnedKernelOutcome } from "../../workers/kernelProtocol";
import type { ExperimentPlan } from "../experiments/experimentPlan";
import { inspectAnalyseProducer, inspectKuramotoResult } from "./resultSources";
import { ResultRefusal } from "./resultModel";
import type { ResultSnapshot } from "./resultModel";
import { ResultInspector } from "./ResultInspector";

/** Original source run supplied by the continuously mounted Workbench owner. */
export interface ResultLoaderProps {
  /** Original immutable plan, or no current source-bound plan. */ readonly plan?: ExperimentPlan | null;
  /** Original worker result, accepted only when disposed and bound to the plan. */ readonly outcome?: OwnedKernelOutcome | null;
}

/** Admit producer metadata before replacing a result; asynchronous stale imports cannot win. */
export function ResultLoader({ plan = null, outcome = null }: ResultLoaderProps) {
  const [json, setJson] = useState("");
  const [native, setNative] = useState<ResultSnapshot | null>(null), [imported, setImported] = useState<ResultSnapshot | null>(null);
  const [message, setMessage] = useState(""), [runMessage, setRunMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const generation = useRef(0), live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => { live.current = false; ++generation.current; };
  }, []);
  useEffect(() => {
    let current = true;
    setNative(null); setRunMessage("");
    if (plan !== null && outcome !== null) {
      void inspectKuramotoResult(plan, outcome).then(result => { if (current) setNative(result); }).catch((cause: unknown) => {
        if (current) setRunMessage(cause instanceof ResultRefusal ? cause.message : "Original run result unavailable; source and workspace retained.");
      });
    }
    return () => { current = false; };
  }, [plan, outcome]);
  const inspect = async () => {
    const epoch = ++generation.current;
    setBusy(true); setMessage("");
    try {
      const result = await inspectAnalyseProducer(json);
      if (live.current && generation.current === epoch) { setImported(result); setMessage("Original producer metadata admitted; saved workspace unchanged."); }
    } catch (cause: unknown) {
      if (live.current && generation.current === epoch) setMessage(cause instanceof ResultRefusal ? cause.message : "Producer result refused; previous result and saved workspace retained.");
    } finally { if (live.current && generation.current === epoch) setBusy(false); }
  };
  return <section aria-label="Source result inspector" className="qsp-workspace">
    <h4>Inspect original result values</h4>
    <p>The latest disposed local experiment is displayed with its original source. Paste an original executive analyse JSON export to inspect its producer-declared axes and raw values. This reads data locally and does not execute a script or submit a job.</p>
    {runMessage && <p role="status">{runMessage}</p>}
    {native === null && plan === null && <p>No current local run result. Run an experiment explicitly in Experiments, or inspect an existing analyse export.</p>}
    {native !== null && <ResultInspector key={native.sourceSha256} result={native} />}
    <label>Result producer JSON<textarea value={json} onChange={event => { ++generation.current; setBusy(false); setJson(event.target.value); }} /></label>
    <button type="button" disabled={busy || json === ""} onClick={() => { void inspect(); }}>Inspect producer result</button>
    <p role="status" aria-label="Result import status">{message}</p>
    {imported !== null && <ResultInspector key={imported.sourceSha256} result={imported} />}
  </section>;
}
