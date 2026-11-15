// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Run-stage panel of the optional local job service

import { useEffect, useRef } from "react";
import type { Fetch } from "../../shared/transport/localJobs";
import { useLocalServiceConnection } from "../../shared/transport/localService";
import type { LocalServiceConnection } from "../../shared/transport/localService";
import { AttemptSection, JobNotices, RequestFields, ServiceFields } from "./LocalJobsSections";
import { useLocalJobs } from "./useLocalJobs";

export { EXAMPLE_REQUEST, POLL_MILLISECONDS } from "./useLocalJobs";

/**
 * Start, follow, reconnect to and cancel durable local jobs.
 *
 * The service is optional and started separately; this panel only talks to it
 * on the loopback interface with the session token it printed. Closing or
 * reloading the page does not stop a job: entering its attempt id reconnects
 * and replays every event from the start.
 *
 * @param props - Component properties.
 * @param props.focus - Focus the panel after it mounts, for a deep link.
 * @param props.fetcher - Fetch implementation; the browser's own when omitted.
 * @param props.connection - The workbench's shared connection; the panel keeps its own when omitted.
 * @returns The panel.
 */
export default function LocalJobsPanel({
  focus = false,
  fetcher,
  connection,
}: {
  readonly focus?: boolean;
  readonly fetcher?: Fetch;
  readonly connection?: LocalServiceConnection;
}) {
  const own = useLocalServiceConnection();
  const active = connection ?? own;
  const jobs = useLocalJobs(active, fetcher);
  const section = useRef<HTMLElement>(null);
  useEffect(() => {
    if (focus && section.current !== null) section.current.focus();
  }, [focus]);
  return (
    <section
      ref={section}
      tabIndex={-1}
      aria-label="Local job service"
      className="qsp-panel"
      id="/experiments/local-jobs"
    >
      <h4>Local job service</h4>
      <p>
        Optional. Start it with{" "}
        <code>python -m scpn_quantum_control.studio_workspace.jobs.service --root DIR</code>; it
        prints its address and session token. Jobs keep running when this page closes. Only local
        verbs run there; nothing is submitted to a provider.
      </p>
      <ServiceFields connection={active} />
      <RequestFields jobs={jobs} />
      <JobNotices jobs={jobs} />
      <AttemptSection jobs={jobs} />
    </section>
  );
}
