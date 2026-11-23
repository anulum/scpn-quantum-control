// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Atlas graph, table and inspector over one view model

import { type KeyboardEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Unverifiable } from "../../shared/browser/Unverifiable";
import type { RawCodec } from "../../shared/contracts";
import {
  type AtlasEdge,
  type AtlasNode,
  type AtlasSnapshot,
  atlasSnapshotDigest,
} from "../../shared/contracts/atlas";
import { type AtlasQueryAnswer, atlasMaxDepth } from "../../shared/contracts/atlasQuery";
import { createParameterDraft } from "../../shared/parameters/parameterDraft";
import { parameterSourceFromArchive } from "../../shared/parameters/parameterRevision";
import {
  type SelectionState,
  resolveSelection,
  selectedObjectOf,
  useSelectionState,
  useSelectionStore,
} from "../../shared/selection/selectionStore";
import { admitWorkspaceArchive } from "../../shared/storage/workspaceArchive";
import {
  type AtlasOpeningTask,
  type AtlasSession,
  type AtlasSessionInfo,
  committedOrigin,
  openAtlasSession,
  openCommittedAtlasSession,
  openSnapshotSession,
  workspaceOverlayOrigin,
} from "./atlasSession";
import { atlasWorkspaceViewOf, overlayAtlasWorkspace } from "./workspaceOverlay";
import { EvidenceLineage, type LineageVerifier } from "./EvidenceLineage";
import {
  type LineageAdmission,
  type LineageRecord,
  admitCommittedEvidenceLineage,
  lineageRecordOf,
  verifyLineageRecord,
} from "./evidenceLineage";
import {
  type AtlasFilter,
  type AtlasLayout,
  type AtlasLens,
  type AtlasPlacedNode,
  type AtlasZoom,
  aggregateAtlasAnswer,
  atlasKeyTarget,
  atlasLensOfKind,
  atlasLenses,
  atlasTableRows,
  atlasViewHref,
  atlasViewQuery,
  atlasZoomLevels,
  automaticAtlasZoom,
  defaultAtlasFilter,
  layoutAtlasAnswer,
  lensOf,
  parseAtlasFilter,
} from "./atlasViewModel";

/** What is selected: one node or one edge, identified by its canonical key. */
type Selection =
  | { readonly kind: "node"; readonly id: string }
  | { readonly kind: "edge"; readonly key: string }
  | null;

/** A made selection: a node or a relation. */
type Selected = NonNullable<Selection>;

type Pins = ReadonlyMap<string, readonly [number, number]>;

function edgeKey(edge: AtlasEdge): string {
  return `${edge.source} ${edge.kind} ${edge.target}`;
}

function contextValue(contextSuffix: string, key: string): string | null {
  return new URLSearchParams(contextSuffix.replace(/^\?/, "")).get(key);
}

function Inspector({
  answer,
  selection,
  onSelect,
}: {
  readonly answer: AtlasQueryAnswer;
  readonly selection: Selection;
  readonly onSelect: (selection: Selected) => void;
}) {
  if (selection === null) return <p>Select a node or a relation to inspect it.</p>;
  if (selection.kind === "edge") {
    const edge = answer.edges.find((item) => edgeKey(item) === selection.key) as AtlasEdge;
    return (
      <dl aria-label="Selected relation">
        <dt>Relation</dt>
        <dd>{edge.kind}</dd>
        <dt>From</dt>
        <dd>{edge.source}</dd>
        <dt>To</dt>
        <dd>{edge.target}</dd>
        <dt>Derived</dt>
        <dd>{`${edge.derivationMethod} from ${edge.sourceRef}`}</dd>
        <dt>Evidence</dt>
        <dd>{edge.evidenceDigest ?? "none required"}</dd>
      </dl>
    );
  }
  const node = answer.nodes.find((item) => item.id === selection.id) as AtlasNode;
  const incident = answer.edges.filter(
    (edge) => edge.source === node.id || edge.target === node.id,
  );
  return (
    <section aria-label="Selected node">
      <dl>
        <dt>ID</dt>
        <dd>{node.id}</dd>
        <dt>Kind</dt>
        <dd>{node.kind}</dd>
        <dt>Source</dt>
        <dd>{node.sourceRef}</dd>
        <dt>Content digest</dt>
        <dd>
          {node.sourceDigest ??
            (node.nullReasons.find(([field]) => field === "source_digest")?.[1] as string)}
        </dd>
        <dt>Status</dt>
        <dd>{node.status}</dd>
      </dl>
      <h4>Relations shown</h4>
      <ul>
        {incident.map((edge) => (
          <li key={edgeKey(edge)}>
            <button type="button" onClick={() => onSelect({ kind: "edge", key: edgeKey(edge) })}>
              {edgeKey(edge)}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function shownSelection(answer: AtlasQueryAnswer, selection: Selection): Selection {
  if (selection?.kind === "node")
    return answer.nodes.some((node) => node.id === selection.id) ? selection : null;
  if (selection?.kind === "edge")
    return answer.edges.some((edge) => edgeKey(edge) === selection.key) ? selection : null;
  return null;
}

const groupSpacing = 200;

function KindGraph({
  answer,
  onZoom,
}: {
  readonly answer: AtlasQueryAnswer;
  readonly onZoom: (zoom: AtlasZoom) => void;
}) {
  const aggregate = aggregateAtlasAnswer(answer);
  const at = new Map(
    aggregate.groups.map((group, index) => [group.kind, 100 + index * groupSpacing]),
  );
  const width = Math.max(320, 100 + aggregate.groups.length * groupSpacing);
  return (
    <>
      <svg viewBox={`0 0 ${width} 200`} width="100%" style={{ maxWidth: width }}>
        <title>Atlas relations by kind</title>
        {aggregate.links
          .filter((link) => link.source !== link.target)
          .map((link) => (
            <line
              key={`${link.source} ${link.kind} ${link.target}`}
              data-link={`${link.source} ${link.kind} ${link.target}`}
              x1={at.get(link.source)}
              y1={80}
              x2={at.get(link.target)}
              y2={80}
              stroke="currentColor"
              strokeWidth={1 + Math.log2(link.count)}
            />
          ))}
        {aggregate.groups.map((group) => (
          <g key={group.kind} data-group={group.kind}>
            <circle cx={at.get(group.kind)} cy={80} r={10 + 2 * Math.sqrt(group.count)} />
            <foreignObject x={(at.get(group.kind) as number) - 90} y={130} width={180} height={32}>
              <button type="button" className="qsp-atlas-node" onClick={() => onZoom("nodes")}>
                {`${group.kind}: ${group.count}`}
              </button>
            </foreignObject>
          </g>
        ))}
      </svg>
      <ul aria-label="Atlas relations by kind">
        {aggregate.links.map((link) => (
          <li key={`${link.source} ${link.kind} ${link.target}`}>
            {`${link.source} ${link.kind} ${link.target}: ${link.count}`}
          </li>
        ))}
      </ul>
    </>
  );
}

function NodeGraph({
  answer,
  layout,
  shown,
  details,
  onSelect,
  onKey,
}: {
  readonly answer: AtlasQueryAnswer;
  readonly layout: AtlasLayout;
  readonly shown: Selection;
  readonly details: boolean;
  readonly onSelect: (selection: Selected) => void;
  readonly onKey: (event: KeyboardEvent<HTMLElement>) => void;
}) {
  const position = new Map(layout.placed.map((item) => [item.node.id, item]));
  const isSelected = (node: AtlasNode): boolean => shown?.kind === "node" && shown.id === node.id;
  return (
    <svg
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      width="100%"
      style={{ maxWidth: layout.width }}
    >
      <title>Atlas relations</title>
      {answer.edges.map((edge) => {
        const from = position.get(edge.source) as AtlasPlacedNode;
        const to = position.get(edge.target) as AtlasPlacedNode;
        return (
          <g key={edgeKey(edge)}>
            <line
              data-edge={edgeKey(edge)}
              data-selected={shown?.kind === "edge" && shown.key === edgeKey(edge)}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="currentColor"
            />
            {details ? (
              <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 4} fontSize={10}>
                {edge.kind}
              </text>
            ) : null}
          </g>
        );
      })}
      {layout.placed.map(({ node, x, y }) => (
        <g key={node.id} data-node={node.id} data-selected={isSelected(node)}>
          <circle cx={x} cy={y} r={8} />
          <foreignObject x={x + 12} y={y - 12} width={196} height={28}>
            <button
              type="button"
              className="qsp-atlas-node"
              aria-pressed={isSelected(node)}
              onClick={() => onSelect({ kind: "node", id: node.id })}
              onKeyDown={onKey}
            >
              {node.label}
            </button>
          </foreignObject>
          {details ? (
            <text x={x + 12} y={y + 26} fontSize={10}>{`${node.kind} · ${node.status}`}</text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

function pageAt(index: number): number {
  return index < 0 ? 0 : Math.floor(index / atlasTableRows);
}

function RowPager({
  label,
  page,
  total,
  onPage,
}: {
  readonly label: string;
  readonly page: number;
  readonly total: number;
  readonly onPage: (page: number) => void;
}) {
  const last = Math.max(0, Math.ceil(total / atlasTableRows) - 1);
  if (last === 0) return null;
  return (
    <nav aria-label={label}>
      <button type="button" disabled={page === 0} onClick={() => onPage(page - 1)}>
        Previous rows
      </button>
      <span>{`Rows ${page * atlasTableRows + 1}–${Math.min(total, (page + 1) * atlasTableRows)} of ${total}`}</span>
      <button type="button" disabled={page === last} onClick={() => onPage(page + 1)}>
        Next rows
      </button>
    </nav>
  );
}

function AtlasAnswerView({
  answer,
  focus,
  pinned,
  selection,
  trail,
  zoom,
  onZoom,
  onSelect,
  onFocus,
  onPin,
  lineageOf,
  verifyRecord,
  contextSuffix,
  draftOf,
}: {
  readonly answer: AtlasQueryAnswer;
  readonly focus: string | null;
  readonly pinned: Pins;
  readonly selection: Selection;
  readonly trail: readonly string[];
  readonly zoom: AtlasZoom | null;
  readonly onZoom: (zoom: AtlasZoom) => void;
  readonly onSelect: (selection: Selection) => void;
  readonly onFocus: (focus: string | null) => void;
  readonly onPin: (id: string, spot: readonly [number, number]) => void;
  readonly lineageOf: (node: AtlasNode) => LineageRecord | null;
  readonly verifyRecord: LineageVerifier;
  readonly contextSuffix: string;
  readonly draftOf: (node: AtlasNode) => ReactNode;
}) {
  const shown = shownSelection(answer, selection);
  const selectedNode =
    shown?.kind === "node"
      ? (answer.nodes.find((node) => node.id === shown.id) as AtlasNode)
      : null;
  const lineageRecord = selectedNode === null ? null : lineageOf(selectedNode);
  const nodeIndex = (id: string): number => answer.nodes.findIndex((node) => node.id === id);
  const edgeIndex = (key: string): number =>
    answer.edges.findIndex((edge) => edgeKey(edge) === key);
  // Table pages belong to one answer; a new answer opens on the page of the selection.
  const [paging, setPaging] = useState({ answer, nodes: 0, edges: 0 });
  const pages =
    paging.answer === answer
      ? paging
      : {
          answer,
          nodes: pageAt(shown?.kind === "node" ? nodeIndex(shown.id) : -1),
          edges: pageAt(shown?.kind === "edge" ? edgeIndex(shown.key) : -1),
        };
  const choose = (next: Selected): void => {
    setPaging({
      answer,
      nodes: next.kind === "node" ? pageAt(nodeIndex(next.id)) : pages.nodes,
      edges: next.kind === "edge" ? pageAt(edgeIndex(next.key)) : pages.edges,
    });
    onSelect(next);
  };
  const level = zoom ?? automaticAtlasZoom(answer);
  const step = atlasZoomLevels.indexOf(level);
  const isSelected = (node: AtlasNode): boolean => shown?.kind === "node" && shown.id === node.id;
  const edgeSelected = (edge: AtlasEdge): boolean =>
    shown?.kind === "edge" && shown.key === edgeKey(edge);
  const onKey = (event: KeyboardEvent<HTMLElement>): void => {
    const selected = shown?.kind === "node" ? shown.id : null;
    if (event.key === "Enter" && selected !== null) {
      onFocus(selected);
      event.preventDefault();
      return;
    }
    if (event.key === "Escape") {
      onFocus(null);
      return;
    }
    const target = atlasKeyTarget(answer, selected, event.key);
    if (target !== null) {
      choose({ kind: "node", id: target });
      event.preventDefault();
    }
  };
  const omissions = `${answer.omitted > 0 ? `; ${answer.omitted} more are not shown` : ""}${answer.omittedEdges > 0 ? `; ${answer.omittedEdges} relations are not shown` : ""}`;
  const layout = useMemo(() => layoutAtlasAnswer(answer, focus, pinned), [answer, focus, pinned]);
  return (
    <>
      <p role="status" aria-label="Atlas summary">
        {`Showing ${answer.nodes.length} of ${answer.total} nodes${omissions}.`}
      </p>
      <fieldset>
        <legend>Atlas zoom</legend>
        <button
          type="button"
          disabled={step === 0}
          onClick={() => onZoom(atlasZoomLevels[step - 1] as AtlasZoom)}
        >
          Zoom out
        </button>
        <span>{`Detail: ${level}`}</span>
        <button
          type="button"
          disabled={step === atlasZoomLevels.length - 1}
          onClick={() => onZoom(atlasZoomLevels[step + 1] as AtlasZoom)}
        >
          Zoom in
        </button>
      </fieldset>
      <ul aria-label="Atlas legend">
        {[...new Set(answer.nodes.map((node) => node.kind))].map((kind) => (
          <li
            key={kind}
            data-kind={kind}
          >{`${kind}: ${answer.nodes.filter((node) => node.kind === kind).length}`}</li>
        ))}
      </ul>
      <nav aria-label="Atlas breadcrumbs">
        {trail.map((id) => (
          <button type="button" key={id} onClick={() => choose({ kind: "node", id })}>
            {id}
          </button>
        ))}
      </nav>
      <section aria-label="Atlas graph" className="qsp-atlas-graph">
        {level === "kinds" ? (
          <KindGraph answer={answer} onZoom={onZoom} />
        ) : (
          <NodeGraph
            answer={answer}
            layout={layout}
            shown={shown}
            details={level === "details"}
            onSelect={choose}
            onKey={onKey}
          />
        )}
      </section>
      <RowPager
        label="Atlas node rows"
        page={pages.nodes}
        total={answer.nodes.length}
        onPage={(nodes) => setPaging({ ...pages, nodes })}
      />
      <div className="qsp-data-scroll">
        <table aria-label="Atlas nodes">
          <thead>
            <tr>
              <th scope="col">Node</th>
              <th scope="col">Kind</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {answer.nodes
              .slice(pages.nodes * atlasTableRows, (pages.nodes + 1) * atlasTableRows)
              .map((node) => (
                <tr key={node.id} aria-selected={isSelected(node)}>
                  <th scope="row">
                    <button type="button" onClick={() => choose({ kind: "node", id: node.id })}>
                      {node.label}
                    </button>
                  </th>
                  <td>{node.kind}</td>
                  <td>{node.status}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <RowPager
        label="Atlas relation rows"
        page={pages.edges}
        total={answer.edges.length}
        onPage={(edges) => setPaging({ ...pages, edges })}
      />
      <div className="qsp-data-scroll">
        <table aria-label="Atlas relations table">
          <thead>
            <tr>
              <th scope="col">From</th>
              <th scope="col">Relation</th>
              <th scope="col">To</th>
            </tr>
          </thead>
          <tbody>
            {answer.edges
              .slice(pages.edges * atlasTableRows, (pages.edges + 1) * atlasTableRows)
              .map((edge) => (
                <tr key={edgeKey(edge)} aria-selected={edgeSelected(edge)}>
                  <td>{edge.source}</td>
                  <td>
                    <button
                      type="button"
                      onClick={() => choose({ kind: "edge", key: edgeKey(edge) })}
                    >
                      {edge.kind}
                    </button>
                  </td>
                  <td>{edge.target}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <section aria-label="Atlas inspector">
        <Inspector answer={answer} selection={shown} onSelect={choose} />
        {lineageRecord !== null ? (
          <EvidenceLineage
            record={lineageRecord}
            verify={verifyRecord}
            contextSuffix={contextSuffix}
          />
        ) : null}
        {selectedNode !== null ? draftOf(selectedNode) : null}
        {shown?.kind === "node" ? (
          <>
            <button type="button" onClick={() => onFocus(shown.id)}>
              Show its neighbourhood
            </button>
            <button
              type="button"
              onClick={() => {
                const spot = layout.placed.find(
                  (item) => item.node.id === shown.id,
                ) as AtlasPlacedNode;
                onPin(shown.id, [spot.x, spot.y + 1]);
              }}
            >
              Pin its position
            </button>
          </>
        ) : null}
      </section>
      <ul aria-label="Atlas explanation">
        {answer.explanation.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </>
  );
}

/** The admitted workspace the Atlas overlays, as the workbench holds it. */
export interface AtlasWorkspaceSource {
  /** Exact archive text. */ readonly json: string;
  /** Trusted verifiers the archive is admitted with. */ readonly codecs: ReadonlyMap<
    string,
    RawCodec
  >;
}

/** Decide whether the parameter editor admits a revision; null means it does. */
export type AtlasDraftCheck = (
  workspace: AtlasWorkspaceSource,
  revision: string,
) => Promise<string | null>;

/**
 * Ask the parameter editor's own admission whether a revision opens as a draft.
 *
 * @param workspace - The admitted workspace.
 * @param revision - The revision hash.
 * @returns Null when the editor admits it, otherwise the editor's refusal.
 */
export async function parameterEditorAdmits(
  workspace: AtlasWorkspaceSource,
  revision: string,
): Promise<string | null> {
  try {
    createParameterDraft(
      await parameterSourceFromArchive(workspace.json, workspace.codecs, revision),
    );
    return null;
  } catch (error) {
    // The archive and parameter owners refuse with an Error that names the reason.
    return (error as Error).message;
  }
}

function RevisionDraft({
  node,
  workspace,
  draftHref,
  check,
}: {
  readonly node: AtlasNode;
  readonly workspace: AtlasWorkspaceSource;
  readonly draftHref: string | null;
  readonly check: AtlasDraftCheck;
}) {
  const revision = node.sourceDigest as string;
  const [state, setState] = useState<{
    readonly revision: string;
    readonly refusal: string | null;
  } | null>(null);
  useEffect(() => {
    let live = true;
    void check(workspace, revision).then((refusal) => {
      if (live) setState({ revision, refusal });
    });
    return () => {
      live = false;
    };
  }, [check, workspace, revision]);
  return (
    <section aria-label="Revision draft">
      {state === null || state.revision !== revision ? (
        <p role="status">Checking whether the parameter editor admits this revision…</p>
      ) : state.refusal === null ? (
        <>
          <p>
            The parameter editor admits this revision. Opening it starts a new draft; saving appends
            a child revision, and this revision stays as it is.
          </p>
          {draftHref !== null ? <a href={draftHref}>Open as a new parameter draft</a> : null}
        </>
      ) : (
        <p role="note">{`Read-only: the parameter editor refuses this revision's configuration (${state.refusal}). Its record is shown as committed to the workspace.`}</p>
      )}
    </section>
  );
}

function SharedSelection({
  shared,
  snapshot,
  digest,
  lens,
  lensHref,
}: {
  readonly shared: SelectionState;
  readonly snapshot: AtlasSnapshot;
  readonly digest: string;
  readonly lens: AtlasLens;
  readonly lensHref: (lens: AtlasLens) => string | null;
}) {
  if (shared.selected === null || shared.origin === null) return null;
  const resolution = resolveSelection(shared.selected, snapshot, digest);
  const home = resolution.state === "present" ? atlasLensOfKind(resolution.node.kind) : null;
  const href = home !== null && home !== lens ? lensHref(home) : null;
  return (
    <section aria-label="Shared selection">
      <p role="status" data-selection={resolution.state}>
        {resolution.state === "present"
          ? `Selected in ${shared.origin}: ${resolution.node.label} (${resolution.node.kind}).`
          : `Selected in ${shared.origin}: ${shared.selected.objectId}. ${resolution.explanation}`}
      </p>
      {href !== null ? (
        <a href={href}>{`Show it in the ${atlasLenses[home as AtlasLens].label} lens`}</a>
      ) : null}
    </section>
  );
}

/** Cancel of an opening that is not running. */
function nothingToCancel(): void {}

/** A pending opening or query; work in a worker can be cancelled. */
interface Pending {
  /** What is running, as the status line says it. */
  readonly label: string;
  /** Stop it, or null for main-thread work that cannot be interrupted. */
  readonly cancel: (() => void) | null;
}

/**
 * Browse an Atlas snapshot through one bounded, access-filtered view model.
 *
 * The snapshot committed with the build opens first; another snapshot pin can be opened from a
 * file. A pin above the worker threshold is admitted and queried in a worker, and its opening
 * and its queries can be cancelled while they run.
 *
 * @param props - The route's instrument (its lens), its context suffix, and the openers to use.
 * @returns The graph, the node and relation tables, the inspector and the answer's explanation.
 */
export default function AtlasView({
  instrument,
  contextSuffix,
  open = openCommittedAtlasSession,
  openFile = (file: File) => openAtlasSession(file, file.name),
  lineage = admitCommittedEvidenceLineage,
  verifyRecord = verifyLineageRecord,
  workspace = null,
  draftHref = null,
  checkDraft = parameterEditorAdmits,
}: {
  readonly instrument: string | null;
  readonly contextSuffix: string;
  readonly open?: () => AtlasOpeningTask;
  readonly openFile?: (file: File) => AtlasOpeningTask;
  readonly lineage?: () => LineageAdmission;
  readonly verifyRecord?: LineageVerifier;
  readonly workspace?: AtlasWorkspaceSource | null;
  readonly draftHref?: string | null;
  readonly checkDraft?: AtlasDraftCheck;
}) {
  const lens = lensOf(instrument);
  const admittedLineage = useMemo(() => lineage(), [lineage]);
  const store = useSelectionStore();
  const shared = useSelectionState(store);
  const [session, setSession] = useState<AtlasSession | null>(null);
  // The committed snapshot the workspace overlay is built on.
  const [committed, setCommitted] = useState<AtlasSession | null>(null);
  const previousOverlay = useRef<AtlasSnapshot | null>(null);
  // Digest of the public pin the shown overlay is built on; links name it, never the overlay.
  const overlayPin = useRef<string | null>(null);
  // Why the workspace is not overlaid; a query never clears it, a later overlay does.
  const [overlayRefusal, setOverlayRefusal] = useState<string | null>(null);
  const [committedRefusal, setCommittedRefusal] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const requestedFilter = contextValue(contextSuffix, "filter");
  const admittedFilter = useMemo(() => parseAtlasFilter(requestedFilter), [requestedFilter]);
  const [filter, setFilter] = useState<AtlasFilter>(
    admittedFilter.ok ? admittedFilter.filter : defaultAtlasFilter,
  );
  const [selection, setSelection] = useState<Selection>(() => {
    const selected = store?.current().selected ?? null;
    return selected === null ? null : { kind: "node", id: selected.objectId };
  });
  const [trail, setTrail] = useState<readonly string[]>([]);
  const [pinned, setPinned] = useState<Pins>(new Map());
  const [zoom, setZoom] = useState<AtlasZoom | null>(null);
  // The requested snapshot an explicitly opened pin overrode; a later request is checked again.
  const [overridden, setOverridden] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AtlasQueryAnswer | { readonly refused: string } | null>(
    null,
  );
  const current = useRef<AtlasSession | null>(null);
  current.current = session;
  const contextSuffixRef = useRef(contextSuffix);
  contextSuffixRef.current = contextSuffix;
  // The latest file opening; an earlier one that finishes later is discarded.
  const opening = useRef<{ readonly sequence: number; readonly cancel: () => void }>({
    sequence: 0,
    cancel: nothingToCancel,
  });
  useEffect(() => {
    setFilter(admittedFilter.ok ? admittedFilter.filter : defaultAtlasFilter);
  }, [admittedFilter]);
  useEffect(() => {
    const task = open();
    let live = true;
    setPending({
      label: "Verifying the committed Atlas snapshot…",
      cancel: task.tier === "worker" ? task.cancel : null,
    });
    void task.result.then((result) => {
      if (!live) return;
      setPending(null);
      if (result === null) setNotice("Opening the committed snapshot was cancelled.");
      else if (!result.ok) setCommittedRefusal(result.reason);
      else {
        current.current?.dispose();
        setSession(result.session);
        setCommitted(result.session);
      }
    });
    return () => {
      live = false;
      task.cancel();
    };
  }, [open]);
  const workspaceJson = workspace?.json ?? null;
  const workspaceCodecs = workspace?.codecs ?? null;
  const workspaceSource = useMemo(
    () =>
      workspaceJson === null || workspaceCodecs === null
        ? null
        : { json: workspaceJson, codecs: workspaceCodecs },
    [workspaceJson, workspaceCodecs],
  );
  useEffect(() => {
    const base = committed?.snapshot ?? null;
    const shownIsOurs = (shown: AtlasSession | null): boolean =>
      shown === null ||
      shown.info.origin === committedOrigin ||
      shown.info.origin === workspaceOverlayOrigin;
    if (committed === null || base === null) return;
    if (workspaceJson === null || workspaceCodecs === null) {
      setOverlayRefusal(null);
      setSession((shown) => (shownIsOurs(shown) ? committed : shown));
      return;
    }
    let live = true;
    void (async () => {
      try {
        const archive = await admitWorkspaceArchive(workspaceJson, workspaceCodecs);
        const overlay = await overlayAtlasWorkspace(
          base,
          atlasWorkspaceViewOf(archive),
          previousOverlay.current,
        );
        const digest = await atlasSnapshotDigest(overlay);
        if (!live) return;
        previousOverlay.current = overlay;
        overlayPin.current = committed.info.digest;
        setOverlayRefusal(null);
        setSession((shown) =>
          shownIsOurs(shown) ? openSnapshotSession(overlay, digest, workspaceOverlayOrigin) : shown,
        );
      } catch (error) {
        if (live)
          // The archive and overlay owners refuse with an Error that names the reason.
          setOverlayRefusal(
            `Your workspace is not shown in the Atlas: ${(error as Error).message}.`,
          );
      }
    })();
    return () => {
      live = false;
    };
  }, [committed, workspaceJson, workspaceCodecs]);
  useEffect(
    () => () => {
      opening.current.cancel();
      opening.current = { sequence: opening.current.sequence + 1, cancel: nothingToCancel };
      current.current?.dispose();
    },
    [],
  );
  const { text, focus, depth } = filter;
  useEffect(() => {
    if (session === null) return;
    const task = session.query(atlasViewQuery(lens, text, focus, depth));
    let live = true;
    if (session.info.tier === "worker")
      setPending({ label: "Querying the snapshot in a worker…", cancel: task.cancel });
    void task.outcome.then((outcome) => {
      if (!live) return;
      setPending(null);
      if (outcome === null) {
        setNotice("The query was cancelled; the previous answer stays shown.");
        return;
      }
      setNotice(null);
      setAnswer(outcome.ok ? outcome.value : { refused: outcome.message });
    });
    return () => {
      live = false;
    };
  }, [session, lens, text, focus, depth]);
  const shownDigest = session?.info.digest ?? null;
  const shownSnapshot = session?.snapshot ?? null;
  const requestedSelection = filter.selected;
  // A linked public selection is restored once its snapshot is open.
  useEffect(() => {
    if (requestedSelection === null || shownSnapshot === null || shownDigest === null) return;
    const node = shownSnapshot.nodes.find((item) => item.id === requestedSelection);
    if (node?.access !== "public") return;
    setSelection({ kind: "node", id: node.id });
    store?.select(selectedObjectOf(shownDigest, node), "atlas");
  }, [requestedSelection, shownSnapshot, shownDigest, store]);
  // Another view's selection is shown here; the Atlas never re-announces it.
  useEffect(() => {
    if (shared.selected === null || shared.origin === null || shared.origin === "atlas") return;
    setSelection({ kind: "node", id: shared.selected.objectId });
  }, [shared]);
  const openPin = (file: File): void => {
    opening.current.cancel();
    const task = openFile(file);
    const sequence = opening.current.sequence + 1;
    opening.current = { sequence, cancel: task.cancel };
    setNotice(null);
    setPending({
      label: `Verifying ${file.name}${task.tier === "worker" ? " in a worker" : ""}…`,
      cancel: task.tier === "worker" ? task.cancel : null,
    });
    void task.result.then((result) => {
      if (sequence !== opening.current.sequence) {
        if (result?.ok) result.session.dispose();
        return;
      }
      setPending(null);
      if (result === null) {
        setNotice(`Opening ${file.name} was cancelled; the open snapshot stays.`);
        return;
      }
      if (!result.ok) {
        setNotice(`${file.name} was refused: ${result.reason} The open snapshot stays.`);
        return;
      }
      current.current?.dispose();
      setSession(result.session);
      setOverridden(contextValue(contextSuffixRef.current, "snapshot"));
      setSelection(null);
      setTrail([]);
      setPinned(new Map());
      // A new snapshot opens at the automatic zoom, so a large page starts on its kind groups.
      setZoom(null);
    });
  };
  const select = (next: Selection): void => {
    setSelection(next);
    if (next?.kind !== "node") return;
    setTrail((previous) => [...previous.filter((id) => id !== next.id), next.id].slice(-8));
    const node = shownSnapshot?.nodes.find((item) => item.id === next.id);
    if (node === undefined || shownDigest === null) return;
    store?.select(selectedObjectOf(shownDigest, node), "atlas");
    // A link restores a public selection only; a private one never enters it.
    setFilter((previous) => ({
      ...previous,
      selected: node.access === "public" ? node.id : null,
    }));
  };
  const requested = contextValue(contextSuffix, "snapshot");
  const info: AtlasSessionInfo | null = session?.info ?? null;
  // The lineage is committed for this build's snapshot; another opened pin has none.
  const overlaid = info?.origin === workspaceOverlayOrigin;
  const lineageOf = (node: AtlasNode): LineageRecord | null =>
    admittedLineage.ok && (info?.origin === committedOrigin || overlaid)
      ? lineageRecordOf(admittedLineage.lineage, node)
      : null;
  // A link names the public pin; the private overlay is rebuilt from it and the workspace.
  const linkDigest = overlaid ? overlayPin.current : info === null ? null : info.digest;
  const mismatch =
    info !== null &&
    linkDigest !== null &&
    requested !== null &&
    requested !== linkDigest &&
    requested !== overridden
      ? `the requested snapshot ${requested} is not the open snapshot (${linkDigest}); open its pin to view it.`
      : null;
  const viewHref = (name: AtlasLens, shownFilter: AtlasFilter): string | null =>
    linkDigest === null ? null : atlasViewHref(name, contextSuffix, linkDigest, shownFilter);
  const draftOf = (node: AtlasNode): ReactNode =>
    node.kind === "revision" && node.access === "private" && workspaceSource !== null ? (
      <RevisionDraft
        node={node}
        workspace={workspaceSource}
        draftHref={draftHref}
        check={checkDraft}
      />
    ) : null;
  const link = viewHref(lens, filter);
  return (
    <article className="qsp-panel qsp-atlas" aria-label="Atlas">
      <h3>Atlas</h3>
      {committedRefusal !== null ? (
        <Unverifiable
          surface="data/studio/atlas_snapshot_20261122.json"
          reason={committedRefusal}
        />
      ) : null}
      <section aria-label="Atlas snapshot">
        {info !== null ? (
          <p>
            {`Snapshot ${info.digest.slice(0, 12)} · ${info.origin} · source tree based on commit ${info.sourceCommit.slice(0, 12)} · ${info.nodes} nodes and ${info.edges} relations · queried ${info.tier === "worker" ? "in a worker" : "on the main thread"}`}
          </p>
        ) : null}
        <label>
          Open a snapshot pin
          <input
            type="file"
            accept=".json,application/json"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file !== undefined) openPin(file);
              event.target.value = "";
            }}
          />
        </label>
        {pending !== null ? (
          <p role="status" aria-label="Atlas progress">
            {pending.label}
            {pending.cancel !== null ? (
              <button
                type="button"
                onClick={() => {
                  pending.cancel?.();
                  setPending(null);
                }}
              >
                Cancel
              </button>
            ) : null}
          </p>
        ) : null}
        {notice !== null ? <p role="alert">{notice}</p> : null}
        {overlayRefusal !== null ? <p role="alert">{overlayRefusal}</p> : null}
        {!admittedLineage.ok ? <p role="alert">{admittedLineage.reason}</p> : null}
      </section>
      {!admittedFilter.ok ? (
        <p role="alert">{`The requested Atlas filter was refused because ${admittedFilter.reason}; the view opens without it.`}</p>
      ) : null}
      {mismatch !== null ? <Unverifiable surface="atlas snapshot" reason={mismatch} /> : null}
      {info !== null && mismatch === null ? (
        <>
          <header>
            <nav aria-label="Atlas lenses">
              {(Object.keys(atlasLenses) as AtlasLens[]).map((name) => (
                <a
                  key={name}
                  href={viewHref(name, filter) ?? (viewHref(name, defaultAtlasFilter) as string)}
                  aria-current={name === lens ? "page" : undefined}
                >
                  {atlasLenses[name].label}
                </a>
              ))}
            </nav>
            <label>
              Search
              <input
                value={filter.text}
                onChange={(event) => setFilter({ ...filter, text: event.target.value })}
              />
            </label>
            <label>
              Hops
              <select
                value={filter.depth}
                onChange={(event) => setFilter({ ...filter, depth: Number(event.target.value) })}
              >
                {Array.from({ length: atlasMaxDepth }, (_unused, index) => index + 1).map(
                  (value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ),
                )}
              </select>
            </label>
            {filter.focus !== null ? (
              <button type="button" onClick={() => setFilter({ ...filter, focus: null })}>
                {`Leave the neighbourhood of ${filter.focus}`}
              </button>
            ) : null}
            {link !== null ? (
              <a href={link}>Link to this view</a>
            ) : (
              <p>This search is too long to link.</p>
            )}
          </header>
          {shownSnapshot !== null && shownDigest !== null ? (
            <SharedSelection
              shared={shared}
              snapshot={shownSnapshot}
              digest={shownDigest}
              lens={lens}
              lensHref={(name) => viewHref(name, { ...filter, focus: null })}
            />
          ) : null}
          {answer === null ? <p role="status">Querying…</p> : null}
          {answer !== null && "refused" in answer ? <p role="alert">{answer.refused}</p> : null}
          {answer !== null && !("refused" in answer) ? (
            <AtlasAnswerView
              answer={answer}
              focus={filter.focus}
              pinned={pinned}
              selection={selection}
              trail={trail}
              zoom={zoom}
              onZoom={setZoom}
              onSelect={select}
              onFocus={(focus) => setFilter({ ...filter, focus })}
              onPin={(id, spot) => setPinned((previous) => new Map(previous).set(id, spot))}
              lineageOf={lineageOf}
              verifyRecord={verifyRecord}
              contextSuffix={contextSuffix}
              draftOf={draftOf}
            />
          ) : null}
        </>
      ) : null}
    </article>
  );
}
