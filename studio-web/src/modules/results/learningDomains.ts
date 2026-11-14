// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — domain-proxy and free-energy data projection

import { finite, list, record, text, texts } from "../../shared/browser/jsonFields";

/** One named source metric. */
export interface NamedMetric {
  /** Source metric name. */ readonly name: string;
  /** Source value. */ readonly value: number;
}
/** A curated domain benchmark: proxy metrics under its honesty kit, never a physical outcome. */
export interface DomainProxy {
  /** Packaged dataset identity. */ readonly datasetId: string;
  /** Source domain label. */ readonly domain: string;
  /** Dataset licence note. */ readonly licenceNote: string;
  /** Public literature the matrix is curated from. */ readonly reference: string;
  /** Proxy metrics in name order. */ readonly metrics: readonly NamedMetric[];
  /** Governing honesty kit. */ readonly kitId: string;
  /** Kit support grade. */ readonly supportStatus: string;
  /** Uses the kit admits. */ readonly allowedUses: readonly string[];
  /** Limits the kit states. */ readonly caveats: readonly string[];
  /** Claims the kit never authorises. */ readonly forbidden: readonly string[];
}
/** The curated free-energy workflow and the owner's covariance refusals. */
export interface FreeEnergyTemplate {
  /** Packaged dataset identity. */ readonly datasetId: string;
  /** Dataset licence note. */ readonly licenceNote: string;
  /** Public literature the workflow follows. */ readonly reference: string;
  /** Free-energy metrics in name order. */ readonly metrics: readonly NamedMetric[];
  /** Application claim boundary. */ readonly boundary: string;
  /** The owner's refusals, each with its case. */ readonly refusals: readonly {
    /** Invalid input described. */ readonly case: string;
    /** The owner's own message. */ readonly message: string;
  }[];
}

function metrics(raw: unknown, field: string): NamedMetric[] {
  const entries = Object.entries(record(raw, field)).map(([name, value]) => ({
    name,
    value: finite(value, `${field}.${name}`),
  }));
  if (entries.length === 0) throw new Error(`${field} must not be empty`);
  return entries.sort((left, right) => left.name.localeCompare(right.name));
}

function parseDomainProxy(raw: unknown): DomainProxy {
  const row = record(raw, "domain proxy");
  if (row["physical_outcome_in_custody"] !== false)
    throw new Error("A domain proxy cannot carry a measured physical outcome");
  const kit = record(row["kit"], "kit");
  if (kit["publication_safe"] !== false)
    throw new Error("A domain proxy cannot be marked publication-safe");
  const forbidden = texts(kit["claims_forbidden"], "claims_forbidden");
  if (forbidden.length === 0) throw new Error("A domain proxy needs its forbidden claims");
  return {
    datasetId: text(row["dataset_id"], "dataset_id"),
    domain: text(row["domain"], "domain"),
    licenceNote: text(row["licence_note"], "licence_note"),
    reference: text(row["public_reference"], "public_reference"),
    metrics: metrics(row["proxy_metrics"], "proxy_metrics"),
    kitId: text(kit["kit_id"], "kit_id"),
    supportStatus: text(kit["support_status"], "support_status"),
    allowedUses: texts(kit["allowed_uses"], "allowed_uses"),
    caveats: texts(kit["caveats"], "caveats"),
    forbidden,
  };
}

/**
 * Parse the domain-proxy rows, refusing a promoted claim or a duplicate dataset.
 * @param raw - Source list of domain-proxy rows.
 * @returns The rows in source order.
 */
export function parseDomainProxies(raw: unknown): DomainProxy[] {
  const rows = list(raw, "domain_proxies").map(parseDomainProxy);
  const ids = rows.map((row) => row.datasetId);
  if (ids.length === 0 || new Set(ids).size !== ids.length)
    throw new Error("Domain proxies need non-empty unique datasets");
  return rows;
}

/**
 * Parse the free-energy workflow, requiring the owner's covariance refusals.
 * @param raw - Source free-energy template.
 * @returns The workflow metrics, boundary and refusals.
 */
export function parseFreeEnergy(raw: unknown): FreeEnergyTemplate {
  const row = record(raw, "free_energy");
  const refusals = list(row["refusals"], "refusals").map((item) => {
    const refusal = record(item, "refusal");
    return { case: text(refusal["case"], "case"), message: text(refusal["message"], "message") };
  });
  if (refusals.length === 0)
    throw new Error("The free-energy owner's covariance refusals are required");
  return {
    datasetId: text(row["dataset_id"], "dataset_id"),
    licenceNote: text(row["licence_note"], "licence_note"),
    reference: text(row["public_reference"], "public_reference"),
    metrics: metrics(row["metrics"], "metrics"),
    boundary: text(row["claim_boundary"], "claim_boundary"),
    refusals,
  };
}

/**
 * Export one domain proxy as public metadata that keeps its proxy status.
 * @param row - A parsed domain proxy.
 * @returns JSON naming the evidence class, the kit and every forbidden claim.
 */
export function exportDomainProxy(row: DomainProxy): string {
  return JSON.stringify(
    {
      dataset_id: row.datasetId,
      domain: row.domain,
      evidence_class: "domain proxy metric",
      physical_outcome_in_custody: false,
      publication_safe: false,
      proxy_metrics: Object.fromEntries(row.metrics.map((metric) => [metric.name, metric.value])),
      kit_id: row.kitId,
      claims_forbidden: row.forbidden,
      licence_note: row.licenceNote,
      public_reference: row.reference,
    },
    null,
    2,
  );
}
