// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — source-bound human review eligibility and record export

import { canonicalDigest } from "../../../shared/contracts/canonical";
import { writeJson } from "../../../shared/contracts/jsonTransport";
import type { HumanReview, OperatorDossierSnapshot, ReviewStatus } from "./operatorDossierModel";
import { operatorTimestamp as timestamp } from "./operatorDossierTime";

/** Resolve current review eligibility against exact source identity and UTC time. */
export function reviewStatus(
  snapshot: OperatorDossierSnapshot,
  review: HumanReview | null,
  now = new Date(),
): ReviewStatus {
  if (review !== null && review.executionSha256 !== snapshot.executionSha256) return "invalidated";
  const instant = now.getTime();
  if (!Number.isFinite(instant)) return "refused";
  const created = Date.parse(String(snapshot.body["created_at"]));
  const expires = Date.parse(String(snapshot.body["expires_at"]));
  let recorded: number | null = null;
  if (review !== null) {
    try {
      recorded = Date.parse(timestamp(review.recordedAt));
    } catch {
      return "refused";
    }
    if (recorded < created) return "refused";
  }
  if (instant < created || (recorded !== null && recorded > instant)) return "not_yet_valid";
  if (review?.choice === "denied") return "denied";
  if (instant >= expires) return "expired";
  if (!snapshot.policy.decision.allowed) return "refused";
  return review === null ? "pending" : "approved";
}

/** Seal a separate human record; the original source is never mutated. */
export async function reviewDocument(review: HumanReview): Promise<string> {
  const wire = {
    schema: "studio.operator-review.v1",
    body: {
      no_submit: true,
      claim_boundary: "human_review_only",
      dossier_sha256: review.dossierSha256,
      execution_sha256: review.executionSha256,
      choice: review.choice,
      reviewer_ref: "studio/local-human",
      recorded_at: timestamp(review.recordedAt),
    },
    extensions: {},
  };
  return `${writeJson({ ...wire, sha256: await canonicalDigest(wire.schema, wire) })}\n`;
}
