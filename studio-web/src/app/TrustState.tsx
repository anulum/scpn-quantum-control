// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — trust-state chips distinguishable without colour

/** The five trust states a result or evidence record can carry in the workbench. */
export type TrustStateKind = "verified" | "bounded" | "fail-closed" | "unverifiable" | "simulated";

/** Visible label and non-colour mark of one trust state. */
export interface TrustStateDefinition {
  /** Text shown in the chip; it alone identifies the state. */
  readonly label: string;
  /** Name of the shape drawn beside the label. */
  readonly shape: "circle-check" | "half-diamond" | "square-bar" | "octagon-cross" | "dashed-ring";
}

/** Label and shape per state; colour only repeats what these already say. */
export const trustStates: Readonly<Record<TrustStateKind, TrustStateDefinition>> = Object.freeze({
  verified: { label: "Verified here", shape: "circle-check" },
  bounded: { label: "Bounded model", shape: "half-diamond" },
  "fail-closed": { label: "Fail-closed", shape: "square-bar" },
  unverifiable: { label: "Unverifiable", shape: "octagon-cross" },
  simulated: { label: "Simulated · operator only", shape: "dashed-ring" },
});

/** Shape strokes of each mark, drawn in the chip's text colour. */
function markStrokes(shape: TrustStateDefinition["shape"]) {
  switch (shape) {
    case "circle-check":
      return (
        <>
          <circle cx="6" cy="6" r="5" fill="none" stroke="currentColor" strokeWidth="1.4" />
          <path d="M3.5 6.2l1.7 1.7 3.3-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
        </>
      );
    case "half-diamond":
      return (
        <>
          <path d="M6 1l5 5-5 5-5-5z" fill="none" stroke="currentColor" strokeWidth="1.4" />
          <path d="M6 1l5 5-5 5z" fill="currentColor" />
        </>
      );
    case "square-bar":
      return (
        <>
          <rect
            x="1.5"
            y="1.5"
            width="9"
            height="9"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
          />
          <path d="M3.5 6h5" stroke="currentColor" strokeWidth="1.6" />
        </>
      );
    case "octagon-cross":
      return (
        <>
          <path
            d="M4 1h4l3 3v4l-3 3H4L1 8V4z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.4"
          />
          <path d="M4.2 4.2l3.6 3.6M7.8 4.2L4.2 7.8" stroke="currentColor" strokeWidth="1.4" />
        </>
      );
    case "dashed-ring":
      return (
        <circle
          cx="6"
          cy="6"
          r="4.6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeDasharray="2 1.6"
        />
      );
  }
}

/** Draw the non-colour mark of a state; the label beside it carries its meaning. */
function TrustMark({ shape }: { readonly shape: TrustStateDefinition["shape"] }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      aria-hidden="true"
      focusable="false"
      data-shape={shape}
    >
      {markStrokes(shape)}
    </svg>
  );
}

/** A chip that names a trust state in words and marks it with a distinct shape. */
export function TrustState({
  kind,
  detail,
}: {
  /** State shown by the chip. */
  readonly kind: TrustStateKind;
  /** Optional qualifier appended to the label, such as "bit-exact". */
  readonly detail?: string;
}) {
  const state = trustStates[kind];
  return (
    <span className="qsp-trust" data-trust={kind}>
      <TrustMark shape={state.shape} />
      <span>
        {state.label}
        {detail === undefined ? null : ` · ${detail}`}
      </span>
    </span>
  );
}
