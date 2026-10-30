// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — failure and loading isolation for one module contribution

import { Component, Suspense } from "react";
import type { ReactNode } from "react";

/** One module contribution rendered inside a shell slot. */
export interface ContributionBoundaryProps {
  /** Heading of the contribution, used in its loading and failure states. */
  readonly label: string;
  /** Identifier of the module that registered the contribution. */
  readonly owner: string;
  /** The contribution's own rendering, called inside the boundary; it may return a lazy component. */
  readonly render: () => ReactNode;
}

/** Calls the contribution while React renders this child, so a throw stays inside the boundary. */
function Contribution({ render }: { readonly render: () => ReactNode }) {
  return render();
}

interface BoundaryState {
  readonly failed: boolean;
}

/** Contains one contribution's failure; a new key remounts it to try again. */
class Boundary extends Component<ContributionBoundaryProps, BoundaryState> {
  /** The contribution renders until it throws. */
  override state: BoundaryState = { failed: false };

  /** Contain the failure without showing the exception payload. */
  static getDerivedStateFromError(): BoundaryState {
    return { failed: true };
  }

  /** The failure notice, or the contribution with its own loading state. */
  override render(): ReactNode {
    const { label, owner, render } = this.props;
    if (this.state.failed)
      return (
        <section role="alert" aria-label={`${label} failed`}>
          <h3>{label} failed</h3>
          <p>
            The {owner} module could not show this part. The rest of the view and your workspace are
            retained.
          </p>
        </section>
      );
    return (
      <Suspense fallback={<p role="status">Loading {label}…</p>}>
        <Contribution render={render} />
      </Suspense>
    );
  }
}

/**
 * Isolate one module contribution from the rest of its slot.
 *
 * A contribution that throws while rendering, or whose lazy code fails to load,
 * is replaced by a notice naming it and its module; the other contributions and
 * the shell keep working. A contribution that is still loading shows its own
 * loading state instead of suspending the whole slot.
 *
 * @param props - Contribution label, owning module and its rendering.
 * @returns The contribution, its loading state or its failure notice.
 */
export function ContributionBoundary(props: ContributionBoundaryProps) {
  return <Boundary {...props} />;
}
