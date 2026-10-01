// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — feature route error isolation

import { Component } from "react";
import type { ReactNode } from "react";

/** A route owns its failures; the persistent workspace editor lives outside this boundary. */
export interface RouteBoundaryProps {
  /** Feature content, including a genuine lazy module request. */
  readonly children: ReactNode;
  /** Admitted recovery URL carrying the same requested context. */
  readonly workspaceHref: string;
}

interface BoundaryState { readonly failed: boolean; }

/** Remount with the route identity as its key to recover on navigation without clearing a draft. */
class FeatureBoundary extends Component<RouteBoundaryProps, BoundaryState> {
  /** A failed feature stays contained until the route key remounts this boundary. */
  override state: BoundaryState = { failed: false };

  /** Contain a render/import failure without exposing exception payloads or mutating storage. */
  static getDerivedStateFromError(): BoundaryState { return { failed: true }; }

  /** Keep route recovery reachable while the original editor remains mounted. */
  override render(): ReactNode {
    if (this.state.failed) return (
      <div role="alert">
        <h3>View unavailable</h3>
        <p>This view could not load. Your workspace and editor are retained.</p>
        <a href={this.props.workspaceHref}>Return to Workspace</a>
      </div>
    );
    return this.props.children;
  }
}

/** Public feature boundary; only the route's props are part of the consumer contract. */
export function RouteBoundary(props: RouteBoundaryProps) {
  return <FeatureBoundary {...props} />;
}
