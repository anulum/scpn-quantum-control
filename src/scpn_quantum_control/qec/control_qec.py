# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Control Qec
"""QEC for quantum control signals using a toric surface code and MWPM decoder.

The decoder uses NetworkX minimum-weight perfect matching on a toric lattice
with optional K_nm weighting. It does not implement union-find decoding or a
hardware-level syndrome-extraction controller.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray


class SurfaceCode:
    """Represent a periodic square-lattice toric surface code.

    Parameters
    ----------
    distance : int, default=3
        Linear lattice dimension ``d``. The code allocates ``2*d**2`` data
        qubits and ``d**2`` checks of each stabilizer type.

    Attributes
    ----------
    d : int
        Linear lattice dimension supplied by the caller.
    num_data : int
        Number of edge-local data qubits, ``2*d**2``.
    Hx : numpy.ndarray
        Vertex-check matrix with shape ``(d**2, 2*d**2)`` and ``int8`` dtype.
    Hz : numpy.ndarray
        Plaquette-check matrix with the same shape and dtype as ``Hx``.

    Notes
    -----
    Horizontal and vertical edges use indices ``h(r, c) = 2*(r*d+c)`` and
    ``v(r, c) = 2*(r*d+c)+1``. This low-level prototype does not validate the
    distance; callers are responsible for supplying a positive dimension that
    represents the intended toric code.

    """

    def __init__(self, distance: int = 3):
        """Build the toric-code parity-check matrices.

        Parameters
        ----------
        distance : int, default=3
            Linear dimension used for both periodic lattice axes.

        """
        self.d = distance
        self.num_data = 2 * distance**2
        self.Hx, self.Hz = self._build_checks()

    def _build_checks(self) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        """Construct the vertex and plaquette parity-check matrices.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(Hx, Hz)`` binary ``int8`` matrices, each with shape
            ``(d**2, 2*d**2)``.

        """
        d = self.d
        N = self.num_data
        Hx: NDArray[np.int8] = np.zeros((d * d, N), dtype=np.int8)
        Hz: NDArray[np.int8] = np.zeros((d * d, N), dtype=np.int8)

        def h(r: int, c: int) -> int:
            return 2 * (r * d + c)

        def v(r: int, c: int) -> int:
            return 2 * (r * d + c) + 1

        for r in range(d):
            for c in range(d):
                stab = r * d + c
                # Vertex (X) stabilizer: 4 edges around vertex
                Hx[stab, h(r, c)] = 1
                Hx[stab, h(r, (c - 1) % d)] = 1
                Hx[stab, v(r, c)] = 1
                Hx[stab, v((r - 1) % d, c)] = 1

                # Plaquette (Z) stabilizer: 4 edges around face
                Hz[stab, h(r, c)] = 1
                Hz[stab, h((r + 1) % d, c)] = 1
                Hz[stab, v(r, c)] = 1
                Hz[stab, v(r, (c + 1) % d)] = 1

        return Hx, Hz


class MWPMDecoder:
    """Decode toric syndromes by minimum-weight perfect matching.

    Parameters
    ----------
    distance : int
        Linear dimension ``d`` of the periodic stabilizer lattice.
    knm_weights : numpy.ndarray or None, optional
        Optional stabilizer-indexed coupling matrix used to rescale pair costs.

    Notes
    -----
    NetworkX solves a maximum-cardinality matching with negative edge weights,
    which is equivalent here to minimizing the integer toric pair costs. The
    optional K_nm matrix changes defect pairing only; corrections still follow
    deterministic Manhattan paths on the primal or dual lattice.

    """

    def __init__(self, distance: int, knm_weights: NDArray[np.float64] | None = None):
        """Configure matching for a toric stabilizer lattice.

        Parameters
        ----------
        distance : int
            Linear stabilizer-lattice dimension.
        knm_weights : numpy.ndarray or None, optional
            Matrix indexed by stabilizer identifiers. For in-range pairs, the
            decoder floors ``base_distance / (1 + K[u, v])`` to an integer and
            clamps the result to at least one. Matrix shape and values are not
            validated.

        """
        self.d = distance
        self.knm_weights = knm_weights

    def decode(self, syndrome: NDArray[np.int8], dual: bool = False) -> NDArray[np.int8]:
        """Construct a binary correction for a toric syndrome.

        Parameters
        ----------
        syndrome : numpy.ndarray
            One-dimensional stabilizer vector. Entries equal to one are treated
            as defects; other values are ignored by the defect selector.
        dual : bool, default=False
            Use the plaquette/dual path convention when true. False decodes
            vertex syndromes for X-error correction; true decodes plaquette
            syndromes for Z-error correction.

        Returns
        -------
        numpy.ndarray
            Binary ``int8`` correction vector with length ``2*d**2``.

        Notes
        -----
        A valid periodic-code syndrome has even defect parity. For compatibility,
        an odd input duplicates its first defect before matching. ``ControlQEC``
        subsequently rejects any correction that leaves a residual syndrome.

        """
        defects = np.where(syndrome == 1)[0]
        if len(defects) == 0:
            return np.zeros(2 * self.d**2, dtype=np.int8)
        if len(defects) % 2 == 1:
            defects = np.append(defects, defects[0])

        G = nx.Graph()
        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                dist = self._distance(defects[i], defects[j])
                G.add_edge(i, j, weight=-dist)

        matching = nx.max_weight_matching(G, maxcardinality=True)

        correction: NDArray[np.int8] = np.zeros(2 * self.d**2, dtype=np.int8)
        for i, j in matching:
            path = self._shortest_path(defects[i], defects[j], dual=dual)
            for qubit in path:
                correction[qubit] ^= 1

        return correction

    def _distance(self, u: int, v: int) -> int:
        """Return the integer matching cost between two stabilizers.

        Parameters
        ----------
        u, v : int
            Flattened stabilizer indices on the ``d`` by ``d`` torus.

        Returns
        -------
        int
            Wrapped Manhattan distance, optionally rescaled by ``K[u, v]``,
            floored through integer conversion, and clamped to at least one for
            weighted in-range pairs.

        """
        d = self.d
        r1, c1 = divmod(u, d)
        r2, c2 = divmod(v, d)
        dr = min(abs(r1 - r2), d - abs(r1 - r2))
        dc = min(abs(c1 - c2), d - abs(c1 - c2))
        base_dist = dr + dc

        if (
            self.knm_weights is not None
            and u < len(self.knm_weights)
            and v < len(self.knm_weights)
        ):
            knm_factor = 1.0 / (1.0 + self.knm_weights[u, v])
            return max(1, int(base_dist * knm_factor))
        return base_dist

    def _shortest_path(self, u: int, v: int, dual: bool = False) -> list[int]:
        """Return qubit indices along a toric Manhattan shortest path.

        Parameters
        ----------
        u, v : int
            Flattened start and end stabilizer indices.
        dual : bool, default=False
            Select plaquette-path edge conventions instead of vertex-path
            conventions.

        Returns
        -------
        list[int]
            Ordered data-qubit indices toggled by the correction.

        Notes
        -----
        Vertex paths use vertical edges for row moves and horizontal edges for
        column moves. Plaquette paths reverse those roles. Plaquettes ``(r, c)``
        and ``((r+1) % d, c)`` share horizontal edge ``h((r+1) % d, c)``.
        Equal-length wrapped directions deterministically choose the forward path.

        """
        d = self.d
        r1, c1 = divmod(u, d)
        r2, c2 = divmod(v, d)
        path: list[int] = []

        dr_fwd = (r2 - r1) % d
        dr_bwd = (r1 - r2) % d
        r = r1
        if dr_fwd <= dr_bwd:
            for _ in range(dr_fwd):
                if dual:
                    r = (r + 1) % d
                    path.append(2 * (r * d + c1))  # horizontal edge h(r, c1)
                else:
                    path.append(2 * (r * d + c1) + 1)  # vertical edge v(r, c1)
                    r = (r + 1) % d
        else:
            for _ in range(dr_bwd):
                if dual:
                    path.append(2 * (r * d + c1))  # horizontal edge h(r, c1)
                    r = (r - 1) % d
                else:
                    r = (r - 1) % d
                    path.append(2 * (r * d + c1) + 1)  # vertical edge v(r, c1)

        dc_fwd = (c2 - c1) % d
        dc_bwd = (c1 - c2) % d
        c = c1
        if dc_fwd <= dc_bwd:
            for _ in range(dc_fwd):
                if dual:
                    c = (c + 1) % d
                    path.append(2 * (r2 * d + c) + 1)  # vertical edge v(r2, c)
                else:
                    path.append(2 * (r2 * d + c))  # horizontal edge h(r2, c)
                    c = (c + 1) % d
        else:
            for _ in range(dc_bwd):
                if dual:
                    path.append(2 * (r2 * d + c) + 1)  # vertical edge v(r2, c)
                    c = (c - 1) % d
                else:
                    c = (c - 1) % d
                    path.append(2 * (r2 * d + c))  # horizontal edge h(r2, c)

        return path


class ControlQEC:
    """Run stochastic toric-code error-correction rounds.

    Parameters
    ----------
    distance : int, default=3
        Linear dimension of the periodic surface-code lattice.
    knm_weights : numpy.ndarray or None, optional
        Stabilizer-indexed coupling weights forwarded to ``MWPMDecoder``.

    Attributes
    ----------
    code : SurfaceCode
        Constructed toric code and its parity-check matrices.
    decoder : MWPMDecoder
        Matching decoder configured for the same distance and optional weights.

    Notes
    -----
    A round succeeds only when both corrected syndromes vanish and neither
    residual error contains a non-contractible toric cycle. This analysis
    surface does not perform hardware syndrome extraction.

    """

    def __init__(self, distance: int = 3, knm_weights: NDArray[np.float64] | None = None):
        """Assemble a toric code and its matching decoder.

        Parameters
        ----------
        distance : int, default=3
            Linear dimension forwarded unchanged to both components.
        knm_weights : numpy.ndarray or None, optional
            Optional stabilizer-pair coupling matrix forwarded to the decoder.

        """
        self.code = SurfaceCode(distance)
        self.decoder = MWPMDecoder(distance, knm_weights)

    def simulate_errors(
        self, p_error: float, rng: np.random.Generator | None = None
    ) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        """Sample independent X- and Z-error vectors.

        Parameters
        ----------
        p_error : float
            Bernoulli probability passed to NumPy for every X and Z component.
        rng : numpy.random.Generator or None, optional
            Random generator to consume. A fresh default generator is created
            when omitted.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Independent ``(err_x, err_z)`` binary ``int8`` vectors, each with
            length ``2*d**2``.

        Raises
        ------
        ValueError
            If NumPy rejects ``p_error``, including probabilities outside
            ``[0, 1]``.

        """
        if rng is None:
            rng = np.random.default_rng()
        N = self.code.num_data
        err_x = rng.binomial(1, p_error, N).astype(np.int8)
        err_z = rng.binomial(1, p_error, N).astype(np.int8)
        return err_x, err_z

    def get_syndrome(
        self, err_x: NDArray[np.int8], err_z: NDArray[np.int8]
    ) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        """Compute vertex and plaquette syndromes modulo two.

        Parameters
        ----------
        err_x, err_z : numpy.ndarray
            X- and Z-error vectors with expected length ``2*d**2``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(syn_z, syn_x)`` where ``syn_z = Hx @ err_x mod 2`` and
            ``syn_x = Hz @ err_z mod 2``. Each vector has length ``d**2``.

        Raises
        ------
        ValueError
            If an error vector is dimensionally incompatible with its check
            matrix.

        """
        syn_z = (self.code.Hx @ err_x) % 2
        syn_x = (self.code.Hz @ err_z) % 2
        return syn_z, syn_x

    def decode_and_correct(self, err_x: NDArray[np.int8], err_z: NDArray[np.int8]) -> bool:
        """Decode one X/Z error pair and test whether correction succeeds.

        Parameters
        ----------
        err_x, err_z : numpy.ndarray
            Binary ``int8`` error vectors with expected length ``2*d**2``.

        Returns
        -------
        bool
            True only when both residual syndromes vanish and no residual has
            odd winding parity across either toric seam.

        Raises
        ------
        ValueError
            If an error vector is dimensionally incompatible with the code.

        Notes
        -----
        A residual with zero syndrome may still represent a logical operator
        when it wraps around a non-contractible toric cycle.

        """
        syn_z, syn_x = self.get_syndrome(err_x, err_z)

        corr_x = self.decoder.decode(syn_z, dual=False)
        corr_z = self.decoder.decode(syn_x, dual=True)

        residual_x = err_x ^ corr_x
        residual_z = err_z ^ corr_z

        new_syn_z = (self.code.Hx @ residual_x) % 2
        new_syn_x = (self.code.Hz @ residual_z) % 2

        if not (np.all(new_syn_z == 0) and np.all(new_syn_x == 0)):
            return False

        return not self._has_logical_error(residual_x, residual_z)

    def _has_logical_error(
        self, residual_x: NDArray[np.int8], residual_z: NDArray[np.int8]
    ) -> bool:
        """Detect non-trivial winding in residual X and Z errors.

        Parameters
        ----------
        residual_x, residual_z : numpy.ndarray
            Corrected binary error vectors with length ``2*d**2``.

        Returns
        -------
        bool
            True when either residual has odd horizontal or vertical seam
            parity; otherwise false.

        Notes
        -----
        Horizontal winding is the parity of ``h(r, 0) = 2*r*d`` over rows.
        Vertical winding is the parity of ``v(0, c) = 2*c + 1`` over columns.

        """
        d = self.code.d
        for residual in (residual_x, residual_z):
            # Horizontal winding: h-edges crossing column 0→1 boundary
            if sum(int(residual[2 * r * d]) for r in range(d)) % 2 == 1:
                return True
            # Vertical winding: v-edges crossing row 0→1 boundary
            if sum(int(residual[2 * c + 1]) for c in range(d)) % 2 == 1:
                return True
        return False
