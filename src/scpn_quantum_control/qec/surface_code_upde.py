# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Surface Code Upde
"""Build a structural surface-code UPDE circuit and resource scaffold.

Each oscillator occupies a distance-d rotated-surface-code-shaped register
(Horsman et al., NJP 14, 123011 (2012)). The generated circuit approximates
encoding fan-out, distributed physical rotations, inter-patch couplings, and
X/Z ancilla interactions for resource and depth analysis. It does not prepare a
verified codespace, measure or reset ancillas, allocate classical syndrome
bits, invoke a decoder, or demonstrate correction of either X or Z errors.

Physical qubit layout per oscillator: d² data + (d²-1) ancilla = 2d²-1.
Total physical qubits: n_osc × (2d² - 1).

Structural operator proxies:
  - distribute an Rz angle over every data qubit in one patch;
  - distribute pairwise RZZ angles across corresponding qubits in two patches;
  - append X- and Z-type ancilla-entangling networks without measurements.

These proxies are not validated fault-tolerant logical gates or lattice-surgery
protocols. Decoding remains a separate responsibility; ``ControlQEC`` provides
an independent toric-code MWPM analysis surface rather than consuming these
unmeasured ancillas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from ..bridge.knm_hamiltonian import build_knm_paper27, omega_for_oscillators


@dataclass(frozen=True)
class SurfaceCodeSpec:
    """Record per-patch rotated-surface-code resource counts.

    Parameters
    ----------
    distance : int
        Odd code distance ``d``.
    n_data : int
        Number of data qubits, ``d**2``.
    n_ancilla : int
        Number of ancillas, ``d**2 - 1``.
    n_physical : int
        Total patch register size, ``2*d**2 - 1``.

    Notes
    -----
    Use :meth:`from_distance` to enforce the odd-distance invariant and derive
    consistent counts. Direct dataclass construction does not revalidate fields.

    """

    distance: int
    n_data: int  # d²
    n_ancilla: int  # d² - 1 (X-type: (d²-1)/2, Z-type: (d²-1)/2)
    n_physical: int  # 2d² - 1

    @classmethod
    def from_distance(cls, d: int) -> SurfaceCodeSpec:
        """Construct consistent patch counts from an odd code distance.

        Parameters
        ----------
        d : int
            Requested odd distance, at least three.

        Returns
        -------
        SurfaceCodeSpec
            Immutable resource counts for one patch.

        Raises
        ------
        ValueError
            If ``d`` is below three or even.

        """
        if d < 3 or d % 2 == 0:
            raise ValueError(f"Distance must be odd >= 3, got {d}")
        n_data = d * d
        n_ancilla = d * d - 1
        return cls(distance=d, n_data=n_data, n_ancilla=n_ancilla, n_physical=2 * d * d - 1)


class SurfaceCodeUPDE:
    """Model a surface-code-shaped Kuramoto-XY circuit scaffold.

    Parameters
    ----------
    n_osc : int
        Number of oscillator patches; must be at least two.
    code_distance : int, default=3
        Odd patch distance, at least three.
    K : numpy.ndarray or None, optional
        Coupling matrix with expected shape ``(n_osc, n_osc)``. The Paper 27
        deterministic matrix is built when omitted.
    omega : numpy.ndarray or None, optional
        Natural-frequency vector with expected length ``n_osc``. The canonical
        16-entry table, periodically extended for larger systems, is used when
        omitted.

    Attributes
    ----------
    spec : SurfaceCodeSpec
        Per-patch resource counts.
    total_qubits : int
        Full circuit register size, ``n_osc * (2*d**2 - 1)``.

    Notes
    -----
    Distributed RZ/RZZ operations and ancilla interactions are structural
    operator proxies. The circuit contains no measurements or classical bits
    and makes no fault-tolerance claim. Custom ``K`` and ``omega`` shapes are
    consumed by :meth:`build_step_circuit` without constructor validation.

    Representative budgets are 68 qubits for ``n_osc=4, d=3``, 196 for
    ``n_osc=4, d=5``, 272 for ``n_osc=16, d=3``, and 784 for
    ``n_osc=16, d=5``.

    """

    def __init__(
        self,
        n_osc: int,
        code_distance: int = 3,
        K: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
    ):
        """Configure the structural circuit and its resource register.

        Parameters
        ----------
        n_osc : int
            Number of oscillator patches.
        code_distance : int, default=3
            Odd patch distance, at least three.
        K : numpy.ndarray or None, optional
            Coupling matrix used by the pair-interaction loop.
        omega : numpy.ndarray or None, optional
            Frequency vector used by patch-local distributed rotations.

        Raises
        ------
        ValueError
            If fewer than two oscillators are requested or the distance is
            below three or even.

        """
        if n_osc < 2:
            raise ValueError(f"Need >= 2 oscillators, got {n_osc}")

        self.n_osc = n_osc
        self.spec = SurfaceCodeSpec.from_distance(code_distance)
        self.K = K if K is not None else build_knm_paper27(L=n_osc)
        self.omega = omega if omega is not None else omega_for_oscillators(n_osc)
        self.total_qubits = n_osc * self.spec.n_physical

    def _osc_data_qubits(self, osc: int) -> list[int]:
        """Return the data-qubit block for an oscillator patch.

        Parameters
        ----------
        osc : int
            Zero-based oscillator index.

        Returns
        -------
        list[int]
            Contiguous data-qubit indices in the global register.

        """
        base = osc * self.spec.n_physical
        return list(range(base, base + self.spec.n_data))

    def _osc_x_ancilla(self, osc: int) -> list[int]:
        """Return the first, X-labelled half of a patch's ancillas.

        Parameters
        ----------
        osc : int
            Zero-based oscillator index.

        Returns
        -------
        list[int]
            Contiguous global-register indices for X-labelled ancillas.

        """
        base = osc * self.spec.n_physical + self.spec.n_data
        n_x = (self.spec.n_ancilla + 1) // 2
        return list(range(base, base + n_x))

    def _osc_z_ancilla(self, osc: int) -> list[int]:
        """Return the second, Z-labelled half of a patch's ancillas.

        Parameters
        ----------
        osc : int
            Zero-based oscillator index.

        Returns
        -------
        list[int]
            Contiguous global-register indices for Z-labelled ancillas.

        """
        base = osc * self.spec.n_physical + self.spec.n_data
        n_x = (self.spec.n_ancilla + 1) // 2
        n_z = self.spec.n_ancilla - n_x
        return list(range(base + n_x, base + n_x + n_z))

    def encode_logical(self, osc: int, qc: QuantumCircuit) -> None:
        """Append the patch's proxy encoding fan-out network.

        Parameters
        ----------
        osc : int
            Oscillator patch whose data block receives the operations.
        qc : qiskit.QuantumCircuit
            Circuit owning the full patch register.

        Notes
        -----
        The method applies ``Ry(omega[osc] mod 2*pi)`` to one representative
        qubit followed by row and column CNOT fans. This is a structural circuit
        proxy, not verified preparation of a logical ``|+_L>`` codespace state.

        """
        data = self._osc_data_qubits(osc)
        theta = float(self.omega[osc]) % (2 * np.pi)

        # Prepare representative qubit
        qc.ry(theta, data[0])

        # X-basis fan-out (row-wise CNOT chain within surface code patch)
        d = self.spec.distance
        for row in range(d):
            row_start = row * d
            for col in range(1, d):
                qc.cx(data[row_start], data[row_start + col])

        # Z-basis fan-out (column-wise CNOT chain)
        for col in range(d):
            for row in range(1, d):
                qc.cx(data[col], data[row * d + col])

    def logical_rz(self, osc: int, angle: float, qc: QuantumCircuit) -> None:
        """Append the distributed physical-RZ operator proxy.

        Parameters
        ----------
        osc : int
            Oscillator patch whose data qubits receive the rotations.
        angle : float
            Aggregate angle divided evenly over the ``d**2`` data qubits.
        qc : qiskit.QuantumCircuit
            Circuit to mutate.

        Notes
        -----
        Applying ``Rz(angle / d**2)`` to every data qubit is a resource-model
        proxy; this module does not establish it as a fault-tolerant logical RZ.

        """
        data = self._osc_data_qubits(osc)
        distributed_angle = angle / len(data)
        for dq in data:
            qc.rz(distributed_angle, dq)

    def logical_zz(
        self,
        osc_i: int,
        osc_j: int,
        angle: float,
        qc: QuantumCircuit,
    ) -> None:
        """Append the distributed inter-patch RZZ operator proxy.

        Parameters
        ----------
        osc_i, osc_j : int
            Distinct oscillator patches coupled pairwise.
        angle : float
            Aggregate angle divided over corresponding data-qubit pairs.
        qc : qiskit.QuantumCircuit
            Circuit to mutate.

        Notes
        -----
        The pairwise physical RZZ layer is not an ancilla-mediated
        merge-and-split lattice-surgery protocol and is not claimed to implement
        a fault-tolerant logical ZZ gate.

        """
        data_i = self._osc_data_qubits(osc_i)
        data_j = self._osc_data_qubits(osc_j)
        distributed_angle = angle / len(data_i)
        for di, dj in zip(data_i, data_j):
            qc.rzz(distributed_angle, di, dj)

    def x_syndrome_extract(self, osc: int, qc: QuantumCircuit) -> None:
        """Append the X-labelled ancilla interaction scaffold.

        Parameters
        ----------
        osc : int
            Oscillator patch to address.
        qc : qiskit.QuantumCircuit
            Circuit to mutate.

        Notes
        -----
        Each ancilla receives Hadamard, four ancilla-to-data CNOTs, and a final
        Hadamard. No measurement, reset, classical bit, or decoded syndrome is
        produced.

        """
        data = self._osc_data_qubits(osc)
        x_anc = self._osc_x_ancilla(osc)
        d = self.spec.distance

        for idx, anc in enumerate(x_anc):
            qc.h(anc)
            row, col = divmod(idx, d - 1)
            neighbors = (
                data[row * d + col],
                data[row * d + col + 1],
                data[(row + 1) * d + col],
                data[(row + 1) * d + col + 1],
            )
            for nb in neighbors:
                qc.cx(anc, nb)
            qc.h(anc)

    def z_syndrome_extract(self, osc: int, qc: QuantumCircuit) -> None:
        """Append the Z-labelled ancilla interaction scaffold.

        Parameters
        ----------
        osc : int
            Oscillator patch to address.
        qc : qiskit.QuantumCircuit
            Circuit to mutate.

        Notes
        -----
        Each ancilla receives four data-to-ancilla CNOTs. No measurement, reset,
        classical bit, or decoded syndrome is produced.

        """
        data = self._osc_data_qubits(osc)
        z_anc = self._osc_z_ancilla(osc)
        d = self.spec.distance

        for idx, anc in enumerate(z_anc):
            row, col = divmod(idx, d - 1)
            neighbors = (
                data[row * d + col],
                data[row * d + col + 1],
                data[(row + 1) * d + col],
                data[(row + 1) * d + col + 1],
            )
            for nb in neighbors:
                qc.cx(nb, anc)

    def build_step_circuit(self, dt: float = 0.1) -> QuantumCircuit:
        """Build one structural Kuramoto-XY circuit step.

        Parameters
        ----------
        dt : float, default=0.1
            Step multiplier applied to frequencies and nonzero couplings.

        Returns
        -------
        qiskit.QuantumCircuit
            Unmeasured circuit containing proxy encoding, distributed RZ/RZZ,
            and X/Z ancilla-interaction layers on ``total_qubits`` qubits.

        Notes
        -----
        Couplings with magnitude at most ``1e-10`` are omitted. The returned
        circuit has no classical register and performs no correction cycle.

        """
        qc = QuantumCircuit(self.total_qubits)

        for osc in range(self.n_osc):
            self.encode_logical(osc, qc)

        for osc in range(self.n_osc):
            self.logical_rz(osc, self.omega[osc] * dt, qc)

        for i in range(self.n_osc):
            for j in range(i + 1, self.n_osc):
                if abs(self.K[i, j]) > 1e-10:
                    self.logical_zz(i, j, self.K[i, j] * dt, qc)

        for osc in range(self.n_osc):
            self.x_syndrome_extract(osc, qc)
            self.z_syndrome_extract(osc, qc)

        return qc

    def physical_qubit_budget(self) -> dict[str, int]:
        """Return the patch-based physical-qubit accounting.

        Returns
        -------
        dict[str, int]
            Oscillator count, distance, per-patch data/ancilla/physical counts,
            total physical count, and the theoretical distance-derived value
            ``(d - 1) // 2`` under key ``correctable_errors``.

        Notes
        -----
        ``correctable_errors`` is a code-distance resource label; this structural
        scaffold does not execute or verify that correction capability.

        """
        return {
            "n_osc": self.n_osc,
            "code_distance": self.spec.distance,
            "data_per_osc": self.spec.n_data,
            "ancilla_per_osc": self.spec.n_ancilla,
            "physical_per_osc": self.spec.n_physical,
            "total_physical": self.total_qubits,
            "correctable_errors": (self.spec.distance - 1) // 2,
        }
