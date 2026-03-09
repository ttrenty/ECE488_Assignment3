"""
Q2: Build specific single-qubit and two-qubit circuit families whose
measurement behaviour follows prescribed target distributions.

Assignment framing:
- Work with a few concrete circuit families built from simple gate blocks.
- Match specific expectation or basis-probability profiles rather than solving
  the fully general state-synthesis problem.
- Compare one chosen two-qubit family under noiseless and noisy simulation.
"""

from __future__ import annotations

import numpy as np

# QuantumCircuit is the basic circuit object; transpile() prepares measured circuits
# for the simulator backend used in the noisy part.
from qiskit import QuantumCircuit, transpile

# Statevector is used for exact probabilities/expectation values, while SparsePauliOp
# is a compact way to define X, Y, Z observables.
from qiskit.quantum_info import SparsePauliOp, Statevector

# AerSimulator is used for the shot-based noisy simulation.
from qiskit_aer import AerSimulator

# These are the building blocks of a simple noise model:
# - NoiseModel collects all noise channels,
# - ReadoutError perturbs measurement outcomes,
# - depolarizing_error perturbs gates.
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

_SINGLE_QUBIT_PAULIS = {
    "X": SparsePauliOp.from_list([("X", 1.0)]),
    "Y": SparsePauliOp.from_list([("Y", 1.0)]),
    "Z": SparsePauliOp.from_list([("Z", 1.0)]),
}


def build_simple_noise_model(noise_strength: float) -> NoiseModel:
    """
    Build the simple assignment noise model used in Q2.e.

    This helper is provided. Students are not expected to design the
    depolarizing/readout channels from scratch; reuse this exact helper inside
    noisy_basis_probabilities().
    """
    noise_strength = float(noise_strength)
    if not (0.0 <= noise_strength < 0.5):
        raise ValueError("noise_strength must be in [0, 0.5)")

    noise_model = NoiseModel()
    gate_error_1q = depolarizing_error(noise_strength, 1)
    gate_error_2q = depolarizing_error(min(2.0 * noise_strength, 0.49), 2)
    noise_model.add_all_qubit_quantum_error(
        gate_error_1q, ["ry", "rz", "rx", "h", "sdg"]
    )
    noise_model.add_all_qubit_quantum_error(gate_error_2q, ["cx"])
    readout_error = ReadoutError(
        [
            [1.0 - noise_strength, noise_strength],
            [noise_strength, 1.0 - noise_strength],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def pauli_expectations_from_circuit(circuit: QuantumCircuit) -> np.ndarray:
    """
    Compute exact <X>, <Y>, <Z> from the circuit statevector.

    Hint: Statevector.from_instruction() lets you evaluate expectation values
    exactly, so no shot-based sampling is needed here. You may find it convenient
    to use the _SINGLE_QUBIT_PAULIS dictionary defined above to get the Pauli operators.

    Expected return contract:
    - return exactly one numpy array
    - the array must be a 1D float numpy array with shape `(3,)`
    - the entries must be ordered as `[<X>, <Y>, <Z>]`
    """

    raise NotImplementedError("pauli_expectations_from_circuit not implemented")


def build_xz_half_circle_encoder(x: float) -> QuantumCircuit:
    """
    Build the 1-qubit family that traces the XZ-plane half-circle for x in [0, 1].

    Required gate set: use only a single RY() rotation on |0>.

    Expected return contract:
    - return exactly one `QuantumCircuit`
    - the circuit must act on exactly 1 qubit
    - the circuit should contain no measurements
    """

    raise NotImplementedError("build_xz_half_circle_encoder not implemented")


def build_mapped_xz_half_circle_encoder(x: float) -> QuantumCircuit:
    """
    Build a remapped version of the same XZ-plane half-circle family.

    Compared with build_xz_half_circle_encoder(), this version follows the same
    geometric path but changes how quickly the state moves along that path.

    Required gate set: use one classical mapping u(x) and then a single RY()
    rotation on |0>.
    Hint: a smooth nonlinear map such as 3x^2 - 2x^3 changes how quickly the
    Bloch vector moves along the same XZ-plane path without changing the basic
    gate block.

    Expected return contract:
    - return exactly one `QuantumCircuit`
    - the circuit must act on exactly 1 qubit
    - the circuit should contain no measurements
    """

    raise NotImplementedError("build_mapped_xz_half_circle_encoder not implemented")


def _smoothstep(x: float) -> float:
    """
    One example of a smooth remapping u(x) for the mapped XZ-plane sweep.

    This helper is optional but useful in build_mapped_xz_half_circle_encoder().
    It keeps the same endpoints while changing how quickly the angle evolves.
    """
    x = float(x)
    return 3.0 * x * x - 2.0 * x * x * x


def build_xy_plane_circle_encoder(x: float) -> QuantumCircuit:
    """
    Build the 1-qubit family that traces a full circle in the XY plane.

    Required gate set: use one H() gate and one RZ() rotation.

    Expected return contract:
    - return exactly one `QuantumCircuit`
    - the circuit must act on exactly 1 qubit
    - the circuit should contain no measurements
    """

    raise NotImplementedError("build_xy_plane_circle_encoder not implemented")


def build_reuploaded_phase_encoder(x: float) -> QuantumCircuit:
    """
    Build a 1-qubit family with one data re-uploading step.

    Required gate set: use one RX() and one RZ(), both depending on x.
    Hint: re-encoding the same scalar twice can introduce higher-frequency
    components into the measured expectation curves.

    Expected return contract:
    - return exactly one `QuantumCircuit`
    - the circuit must act on exactly 1 qubit
    - the circuit should contain no measurements
    """

    raise NotImplementedError("build_reuploaded_phase_encoder not implemented")


def build_bell_family_circuit(x: float) -> QuantumCircuit:
    """
    Build the 2-qubit Bell-like family used in Q2.c and Q2.e.

    Required gate set: one RY() on qubit 0 followed by one CX() from qubit 0 to
    qubit 1.
    Hint: the resulting state should interpolate between |00> and |11>.

    Expected return contract:
    - return exactly one `QuantumCircuit`
    - the circuit must act on exactly 2 qubits
    - the circuit should contain no measurements
    """

    raise NotImplementedError("build_bell_family_circuit not implemented")


def _validate_basis(basis: str) -> str:
    basis = str(basis).upper()
    if basis not in {"X", "Y", "Z"}:
        raise ValueError("basis must be one of {'X', 'Y', 'Z'}")
    return basis


def _append_basis_rotation(circuit: QuantumCircuit, basis: str) -> None:
    basis = _validate_basis(basis)
    for qubit in range(circuit.num_qubits):
        if basis == "X":
            circuit.h(qubit)
        elif basis == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)


def basis_probabilities_from_circuit(
    circuit: QuantumCircuit, basis: str = "Z"
) -> np.ndarray:
    """
    Return exact basis probabilities in the Z, X, or Y product basis.

    Hint: rotate the requested basis back to the computational basis on every
    qubit, then extract exact statevector probabilities.

    Expected return contract:
    - return exactly one numpy array
    - the array must be a 1D float numpy array with shape `(2**n,)`, where `n = circuit.num_qubits`
    - the probabilities must sum to 1
    - use the standard computational-basis ordering after the basis rotation;
      for 2 qubits this means `[p00, p01, p10, p11]`
    """

    raise NotImplementedError("basis_probabilities_from_circuit not implemented")


def _counts_to_probs(
    circuit: QuantumCircuit, counts: dict[str, int], shots: int
) -> np.ndarray:
    probs = np.zeros(1 << circuit.num_qubits, dtype=float)
    for bitstring, count in counts.items():
        index = int(bitstring.replace(" ", ""), 2)
        probs[index] = count / shots
    return probs


_build_simple_noise_model = build_simple_noise_model


def noisy_basis_probabilities(
    circuit: QuantumCircuit,
    basis: str = "Z",
    noise_strength: float = 0.02,
    shots: int = 4096,
    seed: int = 1234,
) -> np.ndarray:
    """
    Estimate basis probabilities with a noisy simulator.

    Hint: append the same basis rotation as in basis_probabilities_from_circuit(),
    add measurements to all qubits (.measure_all()), use AerSimulator to run simulation,
    use build_simple_noise_model(noise_strength), and convert counts to a normalized
    probability vector using _counts_to_probs().

    Expected return contract:
    - return exactly one numpy array
    - the array must be a 1D float numpy array with shape `(2**n,)`, where `n = circuit.num_qubits`
    - the probabilities must sum to approximately 1
    - use the same basis-state ordering as `basis_probabilities_from_circuit()`;
      for 2 qubits this means `[p00, p01, p10, p11]`
    """

    raise NotImplementedError("noisy_basis_probabilities not implemented")
