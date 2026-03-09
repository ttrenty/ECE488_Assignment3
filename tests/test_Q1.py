import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector

from tests_utils import (
    brute_force_best_qubo,
    ensure_output_dir,
    import_impl,
    load_student_info,
    require_impl,
    save_quantum_circuit_image,
)

Q1 = import_impl("Q1_qubo_qaoa")


def _sample_problem():
    values = np.array([2.0, 1.5, 1.2, 1.8], dtype=float)
    conflicts = [(0, 1), (1, 2), (2, 3), (0, 3)]
    penalty = 3.0
    return values, conflicts, penalty


def _write_q1_distribution_histogram(path, bitstrings, probabilities, energies, title):
    bitstrings = list(bitstrings)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    energies = np.asarray(energies, dtype=float).reshape(-1)

    order = np.argsort(probabilities)[::-1]
    bitstrings = [bitstrings[idx] for idx in order]
    probabilities = probabilities[order]
    energies = energies[order]

    cmap = plt.colormaps["viridis_r"]
    energy_min = float(np.min(energies))
    energy_max = float(np.max(energies))
    if np.isclose(energy_min, energy_max):
        colors = [cmap(0.5)] * energies.size
    else:
        colors = [
            cmap((float(energy) - energy_min) / (energy_max - energy_min))
            for energy in energies
        ]

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 7.8), sharex=True)
    axes[0].bar(
        np.arange(probabilities.size),
        probabilities,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_ylabel("probability")
    axes[0].set_title("Final-state probabilities")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(
        np.arange(energies.size),
        energies,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_ylabel("QUBO energy")
    axes[1].set_title("Corresponding basis-state energies")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(np.arange(len(bitstrings)))
    axes[1].set_xticklabels(bitstrings, rotation=90, fontsize=8)
    axes[1].set_xlabel("computational basis state")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def test_build_profit_conflict_qubo_structure():
    values, conflicts, penalty = _sample_problem()
    qubo = require_impl(Q1.build_profit_conflict_qubo, values, conflicts, penalty)
    conflict_set = {tuple(sorted(edge)) for edge in conflicts}

    assert isinstance(
        qubo, np.ndarray
    ), "build_profit_conflict_qubo() must return a numpy array."
    assert qubo.shape == (
        4,
        4,
    ), "QUBO matrix must have shape (n, n) for the four-item sample problem."
    assert np.allclose(qubo, qubo.T), "QUBO matrix must be symmetric."
    assert np.allclose(
        np.diag(qubo), -values
    ), "QUBO diagonal should store the linear profit terms as -values."

    for i, j in conflicts:
        assert np.isclose(
            qubo[i, j] + qubo[j, i], penalty
        ), f"Conflict edge {(i, j)} should contribute the requested pair penalty in x^T Q x."

    for i in range(qubo.shape[0]):
        for j in range(i + 1, qubo.shape[1]):
            if (i, j) not in conflict_set:
                assert np.isclose(
                    qubo[i, j] + qubo[j, i], 0.0
                ), f"Non-conflict pair {(i, j)} should not introduce any quadratic penalty."


def test_qubo_energy_matches_manual_expression():
    values, conflicts, penalty = _sample_problem()
    qubo = require_impl(Q1.build_profit_conflict_qubo, values, conflicts, penalty)

    x = np.array([1, 0, 1, 0], dtype=int)
    energy = require_impl(Q1.qubo_energy, x, qubo)

    manual = -(values[0] + values[2])
    for i, j in conflicts:
        manual += penalty * x[i] * x[j]

    assert np.isclose(
        energy, manual
    ), "qubo_energy() must match the manual profit-minus-conflict expression."


def test_qubo_to_ising_matches_basis_energies():
    values, conflicts, penalty = _sample_problem()
    qubo = require_impl(Q1.build_profit_conflict_qubo, values, conflicts, penalty)
    hamiltonian, offset = require_impl(Q1.qubo_to_ising_hamiltonian, qubo)

    assert (
        hamiltonian.num_qubits == qubo.shape[0]
    ), "Ising Hamiltonian must act on one qubit per QUBO variable."
    assert isinstance(
        offset, float
    ), "qubo_to_ising_hamiltonian() must return a Python float offset."
    for label in hamiltonian.paulis.to_labels():
        assert set(label).issubset(
            {"I", "Z"}
        ), "Cost Hamiltonian should contain only I/Z terms for a diagonal Ising model."

    n = qubo.shape[0]
    for idx in [0, 3, 6, 9, 15]:
        x = np.array([(idx >> i) & 1 for i in range(n)], dtype=int)
        e_qubo = require_impl(Q1.qubo_energy, x, qubo)

        label = "".join(str((idx >> bit) & 1) for bit in range(n - 1, -1, -1))
        state = Statevector.from_label(label)
        e_ising = float(np.real(state.expectation_value(hamiltonian)) + offset)

        assert np.isclose(
            e_qubo, e_ising, atol=1e-8
        ), f"Basis-state energy mismatch for bitstring index {idx}: QUBO and Ising energies must agree."


def test_solve_with_qaoa_grid_is_near_optimal():
    _ = load_student_info()  # keeps consistency with assignment style
    values, conflicts, penalty = _sample_problem()
    qubo = require_impl(Q1.build_profit_conflict_qubo, values, conflicts, penalty)
    hamiltonian, _ = require_impl(Q1.qubo_to_ising_hamiltonian, qubo)
    save_quantum_circuit_image(
        qaoa_ansatz(hamiltonian, reps=1),
        ensure_output_dir() / "Q1_b_qaoa_ansatz_circuit_test.png",
        title="Q1.b QAOA ansatz circuit",
    )

    _, e_opt = brute_force_best_qubo(qubo)
    x_qaoa, e_qaoa, angles = require_impl(Q1.solve_with_qaoa_grid, qubo, 1, 9)

    assert isinstance(
        x_qaoa, np.ndarray
    ), "solve_with_qaoa_grid() must return the selected bitstring as a numpy array."
    assert x_qaoa.shape == (
        4,
    ), "Returned QAOA bitstring must have one entry per QUBO variable."
    assert set(np.unique(x_qaoa)).issubset(
        {0, 1}
    ), "Returned QAOA bitstring must contain only binary values."

    e_from_x = require_impl(Q1.qubo_energy, x_qaoa, qubo)
    assert np.isclose(
        e_qaoa, e_from_x
    ), "Reported QAOA energy must equal qubo_energy(best_x, qubo)."

    assert isinstance(
        angles, dict
    ), "solve_with_qaoa_grid() must return the chosen QAOA angles as a dictionary."
    assert (
        "beta" in angles and "gamma" in angles
    ), "Angle dictionary must contain at least 'beta' and 'gamma'."
    assert np.isscalar(
        angles["beta"]
    ), "For reps=1, beta should be returned as a scalar."
    assert np.isscalar(
        angles["gamma"]
    ), "For reps=1, gamma should be returned as a scalar."

    # QAOA with p=1 and coarse grid is approximate; keep tolerance generous.
    assert (
        e_qaoa <= e_opt + 0.75
    ), "QAOA grid search should find an energy reasonably close to the brute-force optimum."


def test_q1_c_six_item_instance_generates_distribution_histogram():
    values = np.array([4.0, 5.0, 3.0, 6.0, 2.0, 5.0], dtype=float)
    conflicts = [(0, 1), (1, 3), (2, 3), (2, 4), (4, 5)]
    penalty = 8.0
    qubo = require_impl(Q1.build_profit_conflict_qubo, values, conflicts, penalty)

    x_qaoa, e_qaoa, details = require_impl(Q1.solve_with_qaoa_grid, qubo, 2, 9)
    bitstrings = list(details["basis_bitstrings"])
    probabilities = np.asarray(details["final_probabilities"], dtype=float)
    energies = np.asarray(details["basis_energies"], dtype=float)

    assert isinstance(
        details, dict
    ), "solve_with_qaoa_grid() must return a details dictionary as its third output."
    assert (
        "beta" in details and "gamma" in details
    ), "Q1.c details dictionary must include beta and gamma."
    assert isinstance(details["beta"], np.ndarray) and details["beta"].shape == (
        2,
    ), "For Q1.c with reps=2, details['beta'] should be a length-4 numpy array with one beta per QAOA layer."
    assert isinstance(details["gamma"], np.ndarray) and details["gamma"].shape == (
        2,
    ), "For Q1.c with reps=2, details['gamma'] should be a length-4 numpy array with one gamma per QAOA layer."
    assert (
        len(bitstrings) == 2 ** qubo.shape[0]
    ), "Q1.c details should include one bitstring label per computational basis state."
    assert probabilities.shape == (
        2 ** qubo.shape[0],
    ), "Q1.c details should include one probability per computational basis state."
    assert (
        energies.shape == probabilities.shape
    ), "Q1.c details should include one energy per computational basis state."
    assert np.isclose(
        np.sum(probabilities), 1.0, atol=1e-10
    ), "Q1.c final-state probabilities must sum to one."

    x_opt, e_opt = brute_force_best_qubo(qubo)
    assert (
        e_qaoa <= e_opt + 1.0
    ), "Q1.c should use the depth-4 QAOA angles to reach a near-optimal solution on the six-item instance."
    assert np.isclose(
        e_qaoa, require_impl(Q1.qubo_energy, x_qaoa, qubo)
    ), "Q1.c reported energy must match qubo_energy(best_x, qubo)."
    assert np.array_equal(
        x_opt, np.array([1, 0, 0, 1, 0, 1], dtype=int)
    ), "The six-item Q1.c exact optimum should match the assignment correction."

    _write_q1_distribution_histogram(
        ensure_output_dir() / "Q1_c_qaoa_probability_energy_histogram.png",
        bitstrings,
        probabilities,
        energies,
        title="Q1.c six-item QAOA distribution and basis energies",
    )
