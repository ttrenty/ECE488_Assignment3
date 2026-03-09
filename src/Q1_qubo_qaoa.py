"""
Q1: Model a profit-maximization problem with pairwise conflicts and solve a
small instance with QAOA.

Assignment framing:
- The mathematical model should be written for a general number of items n.
- Any simulated instance in this assignment must satisfy n <= 6, because the
  intended mapping uses one qubit per item maximum.
- A conflict edge (i, j) means items i and j cannot both be selected.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# qaoa_ansatz() builds the standard QAOA circuit once you have a cost Hamiltonian.
from qiskit.circuit.library import qaoa_ansatz

# SparsePauliOp is the Ising-Hamiltonian format expected by Qiskit's QAOA tools.
from qiskit.quantum_info import SparsePauliOp

# Statevector lets you evaluate the circuit exactly without shot noise.
from qiskit.quantum_info import Statevector

# minimize() is a convenient way to refine all QAOA layer angles after a coarse grid search.
from scipy.optimize import minimize


ConflictEdge = tuple[int, int]


def build_profit_conflict_qubo(
    values: np.ndarray,
    conflicts: Iterable[ConflictEdge],
    penalty: float,
) -> np.ndarray:
    """
    Build the QUBO matrix Q (shape n x n) for the objective

        E(x) = x^T Q x

    where x is a binary vector (x_i in {0,1}) indicating which items are chosen.

    In this problem:
    - choosing item i gives reward v_i
    - choosing two conflicting items (i, j) adds a penalty λ

    The target objective is therefore

        E(x) = -sum_i v_i x_i + λ sum_(i,j in conflicts) x_i x_j

    Your job is to encode this objective in a QUBO matrix Q.

    Key idea:
    When computing x^T Q x,

        x^T Q x = sum_i Q[i,i] x_i^2 + sum_{i != j} Q[i,j] x_i x_j

    and since x_i^2 = x_i for binary variables, the diagonal entries represent
    the linear terms.

    Implementation hints:
    - Put the linear coefficients (-v_i) on the diagonal of Q.
    - For each conflict (i, j), add a penalty term for x_i x_j.
    - Because Q is symmetric and both Q[i,j] and Q[j,i] appear in x^T Q x,
      split the penalty equally between them.
    - Always check that the indices in conflicts are valid before writing into Q.

    Expected return contract:
    - return exactly one numpy array `qubo`
    - `qubo`: 2D numpy array of floats with shape `(n, n)`
    - `qubo` must be symmetric
    - `np.diag(qubo)` should contain the linear item terms
    - the conflict penalty for (i, j) should appear through
          qubo[i, j] + qubo[j, i]
      in the expression x^T Q x.
    """
    raise NotImplementedError("build_profit_conflict_qubo not implemented")


def qubo_energy(x: np.ndarray, qubo: np.ndarray) -> float:
    """
    Return the scalar QUBO energy x^T Q x.

    Expected return contract:
    - return exactly one scalar value
    - the returned value must be a Python float (or a numpy scalar convertible to float)
    """
    raise NotImplementedError("qubo_energy not implemented")


def _pauli_label_for_qubits(n_qubits: int, z_positions: tuple[int, ...]) -> str:
    """
    Build a Pauli-string label containing Z on the requested qubits and I elsewhere.

    Examples for n_qubits = 3:
    - z_positions = (0,)    -> "IIZ"
    - z_positions = (1, 2)  -> "ZZI"
    - z_positions = ()      -> "III"

    Why this helper is useful:
    Qiskit stores a term such as Z on qubit 0 and Z on qubit 2 as a string like
    "ZIZ" or "IZZ" depending on the qubit ordering convention. This helper
    centralizes that formatting so qubo_to_ising_hamiltonian() can focus on the
    math rather than string assembly.
    """
    chars = ["I"] * n_qubits
    for idx in z_positions:
        chars[n_qubits - 1 - idx] = "Z"
    return "".join(chars)


def qubo_to_ising_hamiltonian(qubo: np.ndarray) -> tuple[SparsePauliOp, float]:
    """
    Complete the missing parts of the scaffold to convert a QUBO objective into
    an Ising Hamiltonian H together with a constant offset cst.

    The goal is to produce H and cst such that, for every computational-basis state,

        E_QUBO(x) == <z|H|z> + cst

    Overview:
    - rewrite each binary variable with x_i = (1 - Z_i) / 2,
    - accumulate the constant, single-Z, and ZZ contributions,
    - rebuild the final SparsePauliOp from those coefficients.

    The scaffold below already separates the work into:
    - diagonal QUBO terms,
    - pairwise QUBO terms,
    - final Pauli-string assembly.

    Expected return contract:
    - return exactly one tuple `(hamiltonian, offset)`
    - `hamiltonian` must be a `SparsePauliOp` acting on `n = qubo.shape[0]` qubits
    - `offset` must be a Python float
    - for every computational-basis bitstring `x`, the QUBO energy must satisfy
      `qubo_energy(x, qubo) == <z|hamiltonian|z> + offset`
    """
    # raise NotImplementedError(
    #     "Remove this line and complete the scaffold below in qubo_to_ising_hamiltonian()."
    # )

    # Step 1: validate the input and extract the number of qubits.
    qubo = np.asarray(qubo, dtype=float)
    if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
        raise ValueError("qubo must be a square matrix")

    n = qubo.shape[0]

    # Step 2: accumulate the constant term, all single-Z coefficients, and all
    # pairwise ZZ coefficients separately.
    constant_offset = 0.0
    z_coeffs = np.zeros(n, dtype=float)
    zz_coeffs: dict[tuple[int, int], float] = {}

    # Step 3: process the diagonal QUBO terms Q[i, i] * x_i.
    # Using x_i = (1 - Z_i) / 2, each diagonal term contributes:
    #   constant += Q[i, i] / 2
    #   Z_i      += -Q[i, i] / 2
    for i in range(n):
        a_i = qubo[i, i]
        constant_offset += ...
        z_coeffs[i] += ...

    # Step 4: process the quadratic terms.
    # Because E(x) = x^T Q x, the coefficient multiplying x_i x_j for i != j is
    # Q[i, j] + Q[j, i]. After substitution, x_i x_j contributes:
    #   constant += b_ij / 4
    #   Z_i      += -b_ij / 4
    #   Z_j      += -b_ij / 4
    #   Z_i Z_j  +=  b_ij / 4
    for i in range(n):
        for j in range(i + 1, n):
            b_ij = qubo[i, j] + qubo[j, i]
            if abs(b_ij) < 1e-14:
                continue
            constant_offset += ...
            z_coeffs[i] += ...
            z_coeffs[j] += ...
            if (i, j) not in zz_coeffs:
                zz_coeffs[(i, j)] = 0.0
            zz_coeffs[(i, j)] += ...

    # Step 5: convert the accumulated coefficients into Pauli terms.
    # The helper _pauli_label_for_qubits() formats strings such as "IIZ" or "ZZI"
    # so no need to build those labels manually.
    terms: list[tuple[str, complex]] = []
    for i, coeff in enumerate(z_coeffs):
        if abs(coeff) > 1e-14:
            terms.append((_pauli_label_for_qubits(n, (i,)), complex(coeff)))

    for (i, j), coeff in zz_coeffs.items():
        if abs(coeff) > 1e-14:
            terms.append((_pauli_label_for_qubits(n, (i, j)), complex(coeff)))

    # Step 6: build the Hamiltonian and return it together with the constant part.
    return SparsePauliOp.from_list(terms).simplify(), float(constant_offset)


def _extract_layer_params(ansatz, prefix: str) -> list:
    """
    Return the QAOA parameters whose names start with a given prefix, sorted by layer index.

    Example:
    - prefix = "β" returns [β[0], β[1], ..., β[p-1]]
    - prefix = "γ" returns [γ[0], γ[1], ..., γ[p-1]]
    """
    params = [p for p in ansatz.parameters if p.name.startswith(prefix)]

    def _idx(param) -> int:
        return int(param.name.split("[")[1].split("]")[0])

    return sorted(params, key=_idx)


def _coerce_layer_values(values: np.ndarray | float, count: int) -> np.ndarray:
    """
    Convert either a scalar or a length-`count` vector into a 1D float array.

    Why this helper is useful:
    - for reps = 1, a single scalar beta/gamma is natural,
    - for reps > 1, QAOA usually needs one beta and one gamma per layer.
    This helper lets you support both cases cleanly when binding parameters.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(count, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != count:
        raise ValueError(f"expected {count} QAOA angles, got {arr.size}")
    return arr.astype(float, copy=False)


def _bind_qaoa_layer_params(
    beta_params: list,
    gamma_params: list,
    beta_values: np.ndarray | float,
    gamma_values: np.ndarray | float,
) -> dict:
    """
    Build the dictionary expected by ansatz.assign_parameters(...).

    This is a small utility so solve_with_qaoa_grid() can focus on the search
    logic instead of repeatedly converting between scalars, vectors, and Qiskit
    Parameter objects.
    """
    beta_arr = _coerce_layer_values(beta_values, len(beta_params))
    gamma_arr = _coerce_layer_values(gamma_values, len(gamma_params))

    bind = {param: float(value) for param, value in zip(beta_params, beta_arr)}
    bind.update({param: float(value) for param, value in zip(gamma_params, gamma_arr)})
    return bind


def solve_with_qaoa_grid(
    qubo: np.ndarray,
    reps: int = 1,
    grid_size: int = 11,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """
    Complete the missing parts of the scaffold to solve a small QUBO instance
    with a QAOA ansatz and a coarse-to-fine angle search.

    Overview:
    - convert the QUBO into an Ising cost Hamiltonian,
    - build a QAOA circuit of depth `reps`,
    - evaluate candidate beta/gamma angles with exact statevectors,
    - refine promising angle choices with a local optimizer,
    - decode a candidate bitstring from the final optimized state.

    The scaffold already handles most of the search logic; your job is to fill in
    the missing expressions that build the Hamiltonian, evaluate the expectation
    value, construct the final bound state.

    Expected return contract:
    - return exactly one tuple `(best_x, best_energy, best_angles)`
    - `best_x`: 1D int numpy array with shape `(n,)` and entries in `{0, 1}`
    - `best_energy`: Python float equal to `qubo_energy(best_x, qubo)`
    - `best_angles`: dictionary containing at least the keys `"beta"` and `"gamma"`
    - for `reps = 1`, `best_angles["beta"]` and `best_angles["gamma"]` should be floats
    - for `reps > 1`, they should be 1D float numpy arrays with shape `(reps,)`
    - also include:
      `best_angles["final_probabilities"]`: float array with shape `(2**n,)`
      `best_angles["basis_energies"]`: float array with shape `(2**n,)`
      `best_angles["basis_bitstrings"]`: list[str] of length `2**n`
    """
    raise NotImplementedError(
        "Remove this line and complete the scaffold below in solve_with_qaoa_grid()."
    )

    qubo = np.asarray(qubo, dtype=float)
    if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
        raise ValueError("qubo must be a square matrix")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    # Build the diagonal Ising cost Hamiltonian used by QAOA, then create the
    # standard alternating cost/mixer circuit of depth `reps`.
    hamiltonian, _ = ...
    ansatz = ...

    # Qiskit stores one symbolic beta and one symbolic gamma per QAOA layer.
    beta_params = _extract_layer_params(ansatz, "β")
    gamma_params = _extract_layer_params(ansatz, "γ")

    # The coarse grid is only used to propose a few good starting points before
    # running the local optimizer on the full angle vector.
    beta_grid = np.linspace(0.0, np.pi / 2.0, int(grid_size))
    gamma_grid = np.linspace(0.0, np.pi, int(grid_size))

    def _expected_energy(theta: np.ndarray) -> float:
        # theta = [beta_0, ..., beta_{p-1}, gamma_0, ..., gamma_{p-1}]
        bind = _bind_qaoa_layer_params(
            beta_params,
            gamma_params,
            theta[:reps],
            theta[reps:],
        )
        state = Statevector.from_instruction(ansatz.assign_parameters(bind))
        return float(np.real(state.expectation_value(...)))

    coarse_scores: list[tuple[float, np.ndarray]] = []
    for beta in beta_grid:
        for gamma in gamma_grid:
            # First try repeated angles across all layers; this is cheap and
            # often gives a reasonable initialization.
            theta = np.concatenate(
                [
                    np.full(reps, float(beta), dtype=float),
                    np.full(reps, float(gamma), dtype=float),
                ]
            )
            coarse_scores.append((_expected_energy(theta), theta))

    coarse_scores.sort(key=lambda item: item[0])
    # Keep the best coarse grid points, then add one hand-crafted schedule and
    # a few random restarts to reduce the chance of getting stuck early.
    search_starts = [
        theta.copy() for _, theta in coarse_scores[: min(3, len(coarse_scores))]
    ]
    search_starts.append(
        np.concatenate(
            [
                np.linspace(np.pi / 8.0, np.pi / 3.0, reps, dtype=float),
                np.linspace(np.pi / 2.0, np.pi / 8.0, reps, dtype=float),
            ]
        )
    )

    rng = np.random.default_rng(0)
    for _ in range(4):
        search_starts.append(
            np.concatenate(
                [
                    rng.uniform(0.0, np.pi / 2.0, size=reps),
                    rng.uniform(0.0, np.pi, size=reps),
                ]
            )
        )

    # Do not keep only one shared beta/gamma when reps > 1.
    # Start from the best repeated-angle initialization, then refine one beta
    # and one gamma per layer with a local optimizer.
    best_exp = float(coarse_scores[0][0])
    best_theta = coarse_scores[0][1].copy()
    for theta0 in search_starts:
        # Also compare the raw starting point itself before refinement.
        initial_exp = _expected_energy(theta0)
        if initial_exp < best_exp:
            best_exp = ...
            best_theta = ...

        result = minimize(
            _expected_energy,
            x0=theta0,
            method="COBYLA",
            options={"maxiter": 300 if reps == 1 else 400, "rhobeg": 0.35, "tol": 1e-4},
        )
        candidate_theta = np.asarray(result.x, dtype=float)
        candidate_exp = _expected_energy(candidate_theta)
        if candidate_exp < best_exp:
            best_exp = ...
            best_theta = ...

    # Split the optimized angle vector back into the beta and gamma blocks.
    best_beta = np.asarray(best_theta[:reps], dtype=float)
    best_gamma = np.asarray(best_theta[reps:], dtype=float)

    # Rebuild the final optimized state so we can inspect its full probability
    # distribution over computational-basis bitstrings.
    final_bind = _bind_qaoa_layer_params(...)
    final_state = ...

    probs = final_state.probabilities()

    # Rather than scanning every basis state first, inspect the few most likely
    # outcomes and keep the one with the best QUBO energy.
    top_k = min(8, probs.size)
    top_idx = np.argpartition(probs, -top_k)[-top_k:]

    top_energies = np.array(
        [
            qubo_energy(
                np.array([(idx >> i) & 1 for i in range(qubo.shape[0])], dtype=int),
                qubo,
            )
            for idx in top_idx
        ],
        dtype=float,
    )

    best_idx = int(top_idx[int(np.argmin(top_energies))])
    n = qubo.shape[0]
    best_x = np.array([(best_idx >> i) & 1 for i in range(n)], dtype=int)
    best_energy = qubo_energy(best_x, qubo)

    # These full-basis arrays are returned so the tests can draw the Q1.c
    # histogram of probabilities and corresponding energies.
    basis_energies = np.array(
        [
            qubo_energy(
                np.array([(idx >> i) & 1 for i in range(qubo.shape[0])], dtype=int),
                qubo,
            )
            for idx in range(probs.size)
        ],
        dtype=float,
    )
    basis_bitstrings = [format(idx, f"0{n}b") for idx in range(probs.size)]

    return (
        best_x,
        float(best_energy),
        {
            "beta": float(best_beta[0]) if reps == 1 else best_beta,
            "gamma": float(best_gamma[0]) if reps == 1 else best_gamma,
            "final_probabilities": probs.astype(float),
            "basis_energies": basis_energies,
            "basis_bitstrings": basis_bitstrings,
        },
    )
