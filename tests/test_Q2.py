from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere
from qiskit.visualization.bloch import Bloch

from tests_utils import (
    ensure_output_dir,
    import_impl,
    load_student_info,
    make_gif_from_matplotlib_frames,
    maybe_publish_reference_artifact,
    require_impl,
    save_quantum_circuit_image,
    write_line_plot,
)

Q2 = import_impl("Q2_distribution_matching")

_BELL_X_LABELS = ["|++>", "|+->", "|-+>", "|-->"]
_COLORS_3 = ["#1f4e79", "#c44536", "#0b6e4f"]
_COLORS_4 = ["#1f4e79", "#c44536", "#0b6e4f", "#7f3c8d"]


def _smoothstep(x: float) -> float:
    x = float(x)
    return 3.0 * x * x - 2.0 * x * x * x


def _instruction_names(circuit) -> list[str]:
    return [
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name != "barrier"
    ]


def _xz_target(x: float) -> np.ndarray:
    return np.array([np.sin(np.pi * x), 0.0, np.cos(np.pi * x)], dtype=float)


def _xy_target(x: float) -> np.ndarray:
    return np.array(
        [np.cos(2.0 * np.pi * x), np.sin(2.0 * np.pi * x), 0.0], dtype=float
    )


def _mapped_xz_path_target(x: float) -> np.ndarray:
    u = _smoothstep(float(x))
    return np.array([np.sin(np.pi * u), 0.0, np.cos(np.pi * u)], dtype=float)


def _reuploaded_target(x: float) -> np.ndarray:
    angle = 2.0 * np.pi * float(x)
    return np.array(
        [
            np.sin(angle) ** 2,
            -np.sin(angle) * np.cos(angle),
            np.cos(angle),
        ],
        dtype=float,
    )


def _bell_z_probs(x: float) -> np.ndarray:
    c = float(np.cos(0.5 * np.pi * x))
    s = float(np.sin(0.5 * np.pi * x))
    return np.array([c * c, 0.0, 0.0, s * s], dtype=float)


def _bell_x_probs(x: float) -> np.ndarray:
    c = float(np.cos(0.5 * np.pi * x))
    s = float(np.sin(0.5 * np.pi * x))
    even = ((c + s) ** 2) / 4.0
    odd = ((c - s) ** 2) / 4.0
    return np.array([even, odd, odd, even], dtype=float)


def _generate_bloch_gif(output_path, bloch_points, title):
    bloch_points = np.asarray(bloch_points, dtype=float).reshape(-1, 3)

    make_gif_from_matplotlib_frames(
        output_path,
        lambda idx: _build_bloch_path_frame(bloch_points, idx, title),
        n_frames=bloch_points.shape[0],
        delay_cs=10,
    )


def _build_bloch_path_frame(points: np.ndarray, idx: int, title: str):
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    bloch = Bloch(fig=fig, axes=ax, font_size=14)
    bloch.vector_color = ["#111111"]
    bloch.point_color = ["#7f3c8d"]
    bloch.point_marker = ["o"]
    bloch.point_size = [18]
    bloch.add_points(points[: idx + 1].T, meth="m")
    bloch.add_vectors(points[idx].tolist())
    bloch.render()

    fig.suptitle(title, y=0.95, fontsize=14)
    return fig


def test_pauli_expectations_from_circuit_computes_exact_expectations():
    # Test on a few simple states with known expectations.
    qc = QuantumCircuit(1)
    qc.id(0)
    assert np.allclose(
        require_impl(Q2.pauli_expectations_from_circuit, qc),
        [0.0, 0.0, 1.0],
        atol=1e-6,
    ), "Expectation values for |0> state are incorrect."

    qc.x(0)
    assert np.allclose(
        require_impl(Q2.pauli_expectations_from_circuit, qc),
        [0.0, 0.0, -1.0],
        atol=1e-6,
    ), "Expectation values for |1> state are incorrect."

    qc = QuantumCircuit(1)
    qc.h(0)
    assert np.allclose(
        require_impl(Q2.pauli_expectations_from_circuit, qc),
        [1.0, 0.0, 0.0],
        atol=1e-6,
    ), "Expectation values for |+> state are incorrect."


def test_xz_half_circle_encoder_matches_target_and_generates_plot():
    xs = np.linspace(0.0, 1.0, 101)
    measured = []
    output_dir = ensure_output_dir()
    save_quantum_circuit_image(
        require_impl(Q2.build_xz_half_circle_encoder, 0.25),
        output_dir / "Q2_b_xz_half_circle_circuit.png",
        title="Q2.b XZ half-circle circuit",
    )
    prototype = require_impl(Q2.build_xz_half_circle_encoder, 0.25)
    assert (
        prototype.num_qubits == 1
    ), "build_xz_half_circle_encoder() should return a one-qubit circuit."
    assert (
        prototype.num_clbits == 0
    ), "Single-qubit encoder circuits should not contain classical bits."
    assert _instruction_names(prototype) == [
        "ry"
    ], "XZ half-circle encoder should use the expected single RY gate structure."

    for x in xs:
        qc = require_impl(Q2.build_xz_half_circle_encoder, float(x))
        bloch = require_impl(Q2.pauli_expectations_from_circuit, qc)
        target = _xz_target(float(x))
        assert bloch.shape == (
            3,
        ), "pauli_expectations_from_circuit() must return [<X>, <Y>, <Z>]."
        assert np.allclose(
            bloch, target, atol=1e-6
        ), "XZ half-circle encoder does not match the target Bloch-vector path."
        measured.append(bloch)

    measured = np.asarray(measured, dtype=float)
    write_line_plot(
        output_dir / "Q2_b_xz_half_circle_expectations.png",
        xs,
        [
            ("<X>", measured[:, 0], _COLORS_3[0]),
            ("<Y>", measured[:, 1], _COLORS_3[1]),
            ("<Z>", measured[:, 2], _COLORS_3[2]),
        ],
        title="Q2 one-qubit family A: XZ-plane half-circle",
        y_min=-1.0,
        y_max=1.0,
        y_label="expectation",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_b_xz_half_circle_expectations.png"
    )
    _generate_bloch_gif(
        output_dir / "Q2_b_xz_half_circle_bloch.gif",
        measured[np.linspace(0, measured.shape[0] - 1, 25, dtype=int)],
        "Q2.b XZ-plane half-circle",
    )
    maybe_publish_reference_artifact(output_dir / "Q2_b_xz_half_circle_bloch.gif")


def test_xy_plane_circle_encoder_matches_target_and_generates_plot():
    xs = np.linspace(0.0, 1.0, 101)
    measured = []
    output_dir = ensure_output_dir()
    save_quantum_circuit_image(
        require_impl(Q2.build_xy_plane_circle_encoder, 0.25),
        output_dir / "Q2_b_xy_plane_circle_circuit.png",
        title="Q2.b XY-plane circle circuit",
    )
    prototype = require_impl(Q2.build_xy_plane_circle_encoder, 0.25)
    assert (
        prototype.num_qubits == 1
    ), "build_xy_plane_circle_encoder() should return a one-qubit circuit."
    assert (
        prototype.num_clbits == 0
    ), "Single-qubit encoder circuits should not contain classical bits."
    assert _instruction_names(prototype) == [
        "h",
        "rz",
    ], "XY-plane circle encoder should use the expected H then RZ gate structure."

    for x in xs:
        qc = require_impl(Q2.build_xy_plane_circle_encoder, float(x))
        bloch = require_impl(Q2.pauli_expectations_from_circuit, qc)
        target = _xy_target(float(x))
        assert bloch.shape == (
            3,
        ), "pauli_expectations_from_circuit() must return [<X>, <Y>, <Z>]."
        assert np.allclose(
            bloch, target, atol=1e-6
        ), "XY-plane circle encoder does not match the target Bloch-vector path."
        measured.append(bloch)

    measured = np.asarray(measured, dtype=float)
    write_line_plot(
        output_dir / "Q2_b_xy_plane_circle_expectations.png",
        xs,
        [
            ("<X>", measured[:, 0], _COLORS_3[0]),
            ("<Y>", measured[:, 1], _COLORS_3[1]),
            ("<Z>", measured[:, 2], _COLORS_3[2]),
        ],
        title="Q2 one-qubit family B: XY-plane circle",
        y_min=-1.0,
        y_max=1.0,
        y_label="expectation",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_b_xy_plane_circle_expectations.png"
    )
    _generate_bloch_gif(
        output_dir / "Q2_b_xy_plane_circle_bloch.gif",
        measured[np.linspace(0, measured.shape[0] - 1, 25, dtype=int)],
        "Q2.b XY-plane circle",
    )
    maybe_publish_reference_artifact(output_dir / "Q2_b_xy_plane_circle_bloch.gif")


def test_mapped_xz_half_circle_encoder_matches_target_and_generates_plot():
    xs = np.linspace(0.0, 1.0, 101)
    measured = []
    output_dir = ensure_output_dir()
    save_quantum_circuit_image(
        require_impl(Q2.build_mapped_xz_half_circle_encoder, 0.25),
        output_dir / "Q2_b_mapped_xz_half_circle_circuit.png",
        title="Q2.b mapped XZ half-circle circuit",
    )
    prototype = require_impl(Q2.build_mapped_xz_half_circle_encoder, 0.25)
    assert (
        prototype.num_qubits == 1
    ), "build_mapped_xz_half_circle_encoder() should return a one-qubit circuit."
    assert (
        prototype.num_clbits == 0
    ), "Single-qubit encoder circuits should not contain classical bits."
    assert _instruction_names(prototype) == [
        "ry"
    ], "Mapped XZ half-circle encoder should use the expected single RY gate structure."

    for x in xs:
        qc = require_impl(Q2.build_mapped_xz_half_circle_encoder, float(x))
        bloch = require_impl(Q2.pauli_expectations_from_circuit, qc)
        target = _mapped_xz_path_target(float(x))
        assert bloch.shape == (
            3,
        ), "pauli_expectations_from_circuit() must return [<X>, <Y>, <Z>]."
        assert np.allclose(
            bloch, target, atol=1e-6
        ), "Mapped XZ half-circle encoder does not match the remapped target path."
        measured.append(bloch)

    measured = np.asarray(measured, dtype=float)
    write_line_plot(
        output_dir / "Q2_b_mapped_xz_half_circle_expectations.png",
        xs,
        [
            ("<X>", measured[:, 0], _COLORS_3[0]),
            ("<Y>", measured[:, 1], _COLORS_3[1]),
            ("<Z>", measured[:, 2], _COLORS_3[2]),
        ],
        title="Q2 one-qubit family C: remapped XZ-plane half-circle",
        y_min=-1.0,
        y_max=1.0,
        y_label="expectation",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_b_mapped_xz_half_circle_expectations.png"
    )
    _generate_bloch_gif(
        output_dir / "Q2_b_mapped_xz_half_circle_bloch.gif",
        measured[np.linspace(0, measured.shape[0] - 1, 25, dtype=int)],
        "Q2.b remapped XZ-plane half-circle",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_b_mapped_xz_half_circle_bloch.gif"
    )


def test_reuploaded_phase_encoder_matches_target_and_generates_plot():
    xs = np.linspace(0.0, 1.0, 101)
    measured = []
    output_dir = ensure_output_dir()
    save_quantum_circuit_image(
        require_impl(Q2.build_reuploaded_phase_encoder, 0.25),
        output_dir / "Q2_b_reuploaded_phase_circuit.png",
        title="Q2.b re-uploaded phase circuit",
    )
    prototype = require_impl(Q2.build_reuploaded_phase_encoder, 0.25)
    assert (
        prototype.num_qubits == 1
    ), "build_reuploaded_phase_encoder() should return a one-qubit circuit."
    assert (
        prototype.num_clbits == 0
    ), "Single-qubit encoder circuits should not contain classical bits."
    assert _instruction_names(prototype) == [
        "rx",
        "rz",
    ], "Re-uploaded phase encoder should use the expected RX then RZ gate structure."

    for x in xs:
        qc = require_impl(Q2.build_reuploaded_phase_encoder, float(x))
        bloch = require_impl(Q2.pauli_expectations_from_circuit, qc)
        target = _reuploaded_target(float(x))
        assert bloch.shape == (
            3,
        ), "pauli_expectations_from_circuit() must return [<X>, <Y>, <Z>]."
        assert np.allclose(
            bloch, target, atol=1e-6
        ), "Re-uploaded phase encoder does not match the target Bloch-vector path."
        measured.append(bloch)

    measured = np.asarray(measured, dtype=float)
    write_line_plot(
        output_dir / "Q2_b_reuploaded_phase_expectations.png",
        xs,
        [
            ("<X>", measured[:, 0], _COLORS_3[0]),
            ("<Y>", measured[:, 1], _COLORS_3[1]),
            ("<Z>", measured[:, 2], _COLORS_3[2]),
        ],
        title="Q2 one-qubit family D: re-uploaded phase",
        y_min=-1.0,
        y_max=1.0,
        y_label="expectation",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_b_reuploaded_phase_expectations.png"
    )
    _generate_bloch_gif(
        output_dir / "Q2_b_reuploaded_phase_bloch.gif",
        measured[np.linspace(0, measured.shape[0] - 1, 25, dtype=int)],
        "Q2.b re-uploaded phase",
    )
    maybe_publish_reference_artifact(output_dir / "Q2_b_reuploaded_phase_bloch.gif")


def test_bell_family_matches_specific_z_and_x_basis_distributions():
    xs = np.linspace(0.0, 1.0, 81)
    output_dir = ensure_output_dir()
    save_quantum_circuit_image(
        require_impl(Q2.build_bell_family_circuit, 0.5),
        output_dir / "Q2_c_bell_family_circuit.png",
        title="Q2.c Bell-family circuit",
    )
    prototype = require_impl(Q2.build_bell_family_circuit, 0.5)
    assert (
        prototype.num_qubits == 2
    ), "build_bell_family_circuit() should return a two-qubit circuit."
    assert (
        prototype.num_clbits == 0
    ), "Bell-family circuits should not contain classical bits."
    assert _instruction_names(prototype) == [
        "ry",
        "cx",
    ], "Bell-family circuit should use the expected RY then CX gate structure."
    z_curves = []
    x_curves = []

    for x in xs:
        qc = require_impl(Q2.build_bell_family_circuit, float(x))
        probs_z = require_impl(Q2.basis_probabilities_from_circuit, qc, "Z")
        probs_x = require_impl(Q2.basis_probabilities_from_circuit, qc, "X")
        assert np.allclose(
            probs_z, _bell_z_probs(float(x)), atol=1e-6
        ), "Bell-family Z-basis probabilities do not match the target distribution."
        assert np.allclose(
            probs_x, _bell_x_probs(float(x)), atol=1e-6
        ), "Bell-family X-basis probabilities do not match the target distribution."
        z_curves.append(probs_z)
        x_curves.append(probs_x)

    z_curves = np.asarray(z_curves, dtype=float)
    x_curves = np.asarray(x_curves, dtype=float)

    write_line_plot(
        output_dir / "Q2_c_bell_family_z_basis_probabilities.png",
        xs,
        [
            ("|00>", z_curves[:, 0], _COLORS_4[0]),
            ("|01>", z_curves[:, 1], _COLORS_4[1]),
            ("|10>", z_curves[:, 2], _COLORS_4[2]),
            ("|11>", z_curves[:, 3], _COLORS_4[3]),
        ],
        title="Q2 Bell-like family in the Z basis",
        y_min=0.0,
        y_max=1.0,
        y_label="probability",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_c_bell_family_z_basis_probabilities.png"
    )
    write_line_plot(
        output_dir / "Q2_c_bell_family_x_basis_probabilities.png",
        xs,
        [
            (label, x_curves[:, idx], _COLORS_4[idx])
            for idx, label in enumerate(_BELL_X_LABELS)
        ],
        title="Q2 Bell-like family in the X basis",
        y_min=0.0,
        y_max=1.0,
        y_label="probability",
    )
    maybe_publish_reference_artifact(
        output_dir / "Q2_c_bell_family_x_basis_probabilities.png"
    )
    make_gif_from_matplotlib_frames(
        output_dir / "Q2_c_bell_family_qsphere.gif",
        lambda idx: plot_state_qsphere(
            Statevector.from_instruction(
                require_impl(
                    Q2.build_bell_family_circuit, float(np.linspace(0.0, 1.0, 21)[idx])
                )
            )
        ),
        n_frames=21,
        delay_cs=12,
    )
    maybe_publish_reference_artifact(output_dir / "Q2_c_bell_family_qsphere.gif")


def test_noisy_simulation_on_bell_family_generates_comparison_plot():
    info = load_student_info()
    xs = np.linspace(0.0, 1.0, 41)
    output_dir = ensure_output_dir()

    ideal_p00 = []
    noisy_p00 = []
    total_variation = []

    for x in xs:
        qc = require_impl(Q2.build_bell_family_circuit, float(x))
        ideal = require_impl(Q2.basis_probabilities_from_circuit, qc, "Z")
        noisy = require_impl(
            Q2.noisy_basis_probabilities,
            qc,
            "Z",
            0.04,
            4096,
            info.seed,
        )

        assert noisy.shape == (
            4,
        ), "noisy_basis_probabilities() must return a four-entry probability vector for two qubits."
        assert np.isclose(
            np.sum(noisy), 1.0, atol=5e-3
        ), "Noisy probability vector should still sum to approximately one."

        ideal_p00.append(float(ideal[0]))
        noisy_p00.append(float(noisy[0]))
        total_variation.append(0.5 * float(np.sum(np.abs(ideal - noisy))))

    ideal_p00 = np.asarray(ideal_p00, dtype=float)
    noisy_p00 = np.asarray(noisy_p00, dtype=float)
    total_variation = np.asarray(total_variation, dtype=float)

    assert (
        float(np.mean(total_variation)) > 0.01
    ), "Noisy simulation should differ measurably from the ideal distribution on average."

    write_line_plot(
        output_dir / "Q2_d_bell_family_noise_p00.png",
        xs,
        [
            ("ideal P(|00>)", ideal_p00, _COLORS_4[0]),
            ("noisy P(|00>)", noisy_p00, _COLORS_4[1]),
        ],
        title="Q2 noiseless vs noisy simulation on the Bell-like family",
        y_min=0.0,
        y_max=1.0,
        y_label="probability",
    )
    write_line_plot(
        output_dir / "Q2_d_bell_family_noise_total_variation.png",
        xs,
        [("TV distance", total_variation, _COLORS_4[2])],
        title="Q2 total variation distance for the Bell-like family",
        y_min=0.0,
        y_max=max(0.2, float(np.max(total_variation)) + 0.02),
        y_label="distance",
    )
