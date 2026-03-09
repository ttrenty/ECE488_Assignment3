import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit_machine_learning.neural_networks import EstimatorQNN

from tests_utils import (
    ensure_output_dir,
    import_impl,
    load_student_info,
    require_impl,
    save_quantum_circuit_image,
    write_classification_prediction_plot,
)

Q3 = import_impl("Q3_vqc_from_scratch")


def _assert_history_dict(history: dict[str, object], epochs: int) -> None:
    required = {
        "train_loss_history",
        "val_loss_history",
        "train_accuracy_history",
        "val_accuracy_history",
    }
    assert required.issubset(
        history
    ), "History dictionary is missing one or more required keys."
    for key in required:
        values = np.asarray(history[key], dtype=float)
        assert values.shape == (
            epochs,
        ), f"History entry '{key}' must contain exactly {epochs} values."
        assert np.isfinite(
            values
        ).all(), f"History entry '{key}' must contain only finite numeric values."


def _write_training_curves_plot(
    path: str | Path,
    xs: np.ndarray,
    loss_series: list[tuple[str, np.ndarray, str]],
    accuracy_series: list[tuple[str, np.ndarray, str]],
    title: str,
    x_label: str,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray(xs, dtype=float).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    for label, values, color in loss_series:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != xs.shape:
            raise ValueError("loss series must have the same shape as xs")
        axes[0].plot(xs, values, label=label, color=color, linewidth=2.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.35)

    for label, values, color in accuracy_series:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != xs.shape:
            raise ValueError("accuracy series must have the same shape as xs")
        axes[1].plot(xs, values, label=label, color=color, linewidth=2.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.35)

    handles, labels = axes[1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82, top=0.83)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_history_training_curves_plot(
    path: str | Path,
    history: dict[str, object],
    title: str,
) -> None:
    train_losses = np.asarray(history["train_loss_history"], dtype=float)
    val_losses = np.asarray(history["val_loss_history"], dtype=float)
    train_acc = np.asarray(history["train_accuracy_history"], dtype=float)
    val_acc = np.asarray(history["val_accuracy_history"], dtype=float)
    xs = np.arange(1, train_losses.size + 1, dtype=float)
    _write_training_curves_plot(
        path,
        xs,
        [
            ("train loss", train_losses, "#1f4e79"),
            ("validation loss", val_losses, "#c44536"),
        ],
        [
            ("train accuracy", train_acc, "#1f4e79"),
            ("validation accuracy", val_acc, "#c44536"),
        ],
        title=title,
        x_label="epoch",
    )


def _write_reuploading_validation_sweep_plot(
    path: str | Path,
    reuploadings: list[int],
    per_model_scores: dict[str, list[float]],
    best_scores: list[float],
) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray(reuploadings, dtype=int)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = {
        "manual": "#1f4e79",
        "polynomial": "#c44536",
        "prebuilt": "#2d6a4f",
        "best": "#111111",
    }

    for model_name in ("manual", "polynomial", "prebuilt"):
        ys = np.asarray(per_model_scores[model_name], dtype=float)
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            color=colors[model_name],
            label=model_name,
        )

    ax.plot(
        xs,
        np.asarray(best_scores, dtype=float),
        marker="D",
        linestyle="--",
        linewidth=2.2,
        color=colors["best"],
        label="best validation",
    )
    ax.set_title("Q3.d validation accuracy vs re-uploading")
    ax.set_xlabel("re_uploading")
    ax.set_ylabel("validation accuracy")
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _update_accuracy_summary(section: str, payload: dict[str, object]) -> Path:
    output_dir = ensure_output_dir()
    summary_filenames = {
        "model_selection": "Q3_d_model_selection_summary.json",
        "model_selection_reuploading_sweep": "Q3_d_reuploading_validation_sweep.json",
        "torch_hybrid": "Q3_e_torch_hybrid_summary.json",
        "noisy_optimizers": "Q3_f_noisy_optimizer_summary.json",
    }
    summary_path = output_dir / summary_filenames[section]
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary_path


def test_dataset_loader_returns_train_validation_test_splits():
    info = load_student_info()
    splits = require_impl(Q3.load_breast_cancer_pca_splits, info.seed, 90, 2)

    required = {
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
    }
    assert required.issubset(
        splits
    ), "Dataset split dictionary is missing one or more required keys."

    total = (
        splits["X_train"].shape[0]
        + splits["X_val"].shape[0]
        + splits["X_test"].shape[0]
    )
    assert (
        total == 90
    ), "load_breast_cancer_pca_splits() should keep the requested total sample count."
    assert (
        splits["X_train"].shape[1] == 2
    ), "Training inputs should contain exactly two PCA features."
    assert (
        splits["X_val"].shape[1] == 2
    ), "Validation inputs should contain exactly two PCA features."
    assert (
        splits["X_test"].shape[1] == 2
    ), "Test inputs should contain exactly two PCA features."
    assert set(np.unique(splits["y_train"])).issubset(
        {0, 1}
    ), "Training labels should use binary classes 0/1."
    assert np.all(
        np.abs(splits["X_train"]) <= 1.000001
    ), "Training features should be rescaled to approximately [-1, 1]."


def test_qnn_builders_have_expected_structure():
    degree = 2
    re_uploading = 2
    ansatz_layers = 1
    manual_qnn_shallow = require_impl(Q3.build_manual_reuploading_qnn, 1, 1, 2)
    manual_qnn = require_impl(
        Q3.build_manual_reuploading_qnn, re_uploading, ansatz_layers, 2
    )
    manual_qnn_three = require_impl(
        Q3.build_manual_reuploading_qnn, re_uploading, ansatz_layers, 3
    )
    poly_qnn = require_impl(
        Q3.build_polynomial_reuploading_qnn, degree, re_uploading, ansatz_layers, 2
    )
    poly_degree1_qnn = require_impl(
        Q3.build_polynomial_reuploading_qnn, 1, re_uploading, ansatz_layers, 2
    )
    poly_qnn_three = require_impl(
        Q3.build_polynomial_reuploading_qnn, degree, re_uploading, ansatz_layers, 3
    )
    prebuilt_qnn_shallow = require_impl(Q3.build_prebuilt_qiskit_qnn, 1, 1, 2)
    prebuilt_qnn = require_impl(
        Q3.build_prebuilt_qiskit_qnn, re_uploading, ansatz_layers, 2
    )
    prebuilt_qnn_deeper = require_impl(Q3.build_prebuilt_qiskit_qnn, re_uploading, 2, 2)
    prebuilt_qnn_three = require_impl(
        Q3.build_prebuilt_qiskit_qnn, re_uploading, ansatz_layers, 3
    )
    output_dir = ensure_output_dir()

    save_quantum_circuit_image(
        manual_qnn.circuit,
        output_dir / "Q3_b_manual_qnn_circuit.png",
        title="Q3.b manual QNN circuit",
    )
    save_quantum_circuit_image(
        poly_qnn.circuit,
        output_dir / "Q3_c_polynomial_qnn_circuit.png",
        title="Q3.c polynomial QNN circuit",
    )
    save_quantum_circuit_image(
        prebuilt_qnn.circuit,
        output_dir / "Q3_d_prebuilt_qnn_circuit.png",
        title="Q3.d prebuilt QNN circuit",
    )

    assert isinstance(
        manual_qnn_shallow, EstimatorQNN
    ), "Manual shallow builder must return an EstimatorQNN."
    assert isinstance(
        manual_qnn, EstimatorQNN
    ), "Manual builder must return an EstimatorQNN."
    assert isinstance(
        poly_qnn, EstimatorQNN
    ), "Polynomial builder must return an EstimatorQNN."
    assert isinstance(
        prebuilt_qnn_shallow, EstimatorQNN
    ), "Prebuilt shallow builder must return an EstimatorQNN."
    assert isinstance(
        prebuilt_qnn, EstimatorQNN
    ), "Prebuilt builder must return an EstimatorQNN."
    assert isinstance(
        prebuilt_qnn_deeper, EstimatorQNN
    ), "Deeper prebuilt builder must return an EstimatorQNN."

    assert (
        manual_qnn_shallow.num_inputs == 2
    ), "Manual shallow QNN should take exactly two inputs."
    assert manual_qnn.num_inputs == 2, "Manual QNN should take exactly two inputs."
    assert (
        manual_qnn_three.num_inputs == 3
    ), "Manual QNN should support three PCA inputs / qubits."
    assert poly_qnn.num_inputs == 2, "Polynomial QNN should take exactly two inputs."
    assert (
        poly_qnn_three.num_inputs == 3
    ), "Polynomial QNN should support three PCA inputs / qubits."
    assert (
        prebuilt_qnn_shallow.num_inputs == 2
    ), "Prebuilt shallow QNN should take exactly two inputs."
    assert prebuilt_qnn.num_inputs == 2, "Prebuilt QNN should take exactly two inputs."
    assert (
        prebuilt_qnn_deeper.num_inputs == 2
    ), "Deeper prebuilt QNN should take exactly two inputs."
    assert (
        prebuilt_qnn_three.num_inputs == 3
    ), "Prebuilt QNN should support three PCA inputs / qubits."

    expected_manual_shallow_weights = 4
    expected_manual_weights = 4 * ansatz_layers * re_uploading
    assert (
        manual_qnn_shallow.num_weights == expected_manual_shallow_weights
    ), "Manual shallow QNN should use the expected four trainable ansatz weights."
    assert (
        manual_qnn.num_weights == expected_manual_weights
    ), "Manual QNN weight count should scale as 4 * ansatz_layers * re_uploading."
    assert (
        manual_qnn_three.num_weights == 2 * 3 * ansatz_layers * re_uploading
    ), "Manual QNN weight count should generalize as 2 * n_qubits * ansatz_layers * re_uploading."

    polynomial_extra_weights = poly_qnn.num_weights - manual_qnn.num_weights
    shared_polynomial_coeffs = 2 * (degree + 1)
    per_block_polynomial_coeffs = 2 * (degree + 1) * re_uploading
    assert polynomial_extra_weights in {
        shared_polynomial_coeffs,
        per_block_polynomial_coeffs,
    }, (
        "Polynomial QNN should add trainable polynomial coefficients for the two input "
        "features, either shared across re-uploading blocks or repeated once per block."
    )

    degree1_extra_weights = poly_degree1_qnn.num_weights - manual_qnn.num_weights
    assert degree1_extra_weights in {
        2 * (1 + 1),
        2 * (1 + 1) * re_uploading,
    }, (
        "With the assignment convention, degree=1 should still implement an affine map "
        "a0 + a1*x for each feature, so it must add both constant and linear coefficients."
    )
    polynomial_extra_weights_three = (
        poly_qnn_three.num_weights - manual_qnn_three.num_weights
    )
    assert polynomial_extra_weights_three in {
        3 * (degree + 1),
        3 * (degree + 1) * re_uploading,
    }, "Polynomial QNN should scale its extra encoder coefficients with the number of PCA inputs / qubits."

    assert (
        prebuilt_qnn.num_weights > prebuilt_qnn_shallow.num_weights
    ), "Increasing pre-uploading should increase the prebuilt QNN weight count."
    assert (
        prebuilt_qnn_deeper.num_weights > prebuilt_qnn.num_weights
    ), "Increasing ansatz_layers should increase the prebuilt QNN weight count."


def test_cross_entropy_training_loop_returns_history_and_signal():
    info = load_student_info()
    splits = require_impl(Q3.load_breast_cancer_pca_splits, info.seed, 80, 2)
    qnn = require_impl(Q3.build_manual_reuploading_qnn, 2, 1, 2)

    weights, history = require_impl(
        Q3.train_qnn_with_cross_entropy,
        qnn,
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        info.seed,
        8,
        0.3,
        1e-2,
    )

    assert isinstance(weights, np.ndarray) and weights.shape == (
        qnn.num_weights,
    ), "train_qnn_with_cross_entropy() must return weights shaped like the QNN parameter vector."
    assert isinstance(
        history, dict
    ), "train_qnn_with_cross_entropy() must return a history dictionary as its second tuple element."
    _assert_history_dict(history, 8)
    assert (
        history["train_loss_history"][-1] <= history["train_loss_history"][0] + 1e-6
    ), "Training loss should not increase overall across the short reference run."

    preds = require_impl(Q3.predict_from_qnn, qnn, weights, splits["X_val"])
    val_acc = float(np.mean(preds == splits["y_val"]))
    assert (
        val_acc >= 0.5
    ), "Trained manual QNN should beat trivial chance-level validation accuracy on the binary task."


def test_model_selection_uses_validation_and_reports_locked_test_result():
    info = load_student_info()
    epochs = 20
    results = require_impl(
        Q3.run_q3_model_selection, info.seed, 1, 90, 3, 2, epochs, 0.2
    )

    assert set(results) >= {
        "manual",
        "polynomial",
        "prebuilt",
        "best_model",
        "best_test_accuracy",
    }, "run_q3_model_selection() is missing required top-level result keys."
    assert results["best_model"] in {
        "manual",
        "polynomial",
        "prebuilt",
    }, "best_model must name one of the three compared QNN families."

    model_payload = {}
    for model_name in ("manual", "polynomial", "prebuilt"):
        model_result = results[model_name]
        assert isinstance(
            model_result, dict
        ), f"Result entry for model '{model_name}' must be a dictionary."
        _assert_history_dict(model_result, epochs)
        assert (
            "train_accuracy" in model_result
        ), f"Model '{model_name}' result is missing train_accuracy."
        assert (
            "val_accuracy" in model_result
        ), f"Model '{model_name}' result is missing val_accuracy."
        train_acc = float(model_result["train_accuracy"])
        val_acc = float(model_result["val_accuracy"])
        assert (
            0.0 <= train_acc <= 1.0
        ), f"Model '{model_name}' train_accuracy must lie in [0, 1]."
        assert (
            0.0 <= val_acc <= 1.0
        ), f"Model '{model_name}' val_accuracy must lie in [0, 1]."
        model_payload[model_name] = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }

        _write_history_training_curves_plot(
            ensure_output_dir()
            / {
                "manual": "Q3_b_manual_training_curves.png",
                "polynomial": "Q3_c_polynomial_training_curves.png",
                "prebuilt": "Q3_d_prebuilt_training_curves.png",
            }[model_name],
            model_result,
            title=f"Q3 {model_name.title()} training curves",
        )

    val_scores = [
        model_payload[name]["val_accuracy"]
        for name in ("manual", "polynomial", "prebuilt")
    ]
    assert (
        max(val_scores) >= 0.7
    ), "At least one Q3 model family should reach a meaningful validation accuracy above a weak baseline."
    best_val_accuracy = max(val_scores)
    tied_best_models = {
        name
        for name in ("manual", "polynomial", "prebuilt")
        if np.isclose(
            model_payload[name]["val_accuracy"], best_val_accuracy, atol=1e-12
        )
    }
    assert (
        results["best_model"] in tied_best_models
    ), "best_model should correspond to one of the validation-best model families."
    assert (
        0.0 <= float(results["best_test_accuracy"]) <= 1.0
    ), "best_test_accuracy must lie in [0, 1]."
    X_test = np.asarray(results["X_test"], dtype=float)
    y_test = np.asarray(results["y_test"], dtype=int)
    best_test_predictions = np.asarray(results["best_test_predictions"], dtype=int)
    assert (
        X_test.ndim == 2 and X_test.shape[1] == 2
    ), "X_test should be a 2D array with two PCA features."
    assert y_test.shape == (
        X_test.shape[0],
    ), "y_test must align with the number of test samples."
    assert (
        best_test_predictions.shape == y_test.shape
    ), "best_test_predictions must align with y_test."
    assert (
        "decision_grid_x" in results
        and "decision_grid_y" in results
        and "decision_grid_predictions" in results
    ), "run_q3_model_selection() should return a decision grid for the 2D PCA prediction plot."
    write_classification_prediction_plot(
        ensure_output_dir() / "Q3_d_best_model_test_predictions.png",
        X_test,
        y_test,
        best_test_predictions,
        title="Q3 best selected model: test predictions",
        decision_grid_x=np.asarray(results.get("decision_grid_x"), dtype=float),
        decision_grid_y=np.asarray(results.get("decision_grid_y"), dtype=float),
        decision_grid_predictions=np.asarray(
            results.get("decision_grid_predictions"), dtype=int
        ),
    )

    summary_path = _update_accuracy_summary(
        "model_selection",
        {
            **model_payload,
            "best_model": str(results["best_model"]),
            "best_test_accuracy": float(results["best_test_accuracy"]),
        },
    )
    assert summary_path.exists(), "Q3 accuracy summary JSON should be written to disk."


@pytest.mark.slow
def test_model_selection_reuploading_sweep_reports_validation_trend():
    info = load_student_info()
    reuploadings = [1, 2, 3, 4]
    per_model_scores = {"manual": [], "polynomial": [], "prebuilt": []}
    best_scores: list[float] = []
    best_models: list[str] = []
    sweep_payload: dict[str, object] = {}

    for re_uploading in reuploadings:
        results = require_impl(
            Q3.run_q3_model_selection, info.seed, 1, 72, re_uploading, 2, 20, 0.2
        )
        run_key = f"re_uploading_{re_uploading}"
        sweep_payload[run_key] = {}

        current_best = -np.inf
        current_best_model = ""
        for model_name in ("manual", "polynomial", "prebuilt"):
            val_acc = float(results[model_name]["val_accuracy"])
            assert (
                0.0 <= val_acc <= 1.0
            ), f"Validation accuracy for model '{model_name}' at re_uploading={re_uploading} must lie in [0, 1]."
            per_model_scores[model_name].append(val_acc)
            sweep_payload[run_key][model_name] = {"val_accuracy": val_acc}
            if val_acc > current_best:
                current_best = val_acc
                current_best_model = model_name

        best_scores.append(float(current_best))
        best_models.append(current_best_model)
        sweep_payload[run_key]["best_model"] = str(results["best_model"])
        sweep_payload[run_key]["best_validation_accuracy"] = float(current_best)

    _write_reuploading_validation_sweep_plot(
        ensure_output_dir() / "Q3_d_reuploading_validation_sweep.png",
        reuploadings,
        per_model_scores,
        best_scores,
    )
    summary_path = _update_accuracy_summary(
        "model_selection_reuploading_sweep",
        {
            "reuploadings": reuploadings,
            "per_model_validation_accuracy": per_model_scores,
            "best_validation_accuracy": best_scores,
            "best_model_from_scores": best_models,
            **sweep_payload,
        },
    )
    assert (
        summary_path.exists()
    ), "Q3.d re-uploading sweep summary JSON should be written to disk."


def test_torch_hybrid_classifier_runs_end_to_end():
    if not importlib.util.find_spec("torch"):
        raise AssertionError("torch must be available in the qiskit environment for Q3")

    info = load_student_info()
    results = require_impl(
        Q3.train_torch_hybrid_classifier, info.seed, 90, 20, 5e-2, 3, 2
    )

    assert set(results) >= {
        "train_loss_history",
        "val_loss_history",
        "train_accuracy_history",
        "val_accuracy_history",
        "val_accuracy",
        "test_accuracy",
    }, "train_torch_hybrid_classifier() is missing required result keys."
    _assert_history_dict(results, 20)
    assert (
        0.0 <= float(results["val_accuracy"]) <= 1.0
    ), "Hybrid validation accuracy must lie in [0, 1]."
    assert (
        0.0 <= float(results["test_accuracy"]) <= 1.0
    ), "Hybrid test accuracy must lie in [0, 1]."
    assert (
        float(results["val_accuracy"]) >= 0.75
    ), "Hybrid model should reach a reasonable validation accuracy."
    X_test = np.asarray(results["X_test"], dtype=float)
    y_test = np.asarray(results["y_test"], dtype=int)
    test_predictions = np.asarray(results["test_predictions"], dtype=int)
    assert (
        X_test.ndim == 2 and X_test.shape[1] == 2
    ), "Hybrid X_test should be a 2D array with two PCA features."
    assert y_test.shape == (
        X_test.shape[0],
    ), "Hybrid y_test must align with the number of test samples."
    assert (
        test_predictions.shape == y_test.shape
    ), "Hybrid test_predictions must align with y_test."
    assert (
        "decision_grid_x" in results
        and "decision_grid_y" in results
        and "decision_grid_predictions" in results
    ), "train_torch_hybrid_classifier() should return a decision grid for the 2D PCA prediction plot."

    _write_history_training_curves_plot(
        ensure_output_dir() / "Q3_e_torch_hybrid_training_curves.png",
        results,
        title="Q3 Torch hybrid training curves",
    )
    write_classification_prediction_plot(
        ensure_output_dir() / "Q3_e_torch_hybrid_test_predictions.png",
        X_test,
        y_test,
        test_predictions,
        title="Q3 torch hybrid: test predictions",
        decision_grid_x=np.asarray(results.get("decision_grid_x"), dtype=float),
        decision_grid_y=np.asarray(results.get("decision_grid_y"), dtype=float),
        decision_grid_predictions=np.asarray(
            results.get("decision_grid_predictions"), dtype=int
        ),
    )
    summary_path = _update_accuracy_summary(
        "torch_hybrid",
        {
            "val_accuracy": float(results["val_accuracy"]),
            "test_accuracy": float(results["test_accuracy"]),
        },
    )
    assert (
        summary_path.exists()
    ), "Q3 accuracy summary JSON should be updated after the hybrid run."


def test_noisy_optimizer_comparison_runs_end_to_end():
    info = load_student_info()
    results = require_impl(
        Q3.compare_noisy_optimizers, info.seed, 60, 2, 20, 0.1, 256, 2
    )

    assert set(results) >= {
        "adam",
        "cobyla",
        "spsa",
        "best_optimizer",
        "best_test_accuracy",
    }, "compare_noisy_optimizers() is missing required top-level result keys."
    assert results["best_optimizer"] in {
        "adam",
        "cobyla",
        "spsa",
    }, "best_optimizer must name one of the compared optimizers."

    payload = {}
    for optimizer_name in ("adam", "cobyla", "spsa"):
        optimizer_result = results[optimizer_name]
        assert isinstance(
            optimizer_result, dict
        ), f"Result entry for optimizer '{optimizer_name}' must be a dictionary."
        _assert_history_dict(optimizer_result, 20)
        for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
            value = float(optimizer_result[key])
            assert (
                0.0 <= value <= 1.0
            ), f"Optimizer '{optimizer_name}' field '{key}' must lie in [0, 1]."
        payload[optimizer_name] = {
            "train_accuracy": float(optimizer_result["train_accuracy"]),
            "val_accuracy": float(optimizer_result["val_accuracy"]),
            "test_accuracy": float(optimizer_result["test_accuracy"]),
        }

    for history_key in (
        "train_loss_history",
        "val_loss_history",
        "train_accuracy_history",
        "val_accuracy_history",
    ):
        adam_initial = float(results["adam"][history_key][0])
        cobyla_initial = float(results["cobyla"][history_key][0])
        spsa_initial = float(results["spsa"][history_key][0])
        assert np.isclose(
            adam_initial, cobyla_initial, atol=1e-12
        ), f"All noisy optimizers should start from the same initial {history_key} value."
        assert np.isclose(
            adam_initial, spsa_initial, atol=1e-12
        ), f"All noisy optimizers should start from the same initial {history_key} value."

    best_val_accuracy = max(
        payload[name]["val_accuracy"] for name in ("adam", "cobyla", "spsa")
    )
    tied_best_optimizers = {
        name
        for name in ("adam", "cobyla", "spsa")
        if np.isclose(payload[name]["val_accuracy"], best_val_accuracy, atol=1e-12)
    }
    assert (
        results["best_optimizer"] in tied_best_optimizers
    ), "best_optimizer should correspond to one of the validation-best noisy optimizers."
    assert np.isclose(
        float(results["best_test_accuracy"]),
        payload[str(results["best_optimizer"])]["test_accuracy"],
        atol=1e-12,
    ), "best_test_accuracy should match the selected optimizer's reported test accuracy."
    X_test = np.asarray(results["X_test"], dtype=float)
    y_test = np.asarray(results["y_test"], dtype=int)
    best_test_predictions = np.asarray(results["best_test_predictions"], dtype=int)
    assert (
        X_test.ndim == 2 and X_test.shape[1] == 2
    ), "Noisy-comparison X_test should be a 2D array with two PCA features."
    assert y_test.shape == (
        X_test.shape[0],
    ), "Noisy-comparison y_test must align with the number of test samples."
    assert (
        best_test_predictions.shape == y_test.shape
    ), "Noisy-comparison best_test_predictions must align with y_test."
    assert (
        "decision_grid_x" in results
        and "decision_grid_y" in results
        and "decision_grid_predictions" in results
    ), "compare_noisy_optimizers() should return a decision grid for the 2D PCA prediction plot."

    xs = np.arange(
        1,
        len(np.asarray(results["adam"]["train_loss_history"], dtype=float)) + 1,
        dtype=float,
    )
    loss_curves = [
        (
            "Adam train loss",
            np.asarray(results["adam"]["train_loss_history"], dtype=float),
            "#1f4e79",
        ),
        (
            "Adam val loss",
            np.asarray(results["adam"]["val_loss_history"], dtype=float),
            "#5fa8d3",
        ),
        (
            "COBYLA train loss",
            np.asarray(results["cobyla"]["train_loss_history"], dtype=float),
            "#c44536",
        ),
        (
            "COBYLA val loss",
            np.asarray(results["cobyla"]["val_loss_history"], dtype=float),
            "#e07a5f",
        ),
        (
            "SPSA train loss",
            np.asarray(results["spsa"]["train_loss_history"], dtype=float),
            "#0b6e4f",
        ),
        (
            "SPSA val loss",
            np.asarray(results["spsa"]["val_loss_history"], dtype=float),
            "#6a994e",
        ),
    ]
    accuracy_curves = [
        (
            "Adam train accuracy",
            np.asarray(results["adam"]["train_accuracy_history"], dtype=float),
            "#1f4e79",
        ),
        (
            "Adam val accuracy",
            np.asarray(results["adam"]["val_accuracy_history"], dtype=float),
            "#5fa8d3",
        ),
        (
            "COBYLA train accuracy",
            np.asarray(results["cobyla"]["train_accuracy_history"], dtype=float),
            "#c44536",
        ),
        (
            "COBYLA val accuracy",
            np.asarray(results["cobyla"]["val_accuracy_history"], dtype=float),
            "#e07a5f",
        ),
        (
            "SPSA train accuracy",
            np.asarray(results["spsa"]["train_accuracy_history"], dtype=float),
            "#0b6e4f",
        ),
        (
            "SPSA val accuracy",
            np.asarray(results["spsa"]["val_accuracy_history"], dtype=float),
            "#6a994e",
        ),
    ]
    _write_training_curves_plot(
        ensure_output_dir() / "Q3_f_noisy_optimizer_training_curves.png",
        xs,
        loss_curves,
        accuracy_curves,
        title="Q3 Noisy optimizer training curves",
        x_label="iteration",
    )
    write_classification_prediction_plot(
        ensure_output_dir() / "Q3_f_noisy_optimizer_test_predictions.png",
        X_test,
        y_test,
        best_test_predictions,
        title="Q3 best noisy optimizer: test predictions",
        decision_grid_x=np.asarray(results.get("decision_grid_x"), dtype=float),
        decision_grid_y=np.asarray(results.get("decision_grid_y"), dtype=float),
        decision_grid_predictions=np.asarray(
            results.get("decision_grid_predictions"), dtype=int
        ),
    )

    best_val = max(payload[name]["val_accuracy"] for name in ("adam", "cobyla", "spsa"))
    assert (
        best_val >= 0.7
    ), "At least one noisy optimizer should reach a reasonable validation accuracy."
    assert (
        0.0 <= float(results["best_test_accuracy"]) <= 1.0
    ), "Noisy-comparison best_test_accuracy must lie in [0, 1]."

    summary_path = _update_accuracy_summary(
        "noisy_optimizers",
        {
            **payload,
            "best_optimizer": str(results["best_optimizer"]),
            "best_test_accuracy": float(results["best_test_accuracy"]),
        },
    )
    assert (
        summary_path.exists()
    ), "Q3 accuracy summary JSON should be updated after the noisy-optimizer comparison."
