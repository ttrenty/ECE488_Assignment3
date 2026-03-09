import json
from pathlib import Path
import warnings

import matplotlib
import numpy as np
import pytest
from qiskit.circuit import ParameterVector

from tests_utils import (
    ensure_output_dir,
    import_impl,
    load_student_info,
    require_impl,
    save_quantum_circuit_image,
    write_classification_prediction_plot,
    write_line_plot,
)

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message=r".*qiskit\.providers\.models is deprecated since Qiskit 1\.2.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The `QasmSimulator` backend will be deprecated.*",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The `StatevectorSimulator` backend will be deprecated.*",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*The `UnitarySimulator` backend will be deprecated.*",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=(
        r".*qiskit\.circuit\.instruction\.Instruction\.condition.*"
        r"deprecated as of qiskit 1\.3\.0.*"
    ),
    category=DeprecationWarning,
)

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:.*qiskit\.providers\.models is deprecated since Qiskit 1\.2.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*The `QasmSimulator` backend will be deprecated.*:PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*The `StatevectorSimulator` backend will be deprecated.*:PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*The `UnitarySimulator` backend will be deprecated.*:PendingDeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:.*qiskit\.circuit\.instruction\.Instruction\.condition.*deprecated as of qiskit 1\.3\.0.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:invalid escape sequence '\\[ls]':DeprecationWarning:squlearn\.kernel\.lowlevel_kernel\.regularization"
    ),
]

from squlearn.encoding_circuit import YZ_CX_EncodingCircuit

Q4 = import_impl("Q4_squlearn_models")


def _write_regression_training_curves_plot(
    path: str | Path,
    iterations: np.ndarray,
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    train_mae: np.ndarray,
    val_mae: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    iterations = np.asarray(iterations, dtype=float).reshape(-1)
    train_loss = np.asarray(train_loss, dtype=float).reshape(-1)
    val_loss = np.asarray(val_loss, dtype=float).reshape(-1)
    train_mae = np.asarray(train_mae, dtype=float).reshape(-1)
    val_mae = np.asarray(val_mae, dtype=float).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))

    axes[0].plot(
        iterations, train_loss, color="#1f4e79", linewidth=2.4, label="train loss"
    )
    axes[0].plot(
        iterations, val_loss, color="#c44536", linewidth=2.4, label="validation loss"
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("squared loss")
    axes[0].set_ylim(bottom=0.0)
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(
        iterations, train_mae, color="#1f4e79", linewidth=2.4, label="train MAE"
    )
    axes[1].plot(
        iterations, val_mae, color="#c44536", linewidth=2.4, label="validation MAE"
    )
    axes[1].set_title("MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean absolute error")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.35)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.82, top=0.84)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def test_load_wine_pca_splits():
    info = load_student_info()
    splits = require_impl(Q4.load_wine_pca_splits, info.seed, 150, 2)

    assert isinstance(splits, dict), "load_wine_pca_splits() must return a dictionary."
    for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert key in splits, f"Wine split dictionary is missing key '{key}'."

    assert (
        splits["X_train"].ndim == 2 and splits["X_train"].shape[1] == 2
    ), "Wine X_train should contain two PCA features."
    assert (
        splits["X_val"].ndim == 2 and splits["X_val"].shape[1] == 2
    ), "Wine X_val should contain two PCA features."
    assert (
        splits["X_test"].ndim == 2 and splits["X_test"].shape[1] == 2
    ), "Wine X_test should contain two PCA features."
    assert splits["y_train"].ndim == 1, "Wine y_train should be a 1D label array."
    assert splits["y_val"].ndim == 1, "Wine y_val should be a 1D label array."
    assert splits["y_test"].ndim == 1, "Wine y_test should be a 1D label array."
    assert (
        splits["X_train"].shape[0]
        + splits["X_val"].shape[0]
        + splits["X_test"].shape[0]
        == 150
    ), "Wine split sizes should sum to the requested sample count."
    assert set(
        np.unique(
            np.concatenate([splits["y_train"], splits["y_val"], splits["y_test"]])
        )
    ).issubset({0, 1, 2}), "Wine labels should stay within the three expected classes."
    assert np.all(
        np.abs(splits["X_train"]) <= 0.950001
    ), "Wine training features should be rescaled to approximately [-0.95, 0.95]."


def test_qsvc_model_selection_outputs_and_artifacts():
    info = load_student_info()

    splits = require_impl(Q4.load_wine_pca_splits, info.seed, 150, 2)

    assert isinstance(splits, dict), "load_wine_pca_splits() must return a dictionary."
    for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert key in splits, f"Wine split dictionary is missing key '{key}'."

    assert (
        splits["X_train"].ndim == 2 and splits["X_train"].shape[1] == 2
    ), "Wine X_train should contain two PCA features."
    assert (
        splits["X_val"].ndim == 2 and splits["X_val"].shape[1] == 2
    ), "Wine X_val should contain two PCA features."
    assert (
        splits["X_test"].ndim == 2 and splits["X_test"].shape[1] == 2
    ), "Wine X_test should contain two PCA features."
    assert splits["y_train"].ndim == 1, "Wine y_train should be a 1D label array."
    assert splits["y_val"].ndim == 1, "Wine y_val should be a 1D label array."
    assert splits["y_test"].ndim == 1, "Wine y_test should be a 1D label array."
    assert (
        splits["X_train"].shape[0]
        + splits["X_val"].shape[0]
        + splits["X_test"].shape[0]
        == 150
    ), "Wine split sizes should sum to the requested sample count."
    assert set(
        np.unique(
            np.concatenate([splits["y_train"], splits["y_val"], splits["y_test"]])
        )
    ).issubset({0, 1, 2}), "Wine labels should stay within the three expected classes."
    assert np.all(
        np.abs(splits["X_train"]) <= 0.950001
    ), "Wine training features should be rescaled to approximately [-0.95, 0.95]."

    results = require_impl(Q4.train_qsvc_models, splits, info.seed, (0.25, 1.0, 4.0))

    assert isinstance(results, dict), "train_qsvc_models() must return a dictionary."
    for key in (
        "c_values",
        "fidelity_train_accuracy",
        "fidelity_val_accuracy",
        "projected_train_accuracy",
        "projected_val_accuracy",
        "best_kernel",
        "best_c",
        "best_val_accuracy",
        "best_test_accuracy",
    ):
        assert key in results, f"QSVC results dictionary is missing key '{key}'."

    c_values = np.asarray(results["c_values"], dtype=float)
    fidelity_train = np.asarray(results["fidelity_train_accuracy"], dtype=float)
    fidelity_val = np.asarray(results["fidelity_val_accuracy"], dtype=float)
    projected_train = np.asarray(results["projected_train_accuracy"], dtype=float)
    projected_val = np.asarray(results["projected_val_accuracy"], dtype=float)

    assert c_values.shape == (
        3,
    ), "c_values should contain the three requested regularization values."
    assert np.allclose(
        c_values, np.array([0.25, 1.0, 4.0], dtype=float)
    ), "Returned c_values should preserve the candidate C values in order."
    assert (
        fidelity_train.shape == c_values.shape
    ), "Fidelity train-accuracy curve must align with c_values."
    assert (
        fidelity_val.shape == c_values.shape
    ), "Fidelity validation-accuracy curve must align with c_values."
    assert (
        projected_train.shape == c_values.shape
    ), "Projected-kernel train-accuracy curve must align with c_values."
    assert (
        projected_val.shape == c_values.shape
    ), "Projected-kernel validation-accuracy curve must align with c_values."
    assert np.isfinite(
        fidelity_train
    ).all(), "Fidelity train accuracies must be finite."
    assert np.isfinite(
        fidelity_val
    ).all(), "Fidelity validation accuracies must be finite."
    assert np.isfinite(
        projected_train
    ).all(), "Projected-kernel train accuracies must be finite."
    assert np.isfinite(
        projected_val
    ).all(), "Projected-kernel validation accuracies must be finite."
    assert np.all(
        (0.0 <= fidelity_train) & (fidelity_train <= 1.0)
    ), "Fidelity train accuracies must lie in [0, 1]."
    assert np.all(
        (0.0 <= fidelity_val) & (fidelity_val <= 1.0)
    ), "Fidelity validation accuracies must lie in [0, 1]."
    assert np.all(
        (0.0 <= projected_train) & (projected_train <= 1.0)
    ), "Projected-kernel train accuracies must lie in [0, 1]."
    assert np.all(
        (0.0 <= projected_val) & (projected_val <= 1.0)
    ), "Projected-kernel validation accuracies must lie in [0, 1]."

    best_kernel = str(results["best_kernel"])
    best_c = float(results["best_c"])
    best_val_accuracy = float(results["best_val_accuracy"])
    best_test_accuracy = float(results["best_test_accuracy"])
    X_test_plot = np.asarray(results["X_test"], dtype=float)
    y_test_plot = np.asarray(results["y_test"], dtype=int)
    test_predictions = np.asarray(results["test_predictions"], dtype=int)

    assert best_kernel in {
        "fidelity",
        "projected",
    }, "best_kernel must name one of the compared kernel families."
    assert best_c in set(
        c_values.tolist()
    ), "best_c must be one of the supplied C values."
    assert 0.0 <= best_val_accuracy <= 1.0, "best_val_accuracy must lie in [0, 1]."
    assert 0.0 <= best_test_accuracy <= 1.0, "best_test_accuracy must lie in [0, 1]."
    assert (
        best_test_accuracy >= 0.75
    ), "Selected QSVC model should reach a reasonable locked-test accuracy."

    if best_kernel == "fidelity":
        chosen_val_curve = fidelity_val
    else:
        chosen_val_curve = projected_val
    selected_index = int(np.where(np.isclose(c_values, best_c))[0][0])
    assert (
        abs(best_val_accuracy - float(chosen_val_curve[selected_index])) < 1e-10
    ), "best_val_accuracy should match the selected kernel/C validation score."
    assert (
        abs(best_val_accuracy - float(max(fidelity_val.max(), projected_val.max())))
        < 1e-10
    ), "best_val_accuracy should equal the best validation score among all tested QSVC settings."
    assert (
        X_test_plot.ndim == 2 and X_test_plot.shape[1] == 2
    ), "Returned X_test should be a 2D array with two PCA features."
    assert y_test_plot.shape == (
        X_test_plot.shape[0],
    ), "Returned y_test should align with the number of test samples."
    assert (
        test_predictions.shape == y_test_plot.shape
    ), "Returned test_predictions should align with y_test."
    assert (
        "decision_grid_x" in results
        and "decision_grid_y" in results
        and "decision_grid_predictions" in results
    ), "Selected kernel-model results should include a decision grid for the 2D PCA prediction plot."

    output_dir = ensure_output_dir()
    write_line_plot(
        output_dir / "Q4_c_kernel_accuracy_curves.png",
        c_values,
        [
            ("fidelity train", fidelity_train, "#1f4e79"),
            ("fidelity val", fidelity_val, "#4c9ed9"),
            ("projected train", projected_train, "#8c2f39"),
            ("projected val", projected_val, "#d77a61"),
        ],
        title="Q4 kernel comparison on wine validation splits",
        x_label="C regularization value",
        y_label="accuracy",
        y_min=0.0,
        y_max=1.02,
    )
    write_classification_prediction_plot(
        output_dir / "Q4_c_kernel_test_predictions.png",
        X_test_plot,
        y_test_plot,
        test_predictions,
        title="Q4 selected kernel model: test predictions",
        decision_grid_x=np.asarray(results["decision_grid_x"], dtype=float),
        decision_grid_y=np.asarray(results["decision_grid_y"], dtype=float),
        decision_grid_predictions=np.asarray(
            results["decision_grid_predictions"], dtype=int
        ),
    )

    summary = {
        "best_kernel": best_kernel,
        "best_c": best_c,
        "best_val_accuracy": best_val_accuracy,
        "best_test_accuracy": best_test_accuracy,
        "c_values": c_values.tolist(),
        "fidelity_train_accuracy": fidelity_train.tolist(),
        "fidelity_val_accuracy": fidelity_val.tolist(),
        "projected_train_accuracy": projected_train.tolist(),
        "projected_val_accuracy": projected_val.tolist(),
    }
    (output_dir / "Q4_c_kernel_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def test_load_linnerud_splits():
    info = load_student_info()
    splits = require_impl(Q4.load_linnerud_splits, info.seed)

    assert isinstance(splits, dict), "load_linnerud_splits() must return a dictionary."
    for key in ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"):
        assert key in splits, f"Linnerud split dictionary is missing key '{key}'."

    assert (
        splits["X_train"].shape[1] == 2
    ), "Linnerud X_train should contain two PCA features."
    assert (
        splits["X_val"].shape[1] == 2
    ), "Linnerud X_val should contain two PCA features."
    assert (
        splits["X_test"].shape[1] == 2
    ), "Linnerud X_test should contain two PCA features."
    assert (
        splits["Y_train"].shape[1] == 2
    ), "Linnerud Y_train should contain the two regression targets."
    assert (
        splits["Y_val"].shape[1] == 2
    ), "Linnerud Y_val should contain the two regression targets."
    assert (
        splits["Y_test"].shape[1] == 2
    ), "Linnerud Y_test should contain the two regression targets."
    assert np.all(
        np.abs(splits["X_train"]) <= 0.950001
    ), "Linnerud training features should be rescaled to approximately [-0.95, 0.95]."
    assert np.allclose(
        np.mean(splits["Y_train"], axis=0), 0.0, atol=1e-7
    ), "Linnerud training targets should be centered after scaling."
    assert np.allclose(
        np.std(splits["Y_train"], axis=0), 1.0, atol=1e-7
    ), "Linnerud training targets should have unit variance after scaling."


def test_shared_readout_qnn_outputs_and_artifacts():
    info = load_student_info()

    splits = require_impl(Q4.load_linnerud_splits, info.seed)

    assert isinstance(splits, dict), "load_linnerud_splits() must return a dictionary."
    for key in ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test"):
        assert key in splits, f"Linnerud split dictionary is missing key '{key}'."

    assert (
        splits["X_train"].shape[1] == 2
    ), "Linnerud X_train should contain two PCA features."
    assert (
        splits["X_val"].shape[1] == 2
    ), "Linnerud X_val should contain two PCA features."
    assert (
        splits["X_test"].shape[1] == 2
    ), "Linnerud X_test should contain two PCA features."
    assert (
        splits["Y_train"].shape[1] == 2
    ), "Linnerud Y_train should contain the two regression targets."
    assert (
        splits["Y_val"].shape[1] == 2
    ), "Linnerud Y_val should contain the two regression targets."
    assert (
        splits["Y_test"].shape[1] == 2
    ), "Linnerud Y_test should contain the two regression targets."
    assert np.all(
        np.abs(splits["X_train"]) <= 0.950001
    ), "Linnerud training features should be rescaled to approximately [-0.95, 0.95]."
    assert np.allclose(
        np.mean(splits["Y_train"], axis=0), 0.0, atol=1e-7
    ), "Linnerud training targets should be centered after scaling."
    assert np.allclose(
        np.std(splits["Y_train"], axis=0), 1.0, atol=1e-7
    ), "Linnerud training targets should have unit variance after scaling."

    results = require_impl(
        Q4.train_shared_readout_qnn,
        splits,
        info.seed,
        (1, 2, 3),
        20,
        n_features=2,
    )

    assert isinstance(
        results, dict
    ), "train_shared_readout_qnn() must return a dictionary."
    for key in (
        "candidate_layers",
        "available_features",
        "n_features_used",
        "train_mean_mae",
        "val_mean_mae",
        "best_iterations",
        "best_train_loss_history",
        "best_val_loss_history",
        "best_train_mae_history",
        "best_val_mae_history",
        "best_layers",
        "best_val_mae",
        "test_mae_per_output",
        "test_mean_mae",
        "test_predictions",
        "Y_test",
        "prediction_correlation",
    ):
        assert (
            key in results
        ), f"Shared-readout QNN results dictionary is missing key '{key}'."

    candidate_layers = np.asarray(results["candidate_layers"], dtype=int)
    available_features = int(results["available_features"])
    n_features_used = int(results["n_features_used"])
    train_mean_mae = np.asarray(results["train_mean_mae"], dtype=float)
    val_mean_mae = np.asarray(results["val_mean_mae"], dtype=float)
    best_iterations = np.asarray(results["best_iterations"], dtype=int)
    best_train_loss = np.asarray(results["best_train_loss_history"], dtype=float)
    best_val_loss = np.asarray(results["best_val_loss_history"], dtype=float)
    best_train_mae = np.asarray(results["best_train_mae_history"], dtype=float)
    best_val_mae_history = np.asarray(results["best_val_mae_history"], dtype=float)
    test_mae_per_output = np.asarray(results["test_mae_per_output"], dtype=float)
    test_predictions = np.asarray(results["test_predictions"], dtype=float)
    y_test = np.asarray(results["Y_test"], dtype=float)
    prediction_correlation = np.asarray(results["prediction_correlation"], dtype=float)

    assert candidate_layers.shape == (
        3,
    ), "candidate_layers should contain the three requested depth options."
    assert (
        available_features >= 2
    ), "available_features should report the PCA width available in the regression splits."
    assert (
        n_features_used == 2
    ), "n_features_used should match the requested retained PCA-feature count."
    assert (
        train_mean_mae.shape == candidate_layers.shape
    ), "train_mean_mae must align with candidate_layers."
    assert (
        val_mean_mae.shape == candidate_layers.shape
    ), "val_mean_mae must align with candidate_layers."
    assert np.isfinite(train_mean_mae).all(), "Training MAE values must be finite."
    assert np.isfinite(val_mean_mae).all(), "Validation MAE values must be finite."
    assert (
        best_iterations.ndim == 1 and best_iterations.size >= 1
    ), "best_iterations should record at least one optimization step for the selected depth."
    assert (
        int(best_iterations[0]) == 0
    ), "best_iterations should start at epoch 0 for the initial untrained model."
    assert (
        best_train_loss.shape == best_iterations.shape
    ), "best_train_loss_history must align with best_iterations."
    assert (
        best_val_loss.shape == best_iterations.shape
    ), "best_val_loss_history must align with best_iterations."
    assert (
        best_train_mae.shape == best_iterations.shape
    ), "best_train_mae_history must align with best_iterations."
    assert (
        best_val_mae_history.shape == best_iterations.shape
    ), "best_val_mae_history must align with best_iterations."
    assert np.isfinite(
        best_train_loss
    ).all(), "best_train_loss_history must contain only finite values."
    assert np.isfinite(
        best_val_loss
    ).all(), "best_val_loss_history must contain only finite values."
    assert np.isfinite(
        best_train_mae
    ).all(), "best_train_mae_history must contain only finite values."
    assert np.isfinite(
        best_val_mae_history
    ).all(), "best_val_mae_history must contain only finite values."

    best_layers = int(results["best_layers"])
    best_val_mae = float(results["best_val_mae"])
    test_mean_mae = float(results["test_mean_mae"])

    assert best_layers in set(
        candidate_layers.tolist()
    ), "best_layers must be one of the candidate depth values."
    best_index = int(np.where(candidate_layers == best_layers)[0][0])
    assert (
        abs(best_val_mae - float(val_mean_mae[best_index])) < 1e-10
    ), "best_val_mae should match the selected depth's validation MAE."
    assert (
        abs(best_val_mae - float(val_mean_mae.min())) < 1e-10
    ), "best_val_mae should equal the minimum validation MAE among the tested depths."

    assert test_mae_per_output.shape == (
        2,
    ), "test_mae_per_output should report one MAE per regression target."
    assert np.isfinite(
        test_mae_per_output
    ).all(), "Per-output test MAE values must be finite."
    assert (
        np.max(test_mae_per_output) < 1.5
    ), "Shared-readout QNN test MAE should stay within a reasonable range."
    assert (
        abs(test_mean_mae - float(np.mean(test_mae_per_output))) < 1e-10
    ), "test_mean_mae should equal the mean of test_mae_per_output."

    assert (
        test_predictions.shape == y_test.shape
    ), "test_predictions must align with Y_test."
    assert y_test.shape[1] == 2, "Y_test should contain exactly two regression targets."
    assert np.isfinite(
        test_predictions
    ).all(), "Predicted regression outputs must be finite."

    assert prediction_correlation.shape == (
        2,
        2,
    ), "prediction_correlation should be a 2x2 matrix for the two outputs."
    assert np.isfinite(
        prediction_correlation
    ).all(), "prediction_correlation must contain only finite values."
    assert np.all(
        np.abs(prediction_correlation) <= 1.0 + 1e-8
    ), "Correlation coefficients should stay within [-1, 1]."

    output_dir = ensure_output_dir()
    if hasattr(Q4, "build_manual_reuploading_encoding_circuit"):
        for n_qubits in (2, 3):
            encoding = Q4.build_manual_reuploading_encoding_circuit(
                re_uploading=3,
                ansatz_layers=1,
                n_qubits=n_qubits,
            )
            features = ParameterVector("x", n_qubits)
            params = ParameterVector("p", int(encoding.num_parameters))
            circuit = encoding.get_circuit(features, params)
            save_quantum_circuit_image(
                circuit,
                output_dir / f"Q4_e_shared_qnn_circuit_{n_qubits}qubits.png",
                title=f"Q4 shared-readout encoder ({n_qubits} qubits)",
            )
    else:
        for n_qubits in (2, 3):
            encoding = YZ_CX_EncodingCircuit(num_qubits=n_qubits, num_layers=1)
            features = ParameterVector("x", n_qubits)
            params = ParameterVector("p", int(encoding.num_parameters))
            circuit = encoding.get_circuit(features, params)
            save_quantum_circuit_image(
                circuit,
                output_dir / f"Q4_e_shared_qnn_circuit_{n_qubits}qubits.png",
                title=f"Q4 shared-readout encoder ({n_qubits} qubits)",
            )
    write_line_plot(
        output_dir / "Q4_e_shared_qnn_layer_mae.png",
        candidate_layers.astype(float),
        [
            ("train mean MAE", train_mean_mae, "#1f4e79"),
            ("val mean MAE", val_mean_mae, "#c44536"),
        ],
        title="Q4 shared-readout QNN depth comparison",
        x_label="number of encoding layers",
        y_label="mean MAE (scaled targets)",
        y_min=0.0,
    )
    _write_regression_training_curves_plot(
        output_dir / "Q4_e_shared_qnn_training_curves.png",
        best_iterations,
        best_train_loss,
        best_val_loss,
        best_train_mae,
        best_val_mae_history,
        title="Q4 shared-readout QNN optimization curves",
    )

    import matplotlib.pyplot as plt

    sample_index = np.arange(y_test.shape[0], dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    labels = ["Weight (scaled)", "Waist (scaled)"]
    colors = [("#1f4e79", "#7fb3d5"), ("#8c2f39", "#d77a61")]
    for output_idx, ax in enumerate(np.atleast_1d(axes)):
        truth_color, pred_color = colors[output_idx]
        ax.plot(
            sample_index,
            y_test[:, output_idx],
            marker="o",
            linewidth=2.2,
            color=truth_color,
            label="target",
        )
        ax.plot(
            sample_index,
            test_predictions[:, output_idx],
            marker="s",
            linewidth=2.2,
            color=pred_color,
            label="prediction",
        )
        ax.set_ylabel(labels[output_idx])
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("test sample index")
    fig.suptitle("Q4 shared-readout QNN test predictions")
    fig.tight_layout()
    fig.savefig(
        output_dir / "Q4_e_shared_qnn_test_predictions.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "available_features": available_features,
        "n_features_used": n_features_used,
        "candidate_layers": candidate_layers.tolist(),
        "train_mean_mae": train_mean_mae.tolist(),
        "val_mean_mae": val_mean_mae.tolist(),
        "best_iterations": best_iterations.tolist(),
        "best_train_loss_history": best_train_loss.tolist(),
        "best_val_loss_history": best_val_loss.tolist(),
        "best_train_mae_history": best_train_mae.tolist(),
        "best_val_mae_history": best_val_mae_history.tolist(),
        "best_layers": best_layers,
        "best_val_mae": best_val_mae,
        "test_mae_per_output": test_mae_per_output.tolist(),
        "test_mean_mae": test_mean_mae,
        "prediction_correlation": prediction_correlation.tolist(),
    }
    (output_dir / "Q4_e_shared_qnn_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def test_shared_readout_qnn_feature_sweep_reports_feature_tradeoff():
    info = load_student_info()
    splits = require_impl(Q4.load_linnerud_splits, info.seed, 3)

    feature_counts = np.array([2, 3], dtype=int)
    train_mae_curve: list[float] = []
    val_mae_curve: list[float] = []

    for n_features in feature_counts:
        results = require_impl(
            Q4.train_shared_readout_qnn,
            splits,
            info.seed,
            (1,),
            8,
            n_features=int(n_features),
        )
        assert int(results["n_features_used"]) == int(
            n_features
        ), "Feature-sweep runs should report the exact retained PCA-feature count used."
        train_mae_curve.append(float(results["best_train_mae"]))
        val_mae_curve.append(float(results["best_val_mae"]))

    train_mae_curve_np = np.asarray(train_mae_curve, dtype=float)
    val_mae_curve_np = np.asarray(val_mae_curve, dtype=float)
    assert np.isfinite(
        train_mae_curve_np
    ).all(), "Feature-sweep training MAE values must be finite."
    assert np.isfinite(
        val_mae_curve_np
    ).all(), "Feature-sweep validation MAE values must be finite."

    output_dir = ensure_output_dir()
    write_line_plot(
        output_dir / "Q4_e_shared_qnn_feature_comparison.png",
        feature_counts.astype(float),
        [
            ("train mean MAE", train_mae_curve_np, "#1f4e79"),
            ("validation mean MAE", val_mae_curve_np, "#c44536"),
        ],
        title="Q4 shared-readout QNN vs retained PCA features",
        x_label="number of retained PCA features / qubits",
        y_label="mean MAE (scaled targets)",
        y_min=0.0,
    )
    (output_dir / "Q4_e_shared_qnn_feature_comparison.json").write_text(
        json.dumps(
            {
                "feature_counts": feature_counts.tolist(),
                "train_mean_mae": train_mae_curve,
                "validation_mean_mae": val_mae_curve,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
