"""
Q4: Use sQUlearn abstractions on two small real datasets.

Workflow:
- multiclass classification on the sklearn wine dataset
- multi-output regression on the sklearn linnerud dataset
- systematic train / validation / test splitting
- model selection only on validation data
- final reporting only on the locked test split of the selected model
"""

from __future__ import annotations

import numpy as np

# sklearn provides the real datasets and the preprocessing steps used before the
# quantum models see the features.
from sklearn.datasets import load_linnerud, load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Executor selects the simulation backend used by sQUlearn.
from squlearn import Executor

# Prebuilt / layered encoding circuits:
# - HubregtsenEncodingCircuit is a compact feature map convenient for kernel methods.
# - LayeredEncodingCircuit lets us build a manual re-uploading circuit for regression.
from squlearn.encoding_circuit import HubregtsenEncodingCircuit, LayeredEncodingCircuit
from squlearn.encoding_circuit.layered_encoding_circuit import Layer

# Quantum kernels and their classifier wrapper used in the multiclass comparison.
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel, QSVC

# SinglePauli lets you define one readout observable per qubit/output.
from squlearn.observables import SinglePauli

# Adam is the optimizer used to fit the QNN regressor.
from squlearn.optimizers import Adam

# QNNRegressor is the multi-output regression model, and SquaredLoss is the
# regression loss used by that model.
from squlearn.qnn import QNNRegressor, SquaredLoss


def load_wine_pca_splits(
    seed: int,
    n_samples: int = 150,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """
    Load a 3-class dataset, then return train/validation/test splits after:
    - standardizing the original features
    - reducing them to n_components principal components
    - rescaling the retained components to [-0.95, 0.95]

    Use sklearn's wine dataset. The original dataset has 13 chemistry features
    and 3 grape cultivars. We reduce it to 2 features so the kernel circuit
    stays small enough for the assignment.
    Use the same ratio of train/validation/test splits as in the breast-cancer part, and use
    stratification to preserve class balance in every split.

    Hint: fit StandardScaler(), PCA(), and the final MinMaxScaler() on the
    training split only, then reuse those fitted transforms on validation and
    test to avoid data leakage.

    Expected return contract:
    - return exactly one dictionary `splits`
    - `splits["X_train"]`: float array with shape `(n_train, n_components)`
    - `splits["X_val"]`: float array with shape `(n_val, n_components)`
    - `splits["X_test"]`: float array with shape `(n_test, n_components)`
    - `splits["y_train"]`: int array with shape `(n_train,)`
    - `splits["y_val"]`: int array with shape `(n_val,)`
    - `splits["y_test"]`: int array with shape `(n_test,)`

    Example layout:
    splits = {
        "X_train": ...,  # np.ndarray of floats, shape (n_train, n_components)
        "X_val": ...,    # np.ndarray of floats, shape (n_val, n_components)
        "X_test": ...,   # np.ndarray of floats, shape (n_test, n_components)
        "y_train": ...,  # np.ndarray of ints, shape (n_train,)
        "y_val": ...,    # np.ndarray of ints, shape (n_val,)
        "y_test": ...,   # np.ndarray of ints, shape (n_test,)
    }
    """

    raise NotImplementedError("load_wine_pca_splits not implemented")


def _require_split_keys(splits: dict[str, np.ndarray], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in splits]
    if missing:
        raise KeyError(f"missing split keys: {missing}")


def _shared_executor(seed: int) -> Executor:
    return Executor("statevector_simulator", seed=int(seed))


def _make_decision_plane_grid(
    X_reference: np.ndarray,
    resolution: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_reference = np.asarray(X_reference, dtype=float)
    if X_reference.ndim != 2 or X_reference.shape[1] != 2:
        raise ValueError("decision-plane grid requires 2D reference inputs")

    x_margin = 0.1 * max(1e-6, float(np.ptp(X_reference[:, 0])))
    y_margin = 0.1 * max(1e-6, float(np.ptp(X_reference[:, 1])))
    x_values = np.linspace(
        float(np.min(X_reference[:, 0]) - x_margin),
        float(np.max(X_reference[:, 0]) + x_margin),
        int(resolution),
    )
    y_values = np.linspace(
        float(np.min(X_reference[:, 1]) - y_margin),
        float(np.max(X_reference[:, 1]) + y_margin),
        int(resolution),
    )
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid_points


def train_qsvc_models(
    splits: dict[str, np.ndarray],
    seed: int,
    c_values: tuple[float, ...] = (0.25, 1.0, 4.0),
) -> dict[str, object]:
    """
    Train two QSVC families on the wine splits:
    - FidelityKernel
    - ProjectedQuantumKernel

    Use the same encoding circuit for both kernel families, evaluate every
    regularization value C from c_values on train and validation data, select
    the single best (kernel, C) pair from validation accuracy only, then refit
    that selected pair on train+validation before evaluating once on test.

    Hint: smaller C usually regularizes the classifier more strongly. If train
    accuracy is much higher than validation accuracy, that is a sign worth
    discussing in your written answer.

    Expected return contract:
    - return exactly one dictionary `results`
    - `results["c_values"]`: list[float] in the same order as the evaluated C values
    - `results["fidelity_train_accuracy"]`: list[float] with one entry per value in `c_values`
    - `results["fidelity_val_accuracy"]`: list[float] with one entry per value in `c_values`
    - `results["projected_train_accuracy"]`: list[float] with one entry per value in `c_values`
    - `results["projected_val_accuracy"]`: list[float] with one entry per value in `c_values`
    - `results["best_kernel"]`: string equal to `"fidelity"` or `"projected"`
    - `results["best_c"]`: selected regularization value as a float
    - `results["best_val_accuracy"]`: float in [0, 1]
    - `results["best_test_accuracy"]`: float in [0, 1]
    - `results["X_test"]`: float array with shape `(n_test, 2)`
    - `results["y_test"]`: int array with shape `(n_test,)`
    - `results["test_predictions"]`: int array with shape `(n_test,)`
    - also return `results["decision_grid_x"]`, `results["decision_grid_y"]`,
      and `results["decision_grid_predictions"]` so the tests can draw the
      2D decision regions behind the wine test samples
    """

    raise NotImplementedError("train_qsvc_models not implemented")
    _require_split_keys(...)

    X_train = np.asarray(splits["X_train"], dtype=float)
    X_val = np.asarray(splits["X_val"], dtype=float)
    X_test = np.asarray(splits["X_test"], dtype=float)
    y_train = np.asarray(splits["y_train"], dtype=int).reshape(-1)
    y_val = np.asarray(splits["y_val"], dtype=int).reshape(-1)
    y_test = np.asarray(splits["y_test"], dtype=int).reshape(-1)

    c_values = tuple(float(c) for c in c_values)
    if len(c_values) == 0:
        raise ValueError("c_values must contain at least one value")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    encoding = HubregtsenEncodingCircuit(num_qubits=2, num_layers=1)
    executor = _shared_executor(seed)
    kernels = {
        "fidelity": ...,
        "projected": ProjectedQuantumKernel(
            encoding_circuit=encoding,
            executor=executor,
            measurement="XYZ",
            outer_kernel="gaussian",
        ),
    }

    results: dict[str, object] = {
        "c_values": list(c_values),
        "fidelity_train_accuracy": [],
        "fidelity_val_accuracy": [],
        "projected_train_accuracy": [],
        "projected_val_accuracy": [],
    }
    best_kernel = ""
    best_c = ...
    best_val_accuracy = ...

    # Loop over every kernel family and regularization value, fit on train, evaluate on
    # train and validation, and track the best (kernel, C) pair by validation accuracy.
    for _ in []:  # <-- REPLACE with correct loop
        train_key = f"{kernel_name}_train_accuracy"
        val_key = f"{kernel_name}_val_accuracy"

        for _ in []:  # <-- REPLACE with correct loop
            model = QSVC(
                quantum_kernel=...,
                C=float(...),
                decision_function_shape="ovr",
            )
            model.fit(X_train, y_train)

            train_accuracy = float(accuracy_score(y_train, model.predict(X_train)))
            val_accuracy = float(accuracy_score(y_val, model.predict(X_val)))

            results[train_key].append(train_accuracy)
            results[val_key].append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_kernel = ...
                best_c = float(...)

    selected_model = QSVC(
        quantum_kernel=...,
        C=...,
        decision_function_shape="ovr",
    )
    # Refit the selected model on train+validation, evaluate once on test, and also
    ...
    test_predictions = ...
    best_test_accuracy = ...

    grid_x, grid_y, grid_points = _make_decision_plane_grid(
        np.vstack([X_train, X_val, X_test]),
        resolution=60,
    )
    decision_grid_predictions = np.asarray(
        selected_model.predict(grid_points),
        dtype=int,
    ).reshape(grid_x.shape)

    results.update(
        {
            "best_kernel": best_kernel,
            "best_c": best_c,
            "best_val_accuracy": float(best_val_accuracy),
            "best_test_accuracy": best_test_accuracy,
            "X_test": X_test.astype(float),
            "y_test": y_test.astype(int),
            "test_predictions": test_predictions.astype(int),
            "decision_grid_x": grid_x.astype(float),
            "decision_grid_y": grid_y.astype(float),
            "decision_grid_predictions": decision_grid_predictions.astype(int),
        }
    )
    return results


def load_linnerud_splits(
    seed: int,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """
    Load a compact real multi-output regression task from sklearn's linnerud
    dataset.

    Keep the first two physiological targets (Weight, Waist), but preprocess the
    three exercise inputs with StandardScaler(), PCA(n_components=n_components),
    and a final MinMaxScaler() to [-0.95, 0.95].
    Use a slightly larger test split than before so the regression plots contain
    more held-out samples.

    Hint: fit all preprocessing on the training split only, exactly as in the
    classification part.

    Expected return contract:
    - return exactly one dictionary `splits`
    - `splits["X_train"]`: float array with shape `(n_train, n_components)`
    - `splits["X_val"]`: float array with shape `(n_val, n_components)`
    - `splits["X_test"]`: float array with shape `(n_test, n_components)`
    - `splits["Y_train"]`: float array with shape `(n_train, 2)`
    - `splits["Y_val"]`: float array with shape `(n_val, 2)`
    - `splits["Y_test"]`: float array with shape `(n_test, 2)`

    Example layout:
    splits = {
        "X_train": ...,  # np.ndarray of floats, shape (n_train, n_components)
        "X_val": ...,    # np.ndarray of floats, shape (n_val, n_components)
        "X_test": ...,   # np.ndarray of floats, shape (n_test, n_components)
        "Y_train": ...,  # np.ndarray of floats, shape (n_train, 2)
        "Y_val": ...,    # np.ndarray of floats, shape (n_val, 2)
        "Y_test": ...,   # np.ndarray of floats, shape (n_test, 2)
    }
    """

    raise NotImplementedError("load_linnerud_splits not implemented")


def _add_manual_encoder_block(
    encoding_circuit: LayeredEncodingCircuit,
) -> None:
    """
    Add one data-encoding block to a layered sQUlearn circuit.

    This helper is provided so students can focus on the model-selection logic
    in `train_shared_readout_qnn()` instead of the low-level circuit API.
    """

    layer = Layer(encoding_circuit)
    layer.Ry("x")
    layer.Rz("x")

    if encoding_circuit.num_qubits >= 2:
        layer.cx_entangling("NN")

    encoding_circuit.add_layer(layer, num_layers=1)


def _add_manual_ansatz_block(
    encoding_circuit: LayeredEncodingCircuit,
    ansatz_layers: int,
) -> None:
    """
    Add one trainable ansatz block to a layered sQUlearn circuit.
    """

    layer = Layer(encoding_circuit)
    layer.Ry("p")
    layer.Rz("p")

    if encoding_circuit.num_qubits >= 2:
        layer.cx_entangling("NN")

    encoding_circuit.add_layer(layer, num_layers=int(ansatz_layers))


def build_manual_reuploading_encoding_circuit(
    re_uploading: int = 3,
    ansatz_layers: int = 1,
    n_qubits: int = 2,
    add_initial_hadamards: bool = False,
) -> LayeredEncodingCircuit:
    """
    Build the manual re-uploading encoding circuit used in Q4.e:

        [ encoder(x) -> ansatz(p) ] repeated `re_uploading` times

    where each encoder applies Ry/Rz feature rotations on every qubit followed
    by nearest-neighbor CX entanglement, and each ansatz block applies trainable
    Ry/Rz rotations followed by the same entanglement pattern.

    Expected return contract:
    - return exactly one `LayeredEncodingCircuit`
    - the circuit must have `n_qubits` qubits
    - the circuit must repeat the encoder/ansatz pattern `re_uploading` times
    """

    raise NotImplementedError(
        "build_manual_reuploading_encoding_circuit not implemented. Remove this "
        "line once you complete the scaffold below."
    )
    re_uploading = int(re_uploading)
    ansatz_layers = int(ansatz_layers)
    n_qubits = int(n_qubits)

    if re_uploading < 1:
        raise ValueError("re_uploading must be at least 1")
    if ansatz_layers < 1:
        raise ValueError("ansatz_layers must be at least 1")
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")

    # Start from an empty layered circuit with the requested number of qubits.
    encoding_circuit = ...

    # Optional: add an initial layer of Hadamards before the repeated
    # encoder/ansatz structure.
    if add_initial_hadamards:
        ...

    # Repeat one encoder block followed by one ansatz block `re_uploading`
    # times. We keep the reverse-entanglement flag in the loop so the structure
    # matches the reference implementation, even though the provided helpers use
    # the same NN entanglement pattern either way.
    for block in range(re_uploading):
        _add_manual_encoder_block(..., ...)
        _add_manual_ansatz_block(
            encoding_circuit,
            ansatz_layers=ansatz_layers,
        )

    # Force a one-time initialization using .generate_initial_parameters()so sQUlearn
    # already knows the parameter count before the regressor starts querying the layered circuit.
    ...

    return encoding_circuit


def train_shared_readout_qnn(
    splits: dict[str, np.ndarray],
    seed: int,
    candidate_layers: tuple[int, ...] = (1, 2, 3),
    epochs: int = 20,
    re_uploading: int = 3,
    n_features: int | None = None,
) -> dict[str, object]:
    """
    Train a shared-encoder QNN regressor that produces two outputs by reading
    two different qubits of the same n-qubit state.

    Use the provided manual re-uploading encoding-circuit family, compare the
    candidate ansatz depths on train and validation data, select the best depth
    from validation mean MAE only, then refit that selected depth on
    train+validation and evaluate once on test.

    Hint: `splits` may contain more PCA features than you want to keep. Use the
    first `n_features` principal components as the model inputs, and use the
    same number of qubits as retained features. Since there are two regression
    outputs, you need at least two retained features / qubits here.

    Expected return contract:
    - return exactly one dictionary `results`
    - `results["candidate_layers"]`: list[int] in the tested order
    - `results["re_uploading"]`: int equal to the number of encoder/ansatz
      blocks used in the selected circuit family
    - `results["available_features"]`: total number of PCA features available in
      `splits`
    - `results["n_features_used"]`: number of retained PCA features / qubits
      actually used by the model
    - `results["n_qubits"]`: int equal to `n_features_used`
    - `results["output_dim"]`: int equal to the number of regression outputs
    - `results["train_mean_mae"]`: list[float] with one value per candidate depth
    - `results["val_mean_mae"]`: list[float] with one value per candidate depth
    - `results["best_iterations"]`: list[int] for the iteration indices recorded
      while training the selected validation-best depth
    - `results["best_train_loss_history"]`: list[float] with one train loss per
      recorded iteration for that selected depth
    - `results["best_val_loss_history"]`: list[float] with one validation loss
      per recorded iteration for that selected depth
    - `results["best_train_mae_history"]`: list[float] with one train mean
      absolute error per recorded iteration for that selected depth
    - `results["best_val_mae_history"]`: list[float] with one validation mean
      absolute error per recorded iteration for that selected depth
    - `results["best_layers"]`: selected depth as an int
    - `results["best_train_mae"]`: float for the selected depth's training MAE
    - `results["best_val_mae"]`: float
    - `results["test_mae_per_output"]`: list[float] of length 2
    - `results["test_mean_mae"]`: float
    - `results["test_predictions"]`: float array with shape `(n_test, 2)`
    - `results["Y_test"]`: float array with shape `(n_test, 2)`
    - `results["prediction_correlation"]`: float array with shape `(2, 2)`
    """

    raise NotImplementedError(
        "train_shared_readout_qnn not implemented. Remove this line once you "
        "complete the scaffold below."
    )
    _require_split_keys(
        splits, ("X_train", "X_val", "X_test", "Y_train", "Y_val", "Y_test")
    )

    X_train_full = np.asarray(splits["X_train"], dtype=float)
    X_val_full = np.asarray(splits["X_val"], dtype=float)
    X_test_full = np.asarray(splits["X_test"], dtype=float)
    Y_train = np.asarray(splits["Y_train"], dtype=float)
    Y_val = np.asarray(splits["Y_val"], dtype=float)
    Y_test = np.asarray(splits["Y_test"], dtype=float)

    if X_train_full.ndim != 2 or X_val_full.ndim != 2 or X_test_full.ndim != 2:
        raise ValueError("all input splits must be 2D arrays")
    if Y_train.ndim != 2 or Y_val.ndim != 2 or Y_test.ndim != 2:
        raise ValueError("all target splits must be 2D arrays")

    available_features = int(X_train_full.shape[1])
    if available_features < 1:
        raise ValueError("the regression task expects at least 1 input feature")
    if (
        X_val_full.shape[1] != available_features
        or X_test_full.shape[1] != available_features
    ):
        raise ValueError("all input splits must have the same number of features")

    output_dim = int(Y_train.shape[1])
    if output_dim < 1:
        raise ValueError("the regression task expects at least 1 output")
    if Y_val.shape[1] != output_dim or Y_test.shape[1] != output_dim:
        raise ValueError("all target splits must have the same number of outputs")
    if n_features is None:
        n_features = available_features
    n_features = int(n_features)
    if not (1 <= n_features <= available_features):
        raise ValueError("n_features must stay within the available PCA dimension")
    if output_dim > n_features:
        raise ValueError(
            "the shared-readout QNN requires the number of outputs to be at most "
            "the number of qubits/features"
        )

    # Only uses the first n_features PCA components as the model inputs, and use the same
    X_train = X_train_full[:, :n_features]
    X_val = X_val_full[:, :n_features]
    X_test = X_test_full[:, :n_features]
    n_qubits = int(n_features)

    candidate_layers = tuple(int(layer) for layer in candidate_layers)
    if len(candidate_layers) == 0:
        raise ValueError("candidate_layers must contain at least one layer count")
    if min(candidate_layers) < 1:
        raise ValueError("candidate layer counts must be positive")

    epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be positive")

    re_uploading = int(re_uploading)
    if re_uploading < 1:
        raise ValueError("re_uploading must be positive")

    readout = [
        SinglePauli(num_qubits=n_qubits, qubit=qubit, op_str="Z")
        for qubit in range(output_dim)
    ]

    train_mean_mae: list[float] = []
    val_mean_mae: list[float] = []
    best_history = {
        "iterations": [],
        "train_loss_history": [],
        "val_loss_history": [],
        "train_mae_history": [],
        "val_mae_history": [],
    }

    best_layers = candidate_layers[0]
    best_train_mae = np.inf
    best_val_mae = np.inf

    # Evaluate every candidate ansatz depth on the fixed train/validation split.
    for layers in candidate_layers:
        # Build the shared encoding circuit for this candidate ansatz depth.
        encoding = ...
        optimizer = Adam(options={"maxiter": epochs, "tol": 1e-3})
        history = {
            "iterations": [],
            "train_loss_history": [],
            "val_loss_history": [],
            "train_mae_history": [],
            "val_mae_history": [],
        }

        def callback(iteration, x, gradient, fval) -> None:
            del gradient, fval
            params = np.asarray(x, dtype=float).reshape(-1)
            num_param = int(model._qnn.num_parameters)
            num_param_op = int(model._qnn.num_parameters_observable)
            param = params[:num_param]
            param_op = params[num_param : num_param + num_param_op]

            pred_train_iter = np.asarray(
                model._qnn.evaluate(X_train, param, param_op, "f")["f"],
                dtype=float,
            )
            pred_val_iter = np.asarray(
                model._qnn.evaluate(X_val, param, param_op, "f")["f"],
                dtype=float,
            )
            if pred_train_iter.ndim == 1:
                pred_train_iter = pred_train_iter.reshape(-1, 1)
            if pred_val_iter.ndim == 1:
                pred_val_iter = pred_val_iter.reshape(-1, 1)

            # The callback is called after an optimizer step, so shift the index
            # by +1 and separately record the untrained model at iteration 0.
            history["iterations"].append(int(iteration) + 1)
            history["train_loss_history"].append(
                float(np.mean((pred_train_iter - Y_train) ** 2))
            )
            history["val_loss_history"].append(
                float(np.mean((pred_val_iter - Y_val) ** 2))
            )
            history["train_mae_history"].append(
                float(np.mean(np.abs(pred_train_iter - Y_train)))
            )
            history["val_mae_history"].append(
                float(np.mean(np.abs(pred_val_iter - Y_val)))
            )

        # Create a sQULearn QNNRegressor, make sure to give the callback=callback argument so the training history gets recorded in the callback function above.
        model = QNNRegressor(...)

        # Query the low-level QNN once so we can record the epoch-0 metrics from
        # the randomly initialized parameters before any training happens.
        model._initialize_lowlevel_qnn(X_train.shape[1])
        initial_param = np.asarray(model._param, dtype=float)
        initial_param_op = np.asarray(model._param_op, dtype=float)
        initial_train = np.asarray(
            model._qnn.evaluate(X_train, initial_param, initial_param_op, "f")["f"],
            dtype=float,
        )
        initial_val = np.asarray(
            model._qnn.evaluate(X_val, initial_param, initial_param_op, "f")["f"],
            dtype=float,
        )
        if initial_train.ndim == 1:
            initial_train = initial_train.reshape(-1, 1)
        if initial_val.ndim == 1:
            initial_val = initial_val.reshape(-1, 1)
        history["iterations"].append(0)
        history["train_loss_history"].append(
            float(np.mean((initial_train - Y_train) ** 2))
        )
        history["val_loss_history"].append(float(np.mean((initial_val - Y_val) ** 2)))
        history["train_mae_history"].append(
            float(np.mean(np.abs(initial_train - Y_train)))
        )
        history["val_mae_history"].append(float(np.mean(np.abs(initial_val - Y_val))))
        model.fit(X_train, Y_train)

        # Evaluate the trained model on train and validation so you can compare
        # candidate depths with mean absolute error.
        pred_train = np.asarray(..., dtype=float)
        pred_val = np.asarray(..., dtype=float)

        if pred_train.ndim == 1:
            pred_train = pred_train.reshape(-1, 1)
        if pred_val.ndim == 1:
            pred_val = pred_val.reshape(-1, 1)

        layer_train_mae = np.mean(np.abs(pred_train - Y_train), axis=0)
        layer_val_mae = np.mean(np.abs(pred_val - Y_val), axis=0)
        mean_train_mae = float(np.mean(layer_train_mae))
        mean_val_mae = float(np.mean(layer_val_mae))

        train_mean_mae.append(mean_train_mae)
        val_mean_mae.append(mean_val_mae)

        # Keep the single best depth according to validation mean MAE only.
        if mean_val_mae < best_val_mae:
            best_train_mae = mean_train_mae
            best_val_mae = mean_val_mae
            best_layers = ...
            best_history = {key: list(values) for key, values in history.items()}

    # Refit the validation-best depth once on train+validation before touching
    # the locked test split.
    X_trainval = np.vstack([X_train, X_val])
    Y_trainval = np.vstack([Y_train, Y_val])

    final_model = QNNRegressor(
        build_manual_reuploading_encoding_circuit(
            re_uploading=...,
            ansatz_layers=...,
            n_qubits=...,
        ),
        operators,
        _shared_executor(seed),
        SquaredLoss(),
        Adam(options={"maxiter": epochs, "tol": 1e-3}),
        parameter_seed=int(seed),
        callback=None,
    )
    final_model.fit(X_trainval, Y_trainval)

    # Evaluate the final selected model once on the locked test split.
    test_predictions = np.asarray(..., dtype=float)
    if test_predictions.ndim == 1:
        test_predictions = test_predictions.reshape(-1, 1)
    test_mae_per_output = np.mean(np.abs(test_predictions - Y_test), axis=0)

    if test_predictions.shape[0] < 2 or test_predictions.shape[1] < 2:
        prediction_correlation = np.eye(test_predictions.shape[1], dtype=float)
    else:
        prediction_correlation = np.corrcoef(test_predictions, rowvar=False)
        prediction_correlation = np.asarray(prediction_correlation, dtype=float)
        prediction_correlation[~np.isfinite(prediction_correlation)] = 0.0

    return {
        "candidate_layers": list(candidate_layers),
        "re_uploading": int(re_uploading),
        "available_features": int(available_features),
        "n_features_used": int(n_features),
        "n_qubits": int(n_qubits),
        "output_dim": int(output_dim),
        "train_mean_mae": train_mean_mae,
        "val_mean_mae": val_mean_mae,
        "best_iterations": [int(step) for step in best_history["iterations"]],
        "best_train_loss_history": [
            float(value) for value in best_history["train_loss_history"]
        ],
        "best_val_loss_history": [
            float(value) for value in best_history["val_loss_history"]
        ],
        "best_train_mae_history": [
            float(value) for value in best_history["train_mae_history"]
        ],
        "best_val_mae_history": [
            float(value) for value in best_history["val_mae_history"]
        ],
        "best_layers": int(...),
        "best_train_mae": float(best_train_mae),
        "best_val_mae": float(...),
        "test_mae_per_output": test_mae_per_output.astype(float).tolist(),
        "test_mean_mae": float(np.mean(test_mae_per_output)),
        "test_predictions": test_predictions.astype(float),
        "Y_test": Y_test.astype(float),
        "prediction_correlation": prediction_correlation.astype(float),
    }
