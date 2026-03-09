"""
Q3: Compare several variational quantum classifiers on a real dataset.

Workflow:
- load a binary classification dataset from sklearn
- split it into train / validation / test subsets
- standardize the raw features, apply PCA, and rescale the retained components
- compare three QNN designs and one separate PyTorch + quantum hybrid model
- optionally compare several optimizers again under finite-shot noisy simulation

Important modeling constraints:
- use cross-entropy for classification, not mean squared error
- keep Q3.a-Q3.e noiseless; Q3.f is the only noisy-simulation extension
- do not use high-level classifier wrappers to hide the training loop

Pedagogical note:
This question is about understanding the full QML pipeline end to end:
data preparation, circuit design, QNN construction, manual optimization, model
selection, and final locked-test evaluation. The code is intentionally lower
level than a one-line sklearn-style classifier so that you can see where each
part of the workflow comes from.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# QuantumCircuit and ParameterVector are the basic tools for building custom
# re-uploading encoders and ansatz blocks by hand.
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator

# These helper functions build standard prebuilt feature maps / ansatz circuits
# for the "use Qiskit's predefined blocks" part of the question.
from qiskit.circuit.library import real_amplitudes, zz_feature_map

# SparsePauliOp defines the observable whose expectation value becomes the QNN output.
from qiskit.quantum_info import SparsePauliOp

# TorchConnector wraps a quantum neural network as a torch module, and EstimatorQNN
# is the low-level Qiskit QNN class used throughout Q3.
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient

# sklearn provides the dataset and the preprocessing pipeline used before encoding.
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.optimize import minimize

# torch.nn is only needed in the hybrid subpart, where a small classical head is
# trained together with the quantum layer.
from torch.nn import CrossEntropyLoss  # for the pytorch integration function
from torch import nn


def build_simple_noise_model(
    single_qubit_noise: float = 0.01,
    two_qubit_noise: float = 0.03,
) -> NoiseModel:
    """
    Build the simple gate-noise model used in the optional Q3.f extension.

    Students are not expected to invent this noise model from scratch. Use this
    helper directly when building the noisy simulator for compare_noisy_optimizers().
    """
    single_qubit_noise = float(single_qubit_noise)
    two_qubit_noise = float(two_qubit_noise)
    if not (0.0 <= single_qubit_noise < 0.5):
        raise ValueError("single_qubit_noise must be in [0, 0.5)")
    if not (0.0 <= two_qubit_noise < 0.5):
        raise ValueError("two_qubit_noise must be in [0, 0.5)")

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(single_qubit_noise, 1),
        ["ry", "rz"],
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(two_qubit_noise, 2),
        ["cx"],
    )
    return noise_model


def load_breast_cancer_pca_splits(
    seed: int,
    n_samples: int = 120,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """
    Load a small binary dataset, then return train/validation/test splits after
    standardization, PCA, and feature rescaling.

    The intended dataset is sklearn's breast-cancer dataset.

    Hint: fit StandardScaler(), PCA(), and the final rescaling on the training
    split only, then reuse those fitted transforms on the validation and test
    splits to avoid data leakage.

    Expected return contract:
    - return exactly one dictionary `splits`
    - `splits["X_train"]`, `splits["X_val"]`, `splits["X_test"]`: float arrays
      with shape (n_split, n_components)
    - `splits["y_train"]`, `splits["y_val"]`, `splits["y_test"]`: int arrays
      with shape (n_split,)
    """
    raise NotImplementedError(
        "Remove this line and complete the scaffold below in load_breast_cancer_pca_splits()."
    )

    # Step 1: load the full breast-cancer dataset as feature matrix X and label vector y.
    X, y = ...

    # Step 2: keep only a subset of size n_samples using train_test_split(). Use
    # stratify=y so the class balance is approximately preserved in the sampled subset.
    X_subset, _, y_subset, _ = ...

    # Step 3: split the subset into train and temporary parts. Again, use
    # stratify=y_subset when calling train_test_split(...). Use test_size=0.4 to keep
    # 60% of the dataset for training
    X_train, X_temp, y_train, y_temp = ...

    # Step 4: split the temporary part equally into validation and test sets.
    # Use stratify=y_temp here as well. Use test_size=0.5 to keep 20% of the entire
    # dataset for validation and 20% for testing.
    X_val, X_test, y_val, y_test = ...

    # Step 5: fit the standardizer on X_train only, then transform train/val/test.
    standardizer = ...
    X_train_std = ...
    X_val_std = ...
    X_test_std = ...

    # Step 6: fit PCA on the standardized training data only, then project all splits
    # to n_components dimensions. Make sure to use .fit() with the training data.
    pca = ...
    X_train_pca = ...
    X_val_pca = ...
    X_test_pca = ...

    # Step 7: rescale the PCA features to [-1, 1] by fitting the scaler on the
    # training projection only, then transforming all splits with that same scaler.
    # Make sure to use the .fit() method with the training projection.
    feature_scaler = ...

    # Step 8: return the transformed splits, integer labels
    return {
        "X_train": ...,
        "X_val": ...,
        "X_test": ...,
        "y_train": ...,
        "y_val": ...,
        "y_test": ...,
    }


def _append_chain_entanglement(
    qc: QuantumCircuit,
    reverse_entanglement: bool,
) -> None:
    """
    Append a nearest-neighbor CX chain in one of the two directions.

    This small helper is provided so the encoder and ansatz blocks can share the
    same entanglement pattern without duplicating the loop logic.
    """
    n_qubits = qc.num_qubits
    if n_qubits < 2:
        return

    if reverse_entanglement:
        for control in range(n_qubits - 1, 0, -1):
            qc.cx(control, control - 1)
    else:
        for control in range(n_qubits - 1):
            qc.cx(control, control + 1)


def _append_encoder_block(
    qc: QuantumCircuit,
    angles,
    reverse_entanglement: bool,
) -> None:
    """
    Append one encoder block on all qubits in place.

    This helper is provided so that the manual-QNN construction questions can
    focus on model design rather than on rewriting the same gate pattern.

    The encoder uses one retained PCA feature per qubit, applies a local
    rotation pattern on each qubit, then adds a nearest-neighbor entangling
    chain.

    Implementation note:
    `angles` should contain exactly one angle per qubit. Its entries may be
    plain numbers, Parameters, or more general ParameterExpression objects.
    """
    angle_list = list(angles)
    if len(angle_list) != qc.num_qubits:
        raise ValueError("encoder angle count must match the number of qubits")

    for qubit, angle in enumerate(angle_list):
        qc.ry(angle, qubit)
        qc.rz(angle, qubit)

    _append_chain_entanglement(qc, reverse_entanglement)


def _append_manual_ansatz(
    qc: QuantumCircuit,
    weights,
    ansatz_layers: int,
    reverse_entanglement: bool,
) -> None:
    """
    Append one small trainable ansatz block in place.

    For the suggested construction, each ansatz layer uses two trainable
    rotations per qubit, followed by the same nearest-neighbor entangling chain
    as the encoder. Therefore each ansatz layer consumes exactly
    `2 * qc.num_qubits` trainable parameters.
    """
    idx = 0
    for _ in range(int(ansatz_layers)):
        for qubit in range(qc.num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
            qc.rz(weights[idx], qubit)
            idx += 1
        _append_chain_entanglement(qc, reverse_entanglement)


def _build_z_observables(
    n_qubits: int,
    observable_mode: str = "global",
) -> SparsePauliOp:
    n_qubits = int(n_qubits)
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")

    if observable_mode == "global":
        return SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])

    if observable_mode == "single":
        paulis = []
        for qubit in range(n_qubits):
            label = ["I"] * n_qubits
            label[qubit] = "Z"
            paulis.append(("".join(label), 1.0))
        return SparsePauliOp.from_list(paulis)

    if observable_mode == "single_and_global":
        paulis = []
        for qubit in range(n_qubits):
            label = ["I"] * n_qubits
            label[qubit] = "Z"
            paulis.append(("".join(label), 1.0))
        paulis.append(("Z" * n_qubits, 1.0))
        return SparsePauliOp.from_list(paulis)

    raise ValueError(
        "observable_mode must be 'global', 'single', or 'single_and_global'"
    )


def _build_estimator_qnn(
    circuit: QuantumCircuit,
    observable_mode: str,
    input_params,
    weight_params,
) -> EstimatorQNN:
    """
    Build a basic EstimatorQNN with a statevector estimator and parameter-shift
    gradient support.

    This helper is provided so the main QNN-construction functions can focus on
    circuit structure, observables, and parameter bookkeeping.
    """
    observables = _build_z_observables(circuit.num_qubits, observable_mode)
    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator)
    return EstimatorQNN(
        circuit=circuit,
        estimator=estimator,
        observables=observables,
        input_params=list(input_params),
        weight_params=list(weight_params),
        gradient=gradient,
    )


def build_manual_reuploading_qnn(
    re_uploading: int = 3,
    ansatz_layers: int = 1,
    n_qubits: int = 2,
    observable_mode: str = "global",
) -> EstimatorQNN:
    """
    Build a QNN whose encoder and ansatz are both written manually.

    Requirements:
    - use explicit rotation gates and CNOT entanglers
    - organize the circuit as repeated encoder-ansatz blocks
    - use `re_uploading` to control how many encoder-ansatz blocks are stacked one after
    another
    - keep trainable parameters only in the ansatz for this first model

    Important:
    You do not need to manually write all the gates here. You can reuse the
    helper functions defined above:
    - `_append_encoder_block(...)`
    - `_append_manual_ansatz(...)`
    - `_build_estimator_qnn(...)`

    These helpers are provided so that this function can focus on assembling the
    overall circuit structure rather than rewriting the same gate patterns.

    Suggested construction:
    - use one qubit per retained PCA feature
    - define an input `ParameterVector x` of length `n_qubits`
    - define a trainable `ParameterVector theta`
    - for each block:
        1. append an encoder using all entries of `x`
        2. append an ansatz using a slice of `theta`
    - use a simple observable such as `Z ⊗ ... ⊗ Z` for the output expectation

    Expected return contract:
    - return exactly one `EstimatorQNN`
    - `qnn.num_inputs == n_qubits`
    - the trainable weight count should be
      `2 * n_qubits * ansatz_layers * re_uploading` for the suggested ansatz structure
    - the QNN output should be one scalar expectation value per sample
    """
    raise NotImplementedError("build_manual_reuploading_qnn not implemented")

    # Step 1: validate re_uploading, ansatz_layers, and n_qubits.
    re_uploading = int(re_uploading)
    ansatz_layers = int(ansatz_layers)
    n_qubits = int(n_qubits)
    if re_uploading < 1:
        raise ValueError("re_uploading must be at least 1")
    if ansatz_layers < 1:
        raise ValueError("ansatz_layers must be at least 1")
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")

    # Step 2: create the input parameters and the trainable ansatz weights.
    x = ParameterVector("x", n_qubits)
    theta = ...

    # Step 3: build the n-qubit circuit by repeating
    # [encoder -> ansatz] re_uploading times.
    qc = ...
    idx = 0
    for block in range(re_uploading):
        reverse_entanglement = bool(block % 2)
        _append_encoder_block(...)
        _append_manual_ansatz(...)
        idx += ...

    # Step 4: return an EstimatorQNN with a simple scalar Z...Z observable.
    return _build_estimator_qnn(
        ...,
        observable_mode,
        ...,
        ...,
    )


def _expectation_to_logit(
    expectation_values: np.ndarray,
    scale: float = 2.5,
) -> np.ndarray:
    """
    Convert bounded expectation values into effective logits for binary
    cross-entropy.

    Why this helper is useful:
    QNN outputs are expectation values, usually in a bounded interval close to
    [-1, 1], while cross-entropy is more naturally written in terms of logits on
    the whole real line.

    Strategy:
    - clip the expectation values slightly away from -1 and 1
    - apply a monotone transformation to obtain logits

    Expected return contract:
    - return exactly one numpy float array
    - larger expectations should map to larger logits
    """
    clipped = np.clip(np.asarray(expectation_values, dtype=float), -0.98, 0.98)
    return float(scale) * np.arctanh(clipped)


def _forward_logits(
    qnn: EstimatorQNN,
    X: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Run the QNN forward pass and convert its raw outputs to 1D logits.

    This helper is useful because several parts of Q3 repeatedly need:
    - qnn.forward(...)
    - conversion to numpy
    - flattening
    - mapping expectation values to logits

    Expected return contract:
    - return exactly one 1D float numpy array with shape `(n_samples,)`
    """
    expectations = np.asarray(
        qnn.forward(np.asarray(X, dtype=float), np.asarray(weights, dtype=float)),
        dtype=float,
    )
    return _expectation_to_logit(expectations.reshape(-1))


def _binary_cross_entropy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Return the mean binary cross-entropy loss from logits and binary targets.

    The targets must use the convention `{0, 1}`.

    A numerically stable per-sample formula is

        logaddexp(0, z) - y * z

    where z is the logit.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    targets = np.asarray(y, dtype=float).reshape(-1)
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


def _binary_cross_entropy_logit_gradient(
    logits: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Return the derivative of the mean binary cross-entropy with respect to each
    sample logit.

    This helper is useful in the manual training loop, where you will combine:
    - d(loss)/d(logit)
    - d(logit)/d(expectation)
    - d(expectation)/d(weights)

    with the chain rule.

    Expected return contract:
    - return exactly one 1D float numpy array with shape `(n_samples,)`
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    targets = np.asarray(y, dtype=float).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs - targets) / float(logits.shape[0])


def _accuracy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Compute binary classification accuracy from logits.

    Convention:
    - logit >= 0  -> class 1
    - logit <  0  -> class 0

    Expected return contract:
    - return exactly one Python float in [0, 1]
    """
    preds = (np.asarray(logits, dtype=float).reshape(-1) >= 0.0).astype(int)
    targets = np.asarray(y, dtype=int).reshape(-1)
    return float(np.mean(preds == targets))


def train_qnn_with_cross_entropy(
    qnn: EstimatorQNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int = 12,
    lr: float = 0.25,
    eps: float = 1e-2,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """
    Train QNN weights with your own optimization loop and a cross-entropy loss.

    Targets should use the binary convention {0, 1}. This part is intentionally
    low level: do not use a high-level classifier wrapper.

    You can use the helper functions defined above.

    Main idea:
    1. initialize the trainable weights
    2. run a forward pass on the training set
    3. convert expectations to logits
    4. compute the binary cross-entropy
    5. use `qnn.backward(...)` to obtain the Jacobian with respect to weights
    6. combine everything with the chain rule
    7. update the weights
    8. record train/validation loss and accuracy at each epoch

    Chain rule structure:
        dL/dw = dL/dlogit · dlogit/dexpectation · dexpectation/dw

    Hint:
    because the expectation-to-logit map is nonlinear, you need to include
    `dlogit/dexpectation` explicitly in the gradient.

    Expected return contract:
    - return exactly one tuple `(weights, history)`
    - `weights`: 1D numpy float array with shape `(qnn.num_weights,)`
    - `history`: dictionary with exactly the keys
      `"train_loss_history"`, `"val_loss_history"`,
      `"train_accuracy_history"`, `"val_accuracy_history"`
    - each history list must have length `epochs`
    """
    raise NotImplementedError("train_qnn_with_cross_entropy not implemented")

    # Step 1: coerce arrays to the expected dtypes and shapes.
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int).reshape(-1)

    # Step 2: validate the feature dimensions and label lengths.
    if X_train.ndim != 2 or X_train.shape[1] != qnn.num_inputs:
        raise ValueError("X_train has incorrect shape")
    if X_val.ndim != 2 or X_val.shape[1] != qnn.num_inputs:
        raise ValueError("X_val has incorrect shape")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError("y_train length must match X_train")
    if y_val.shape[0] != X_val.shape[0]:
        raise ValueError("y_val length must match X_val")

    # Step 3: initialize the trainable weights with a small random normal.
    rng = np.random.default_rng(int(seed))
    weights = rng.normal(loc=0.0, scale=0.15, size=qnn.num_weights)

    # Step 4: initialize the history dictionary.
    history = {
        "train_loss_history": [],
        "val_loss_history": [],
        "train_accuracy_history": [],
        "val_accuracy_history": [],
    }

    # Step 5: run gradient-based training for the requested number of epochs.
    for _ in range(int(epochs)):
        # 5a: forward pass on the training set to obtain expectations/logits.
        expectations = np.asarray(..., dtype=float).reshape(-1)
        logits = ...

        # 5b: compute the Jacobian of expectations with respect to weights.
        _, weight_jacobian = qnn.backward(X_train, weights)
        weight_jacobian = np.asarray(weight_jacobian, dtype=float).reshape(
            X_train.shape[0], qnn.num_weights
        )

        # 5c: compute dlogit/dexpectation for your expectation-to-logit mapping.
        clipped_expectations = np.clip(expectations, -0.98, 0.98)
        dlogit_dexpectation = 2.5 / (1.0 - clipped_expectations**2)

        # 5d: combine the pieces with the chain rule to obtain the full gradient.
        grad = (
            _binary_cross_entropy_logit_gradient(logits, y_train) * dlogit_dexpectation
        ) @ weight_jacobian

        # 5e: optionally clip the gradient norm if it becomes too large.
        grad_norm = ...
        if grad_norm > 5.0:
            grad = grad * (5.0 / grad_norm)

        # 5f: perform one gradient-descent update step.
        weights = ...

        # 5g: evaluate train/validation logits after the update.
        train_logits = _forward_logits(...)
        val_logits = _forward_logits(...)

        # 5h: append train/validation losses and accuracies to the history.
        history["train_loss_history"].append(...)
        history["val_loss_history"].append(...)
        history["train_accuracy_history"].append(...)
        history["val_accuracy_history"].append(...)

    return weights.astype(float), history


def predict_from_qnn(
    qnn: EstimatorQNN,
    weights: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return binary class predictions in {0, 1}.

    Hint:
    - run the forward pass
    - convert the output to logits
    - threshold at 0

    Expected return contract:
    - return exactly one numpy array
    - the array must be a 1D int numpy array with shape `(n_samples,)`
    - every entry must be either `0` or `1`
    """
    raise NotImplementedError("predict_from_qnn not implemented")

    logits = ...
    return ...


def build_polynomial_reuploading_qnn(
    degree: int = 1,
    re_uploading: int = 3,
    ansatz_layers: int = 1,
    n_qubits: int = 2,
    observable_mode: str = "global",
) -> EstimatorQNN:
    """
    Build a manual QNN where the encoder itself has trainable parameters.

    The encoder should map each retained PCA feature x to a trainable polynomial
    in x of the chosen degree before feeding that value into the rotation gates.
    Use one qubit per retained PCA feature, so the circuit width is `n_qubits`.

    Here, degree means the highest exponent kept, so:
    - degree 1 means `a0 + a1 * x`
    - degree 2 means `a0 + a1 * x + a2 * x**2`

    Main difference from build_manual_reuploading_qnn():
    - the first manual model keeps trainable parameters only in the ansatz
    - this model also learns the feature map itself

    Hint:
    Qiskit ParameterExpression objects let you build angles such as
    a0 + a1 * x + a2 * x**2 directly inside the circuit.

    Expected return contract:
    - return exactly one `EstimatorQNN`
    - `qnn.num_inputs == n_qubits`
    - if each block has its own polynomial coefficients, the trainable weight
      count should be
      `n_qubits * (degree + 1) * re_uploading
      + 2 * n_qubits * ansatz_layers * re_uploading`
      for the suggested construction
    - the QNN output should be one scalar expectation value per sample
    """
    raise NotImplementedError("build_polynomial_reuploading_qnn not implemented")

    # Step 1: validate all integer hyperparameters.
    degree = ...
    re_uploading = ...
    ansatz_layers = ...
    n_qubits = ...
    if ...:
        raise ValueError("degree must be at least 1")
    if ...:
        raise ValueError("re_uploading must be at least 1")
    if ...:
        raise ValueError("ansatz_layers must be at least 1")
    if ...:
        raise ValueError("n_qubits must be at least 1")

    # Step 2: define the input parameters, encoder polynomial coefficients,
    # and ansatz parameters.
    x = ...
    coeffs = ...
    theta = ...

    # Step 3: build the circuit block by block.
    qc = ...
    theta_idx = 0
    for block in range(re_uploading):
        coeff_offset = ...

        # Build one polynomial angle per retained PCA feature / qubit.
        block_angles = ...

        reverse_entanglement = ...
        _append_encoder_block(...)
        _append_manual_ansatz(...)

        theta_idx += ...

    # Step 4: return the EstimatorQNN. The full trainable parameter list must
    # include both encoder coefficients and ansatz weights.
    return _build_estimator_qnn(
        ...,
        ...,
        ...,
        ...,
    )


def build_prebuilt_qiskit_qnn(
    re_uploading: int = 3,
    ansatz_layers: int = 1,
    n_qubits: int = 2,
    observable_mode: str = "global",
) -> EstimatorQNN:
    """
    Build a noiseless reference QNN from prebuilt Qiskit feature-map / ansatz
    blocks.

    Requirements:
    - use prebuilt Qiskit blocks such as `zz_feature_map()` and
      `real_amplitudes()`
    - still organize the full circuit as repeated encoder-ansatz blocks
    - use `re_uploading` to control how many blocks are stacked
    - use `ansatz_layers` to control the depth inside each ansatz block

    Why this model is useful:
    it provides a reference based on standard Qiskit components, so you can
    compare manual design choices against a more library-based construction.

    Expected return contract:
    - return exactly one `EstimatorQNN`
    - `qnn.num_inputs == n_qubits`
    - the trainable weight count must be positive and should increase when
      `re_uploading`, `ansatz_layers`, or `n_qubits` increases
    - the QNN output should be one scalar expectation value per sample
    """
    raise NotImplementedError("build_prebuilt_qiskit_qnn not implemented")

    # Step 1: validate the hyperparameters.
    re_uploading = ...
    ansatz_layers = ...
    n_qubits = ...
    if ...:
        raise ValueError("re_uploading must be at least 1")
    if ...:
        raise ValueError("ansatz_layers must be at least 1")
    if ...:
        raise ValueError("n_qubits must be at least 1")

    # Step 2: create the shared input parameter vector and an empty circuit.
    zz_feature_map_params = ParameterVector("x", n_qubits)
    circuit = ...
    weight_params = []

    # Step 3: for each re-uploading block:
    # - instantiate an n-qubit zz_feature_map
    # - remap its feature parameters to the shared vector x
    # - instantiate a real_amplitudes ansatz with its own trainable parameters
    # - append both blocks to the circuit
    # - store the ansatz parameters in weight_params
    for block in range(re_uploading):
        feature_map = ...
        feature_params = list(feature_map.parameters)
        feature_map = feature_map.assign_parameters(
            {...},
            inplace=False,
        )

        ansatz = ...

        circuit.compose(..., inplace=True)
        circuit.compose(..., inplace=True)
        weight_params.extend(...)

    # Step 4: return the final EstimatorQNN.
    return _build_estimator_qnn(
        ...,
        ...,
        ...,
        ...,
    )


def _make_decision_plane_grid(
    X_reference: np.ndarray,
    resolution: int,
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


def run_q3_model_selection(
    seed: int,
    degree: int = 1,
    n_samples: int = 120,
    re_uploading: int = 3,
    pca_components: int = 2,
    epochs=20,
    lr=0.2,
) -> dict[str, object]:
    """
    Compare the first three QNN variants on validation data only, then evaluate
    the single best model on the locked test set.

    Why this protocol matters:
    - training data is used to fit weights
    - validation data is used to compare model families
    - the test set must stay locked until after model selection

    Recommended procedure:
    1. create the train/validation/test splits once
    2. use `pca_components` both for the PCA reduction and for the number of
       qubits / classical inputs in the QNN builders
    3. train the manual QNN
    4. train the polynomial QNN
    5. train the prebuilt Qiskit QNN
    6. compare their validation accuracies
    7. evaluate only the best selected model on the test set

    Expected return contract:
    - return exactly one dictionary `results`
    - `results["manual"]`, `results["polynomial"]`, `results["prebuilt"]`:
      dicts with the same four history lists as train_qnn_with_cross_entropy(),
      plus scalar train/validation accuracies
    - `results["best_model"]`: one of `"manual"`, `"polynomial"`, `"prebuilt"`
    - `results["best_test_accuracy"]`: float in [0, 1]
    - `results["X_test"]`: float array with shape `(n_test, pca_components)`
    - `results["y_test"]`: int array with shape `(n_test,)`
    - `results["best_test_predictions"]`: int array with shape `(n_test,)`
    - when `pca_components == 2`, also return
      `results["decision_grid_x"]`, `results["decision_grid_y"]`,
      `results["decision_grid_predictions"]` for the 2D decision-region plot
    """
    raise NotImplementedError("run_q3_model_selection not implemented")

    # Step 1: build the dataset splits once with the requested PCA dimension.
    splits = ...

    # Step 2: train the manual model and compute its final train/validation accuracy.
    manual_qnn = ...
    manual_weights, manual_history = ...
    manual_train_acc = ...
    manual_val_acc = ...

    # Step 3: train the polynomial model and compute its final train/validation accuracy.
    poly_qnn = ...
    poly_weights, poly_history = ...
    poly_train_acc = ...
    poly_val_acc = ...

    # Step 4: train the prebuilt model and compute its final train/validation accuracy.
    prebuilt_qnn = ...
    prebuilt_weights, prebuilt_history = ...
    prebuilt_train_acc = ...
    prebuilt_val_acc = ...

    # Step 5: select the best model using validation accuracy only.
    validation_scores = {
        "manual": ...,
        "polynomial": ...,
        "prebuilt": ...,
    }
    best_model = ...

    # Step 6: evaluate only the selected best model on the locked test set.
    if best_model == "manual":
        best_test_predictions = ...
        best_test_acc = ...
    elif best_model == "polynomial":
        best_test_predictions = ...
        best_test_acc = ...
    else:
        best_test_predictions = ...
        best_test_acc = ...

    decision_payload: dict[str, np.ndarray] = {}
    if int(pca_components) == 2:
        X_reference = np.vstack([splits["X_train"], splits["X_val"], splits["X_test"]])
        grid_x, grid_y, grid_points = _make_decision_plane_grid(
            X_reference, resolution=60
        )
        if best_model == "manual":
            grid_predictions = predict_from_qnn(manual_qnn, manual_weights, grid_points)
        elif best_model == "polynomial":
            grid_predictions = predict_from_qnn(poly_qnn, poly_weights, grid_points)
        else:
            grid_predictions = predict_from_qnn(
                prebuilt_qnn, prebuilt_weights, grid_points
            )
        decision_payload = {
            "decision_grid_x": grid_x.astype(float),
            "decision_grid_y": grid_y.astype(float),
            "decision_grid_predictions": np.asarray(
                grid_predictions, dtype=int
            ).reshape(grid_x.shape),
        }

    # Step 7: return the final nested results dictionary.
    return {
        "manual": {
            **manual_history,
            "train_accuracy": ...,
            "val_accuracy": ...,
        },
        "polynomial": {
            **poly_history,
            "train_accuracy": ...,
            "val_accuracy": ...,
        },
        "prebuilt": {
            **prebuilt_history,
            "train_accuracy": ...,
            "val_accuracy": ...,
        },
        "best_model": ...,
        "best_test_accuracy": ...,
        "X_test": ...,
        "y_test": ...,
        "best_test_predictions": ...,
        **decision_payload,
    }


def train_torch_hybrid_classifier(
    seed: int,
    n_samples: int = 120,
    epochs: int = 20,
    lr: float = 5e-2,
    re_uploading: int = 3,
    pca_components: int = 2,
) -> dict[str, object]:
    """
    Train a simple hybrid classifier with a quantum layer followed by a small
    classical neural network using PyTorch autograd.

    Main idea:
    - build a quantum layer that outputs one scalar feature per sample
    - wrap it with TorchConnector so it behaves like a torch module
    - feed that quantum output into a small classical head
    - optimize the whole hybrid model end to end with standard PyTorch training

    Suggested head:
        Linear(1, hidden) -> nonlinearity -> Linear(hidden, 2)

    Suggested training loop:
    - use CrossEntropyLoss()
    - use a standard torch optimizer such as Adam
    - track train/validation loss and accuracy
    - keep the best validation checkpoint
    - evaluate the restored best model on the locked test set

    Expected return contract:
    - return exactly one dictionary `results`
    - `results["train_loss_history"]`: list[float] with one value per epoch
    - `results["val_loss_history"]`: list[float] with one value per epoch
    - `results["train_accuracy_history"]`: list[float] with one value per epoch
    - `results["val_accuracy_history"]`: list[float] with one value per epoch
    - `results["val_accuracy"]`: final validation accuracy as a float in [0, 1]
    - `results["test_accuracy"]`: final locked-test accuracy as a float in [0, 1]
    - `results["X_test"]`: float array with shape `(n_test, pca_components)`
    - `results["y_test"]`: int array with shape `(n_test,)`
    - `results["test_predictions"]`: int array with shape `(n_test,)`
    - when `pca_components == 2`, also return
      `results["decision_grid_x"]`, `results["decision_grid_y"]`,
      `results["decision_grid_predictions"]` for the 2D decision-region plot
    """
    raise NotImplementedError("train_torch_hybrid_classifier not implemented")

    # Step 1: build the dataset splits and the underlying QNN with the requested
    # PCA dimension / number of qubits.
    splits = ...
    qnn = build_manual_reuploading_qnn(
        int(re_uploading), 1, int(pca_components), observable_mode="single_and_global"
    )

    # Step 2: wrap the QNN as a torch module with TorchConnector.
    rng = ...
    quantum_layer = TorchConnector(
        qnn,
        initial_weights=rng.normal(loc=0.0, scale=0.15, size=qnn.num_weights),
    )

    # Step 3: define a small hybrid model:
    # quantum layer -> small classical head.
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantum = ...
            self.head = nn.Sequential(
                nn.Linear(int(qnn.output_shape[0]), 8),
                nn.Tanh(),
                nn.Linear(8, 2),
            )

        def forward(self, x):
            q_out = ...
            if q_out.ndim == 1:
                q_out = q_out.unsqueeze(0)
            return ...

    # Step 4: create the model, optimizer, and loss.
    torch.manual_seed(int(seed))
    model = ...
    optimizer = torch.optim.Adam(...)
    criterion = ...

    # Step 5: convert the dataset splits to torch tensors, use the correct dtypes.
    X_train = ...
    y_train = ...
    X_val = ...
    y_val = ...
    X_test = ...
    y_test = ...

    # Step 6: initialize histories and best-validation tracking.
    history = {
        "train_loss_history": [],
        "val_loss_history": [],
        "train_accuracy_history": [],
        "val_accuracy_history": [],
    }
    best_val_acc = ...
    best_state = ...

    # Step 7: train for the requested number of epochs.
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = ...
        loss = ...
        loss.backward()
        optimizer.step()

        train_pred = ...
        train_acc = ...

        model.eval()
        with torch.no_grad():
            val_logits = ...
            val_loss = ...
            val_pred = ...
            val_acc = ...

        history["train_loss_history"].append(...)
        history["val_loss_history"].append(...)
        history["train_accuracy_history"].append(...)
        history["val_accuracy_history"].append(...)

        if val_acc > best_val_acc:
            best_val_acc = ...
            best_state = ...

    # Step 8: restore the best validation checkpoint and evaluate on test.
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = ...
        test_pred = ...
        test_acc = ...

    decision_payload: dict[str, np.ndarray] = {}
    if int(pca_components) == 2:
        X_reference = np.vstack([splits["X_train"], splits["X_val"], splits["X_test"]])
        grid_x, grid_y, grid_points = _make_decision_plane_grid(
            X_reference, resolution=60
        )
        with torch.no_grad():
            grid_logits = model(torch.tensor(grid_points, dtype=torch.float32))
            grid_predictions = (
                torch.argmax(grid_logits, dim=1).cpu().numpy().astype(int)
            )
        decision_payload = {
            "decision_grid_x": grid_x.astype(float),
            "decision_grid_y": grid_y.astype(float),
            "decision_grid_predictions": grid_predictions.reshape(grid_x.shape),
        }

    # Step 9: return the final results dictionary.
    return {
        **history,
        "val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "X_test": splits["X_test"].astype(float),
        "y_test": splits["y_test"].astype(int),
        "test_predictions": test_pred.detach().cpu().numpy().astype(int),
        **decision_payload,
    }


# ---------------------------------------------------------------------------
# Provided Q3.f helpers
# ---------------------------------------------------------------------------


def _build_noisy_simulator() -> AerSimulator:
    return AerSimulator(noise_model=build_simple_noise_model())


def _z_string_expectation_from_counts(counts: dict[str, int]) -> float:
    total_shots = float(sum(counts.values()))
    expectation = 0.0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        parity = 1.0
        for bit in bits:
            parity *= 1.0 if bit == "0" else -1.0
        expectation += parity * float(count) / total_shots
    return float(expectation)


def _forward_noisy_logits(
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    X: np.ndarray,
    weights: np.ndarray,
    backend: AerSimulator,
    rng: np.random.Generator,
    shots: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    circuits = []
    weight_mapping = {
        param: float(value) for param, value in zip(weight_params, weights, strict=True)
    }
    for row in X:
        mapping = {
            param: float(value) for param, value in zip(input_params, row, strict=True)
        }
        mapping.update(weight_mapping)
        circuits.append(prepared_circuit.assign_parameters(mapping, inplace=False))

    result = backend.run(
        circuits,
        shots=int(shots),
        seed_simulator=int(rng.integers(0, 2**31 - 1)),
    ).result()

    expectations = np.empty(X.shape[0], dtype=float)
    for idx in range(X.shape[0]):
        expectations[idx] = _z_string_expectation_from_counts(result.get_counts(idx))

    return _expectation_to_logit(expectations)


def _evaluate_noisy_split(
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    backend: AerSimulator,
    rng: np.random.Generator,
    shots: int,
) -> dict[str, object]:
    logits = _forward_noisy_logits(
        prepared_circuit,
        input_params,
        weight_params,
        X,
        weights,
        backend,
        rng,
        shots,
    )
    return {
        "logits": logits,
        "loss": _binary_cross_entropy_from_logits(logits, y),
        "accuracy": _accuracy_from_logits(logits, y),
    }


def _evaluate_noisy_splits(
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    weights: np.ndarray,
    backend: AerSimulator,
    rng: np.random.Generator,
    shots: int,
    split_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, object]]:
    return {
        split_name: _evaluate_noisy_split(
            prepared_circuit,
            input_params,
            weight_params,
            X,
            y,
            weights,
            backend,
            rng,
            shots,
        )
        for split_name, (X, y) in split_data.items()
    }


def _noisy_batch_loss(
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    backend: AerSimulator,
    rng: np.random.Generator,
    shots: int,
) -> float:
    logits = _forward_noisy_logits(
        prepared_circuit,
        input_params,
        weight_params,
        X,
        weights,
        backend,
        rng,
        shots,
    )
    return _binary_cross_entropy_from_logits(logits, y)


def _sample_minibatch(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.choice(X.shape[0], size=int(batch_size), replace=False)
    return X[idx], y[idx]


def _record_noisy_metrics(
    history: dict[str, list[float]],
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    weights: np.ndarray,
    backend: AerSimulator,
    rng: np.random.Generator,
    shots: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    metrics = _evaluate_noisy_splits(
        prepared_circuit,
        input_params,
        weight_params,
        weights,
        backend,
        rng,
        shots,
        {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
        },
    )

    history["train_loss_history"].append(float(metrics["train"]["loss"]))
    history["val_loss_history"].append(float(metrics["val"]["loss"]))
    history["train_accuracy_history"].append(float(metrics["train"]["accuracy"]))
    history["val_accuracy_history"].append(float(metrics["val"]["accuracy"]))

    return history["train_accuracy_history"][-1], history["val_accuracy_history"][-1]


def _pad_history_to_length(
    history: dict[str, list[float]],
    target_length: int,
) -> dict[str, list[float]]:
    padded = {key: list(values) for key, values in history.items()}
    for key, values in padded.items():
        if not values:
            raise ValueError("history must contain at least one value before padding")
        if len(values) > int(target_length):
            del values[int(target_length) :]
        while len(values) < int(target_length):
            values.append(float(values[-1]))
    return padded


def _train_noisy_with_optimizer(
    prepared_circuit: QuantumCircuit,
    input_params,
    weight_params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    shots: int,
    optimizer_name: str,
    initial_weights: np.ndarray,
    metric_seed: int,
    lr: float | None = None,
    eps: float | None = None,
    perturbation: float | None = None,
    batch_size: int | None = None,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    rng = np.random.default_rng(int(seed))
    metric_rng = np.random.default_rng(int(metric_seed))
    backend = _build_noisy_simulator()

    optimizer_name = str(optimizer_name).lower()
    if optimizer_name not in {"adam", "cobyla", "spsa"}:
        raise ValueError("optimizer_name must be 'adam', 'cobyla', or 'spsa'")

    weights = np.asarray(initial_weights, dtype=float).copy()
    history = {
        "train_loss_history": [],
        "val_loss_history": [],
        "train_accuracy_history": [],
        "val_accuracy_history": [],
    }
    effective_batch_size = min(int(batch_size or X_train.shape[0]), X_train.shape[0])

    _record_noisy_metrics(
        history,
        prepared_circuit,
        input_params,
        weight_params,
        weights,
        backend,
        metric_rng,
        shots,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    if optimizer_name == "adam":
        if lr is None or eps is None:
            raise ValueError("adam requires lr and eps")

        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        beta1 = 0.9
        beta2 = 0.999

        for step_idx in range(1, int(epochs) + 1):
            X_batch, y_batch = _sample_minibatch(
                X_train, y_train, rng, effective_batch_size
            )
            grad = np.zeros_like(weights)

            for idx in range(weights.size):
                direction = np.zeros_like(weights)
                direction[idx] = float(eps)

                loss_plus = _noisy_batch_loss(
                    prepared_circuit,
                    input_params,
                    weight_params,
                    X_batch,
                    y_batch,
                    weights + direction,
                    backend,
                    rng,
                    shots,
                )
                loss_minus = _noisy_batch_loss(
                    prepared_circuit,
                    input_params,
                    weight_params,
                    X_batch,
                    y_batch,
                    weights - direction,
                    backend,
                    rng,
                    shots,
                )
                grad[idx] = (loss_plus - loss_minus) / (2.0 * float(eps))

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad**2)
            m_hat = m / (1.0 - beta1**step_idx)
            v_hat = v / (1.0 - beta2**step_idx)
            weights = weights - float(lr) * m_hat / (np.sqrt(v_hat) + 1e-8)

            _record_noisy_metrics(
                history,
                prepared_circuit,
                input_params,
                weight_params,
                weights,
                backend,
                metric_rng,
                shots,
                X_train,
                y_train,
                X_val,
                y_val,
            )

        return weights.astype(float), _pad_history_to_length(history, int(epochs))

    if optimizer_name == "spsa":
        if lr is None or perturbation is None:
            raise ValueError("spsa requires lr and perturbation")

        for step_idx in range(int(epochs)):
            X_batch, y_batch = _sample_minibatch(
                X_train, y_train, rng, effective_batch_size
            )
            ck = float(perturbation) / ((step_idx + 1.0) ** 0.101)
            ak = float(lr) / ((step_idx + 1.0) ** 0.602)
            delta = rng.choice([-1.0, 1.0], size=weights.size)

            loss_plus = _noisy_batch_loss(
                prepared_circuit,
                input_params,
                weight_params,
                X_batch,
                y_batch,
                weights + ck * delta,
                backend,
                rng,
                shots,
            )
            loss_minus = _noisy_batch_loss(
                prepared_circuit,
                input_params,
                weight_params,
                X_batch,
                y_batch,
                weights - ck * delta,
                backend,
                rng,
                shots,
            )
            grad = ((loss_plus - loss_minus) / (2.0 * ck)) * delta
            weights = weights - ak * grad

            _record_noisy_metrics(
                history,
                prepared_circuit,
                input_params,
                weight_params,
                weights,
                backend,
                metric_rng,
                shots,
                X_train,
                y_train,
                X_val,
                y_val,
            )

        return weights.astype(float), _pad_history_to_length(history, int(epochs))

    def objective(current_weights: np.ndarray) -> float:
        return _noisy_batch_loss(
            prepared_circuit,
            input_params,
            weight_params,
            X_train,
            y_train,
            np.asarray(current_weights, dtype=float),
            backend,
            rng,
            shots,
        )

    def callback(current_weights: np.ndarray) -> None:
        if len(history["train_loss_history"]) >= int(epochs):
            return
        _record_noisy_metrics(
            history,
            prepared_circuit,
            input_params,
            weight_params,
            np.asarray(current_weights, dtype=float),
            backend,
            metric_rng,
            shots,
            X_train,
            y_train,
            X_val,
            y_val,
        )

    result = minimize(
        objective,
        weights,
        method="COBYLA",
        callback=callback,
        options={"maxiter": max(12, 4 * int(epochs)), "rhobeg": 0.35, "disp": False},
    )
    final_weights = np.asarray(result.x, dtype=float)

    final_metrics = _evaluate_noisy_splits(
        prepared_circuit,
        input_params,
        weight_params,
        final_weights,
        backend,
        metric_rng,
        shots,
        {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
        },
    )

    if not history["train_loss_history"]:
        history["train_loss_history"].append(float(final_metrics["train"]["loss"]))
        history["val_loss_history"].append(float(final_metrics["val"]["loss"]))
        history["train_accuracy_history"].append(
            float(final_metrics["train"]["accuracy"])
        )
        history["val_accuracy_history"].append(float(final_metrics["val"]["accuracy"]))
    else:
        history["train_loss_history"][-1] = float(final_metrics["train"]["loss"])
        history["val_loss_history"][-1] = float(final_metrics["val"]["loss"])
        history["train_accuracy_history"][-1] = float(
            final_metrics["train"]["accuracy"]
        )
        history["val_accuracy_history"][-1] = float(final_metrics["val"]["accuracy"])

    return final_weights.astype(float), _pad_history_to_length(history, int(epochs))


def compare_noisy_optimizers(
    seed: int,
    n_samples: int = 60,
    re_uploading: int = 2,
    epochs: int = 20,
    lr: float = 0.1,
    shots: int = 256,
    pca_components: int = 2,
) -> dict[str, object]:
    """
    Q3.f: compare several optimizers on the same shallow
    re-uploading classifier under finite-shot noisy simulation.

    Suggested approach:
    - reuse the manual re-uploading circuit from Q3.b with `pca_components`
      qubits / inputs
    - keep the circuit fixed and compare only the optimizer choice
    - use a noisy Aer simulation with finite shots and the provided
      build_simple_noise_model() helper
    - compare Adam, COBYLA, and SPSA on validation accuracy
    - keep the train / validation / test discipline from the rest of Q3

    One reasonable implementation strategy is:
    1. Build the shallow manual QNN and extract its circuit / parameters.
    2. Sample one shared initial weight vector and use that exact same starting
       point for Adam, COBYLA, and SPSA.
    3. Add measurements and evaluate the `Z ⊗ ... ⊗ Z` expectation
       from noisy shot counts.
    4. Record the initial train/validation loss and accuracy before any update,
       so all optimizer curves start from the same point.
    5. Train the same circuit three times, once with Adam, once with COBYLA,
       and once with SPSA.
    6. Select the best optimizer by validation accuracy, then report its locked
       test accuracy.

    Important:
    - the noisy helper functions directly above are already provided
    - you are not expected to create the noise model, shot-based forward pass,
      or optimizer-specific noisy training code from scratch
    - your work here is mainly to assemble the comparison pipeline and return
      the final dictionary in the exact required format
    """
    raise NotImplementedError("compare_noisy_optimizers not implemented")
    # Step 1: load the PCA-reduced train/validation/test splits, then build the
    # shallow manual QNN used for all three noisy optimizer runs.
    splits = ...
    qnn = ...
    backend = ...

    # Step 2: measure the circuit so the noisy simulator can return bitstring
    # counts, then transpile it once for the Aer backend.
    measured_circuit = ...
    measured_circuit.measure_all()
    prepared_circuit = ...

    # Step 3: prepare the shared parameter lists and the common initial weight
    # vector so the optimizer comparison starts fairly from the same point.
    input_params = list(qnn.input_params)
    weight_params = list(qnn.weight_params)
    rng = np.random.default_rng(int(seed))
    initial_weights = rng.uniform(-np.pi / 16, np.pi / 16, size=len(weight_params))
    metric_seed = int(seed) + 10000

    # Step 4: reuse the provided noisy training helper for each optimizer.
    adam_weights, adam_history = _train_noisy_with_optimizer(
        ...,
        metric_seed=metric_seed,
        lr=0.08,
        eps=lr,
        batch_size=12,
    )
    cobyla_weights, cobyla_history = _train_noisy_with_optimizer(
        ...,
        metric_seed=metric_seed,
    )
    spsa_weights, spsa_history = _train_noisy_with_optimizer(
        ...,
        metric_seed=metric_seed,
        lr=lr,
        perturbation=0.18,
        batch_size=12,
    )

    # Step 5: evaluate each trained weight vector on train/validation/test so
    # the final comparison uses the same reporting logic for all optimizers.
    def summarize(weights: np.ndarray, evaluation_seed: int) -> dict[str, object]:
        rng = np.random.default_rng(int(evaluation_seed))
        metrics = _evaluate_noisy_splits(
            prepared_circuit,
            input_params,
            weight_params,
            weights,
            backend,
            rng,
            int(shots),
            {
                "train": (splits["X_train"], splits["y_train"]),
                "val": (splits["X_val"], splits["y_val"]),
                "test": (splits["X_test"], splits["y_test"]),
            },
        )
        test_logits = np.asarray(metrics["test"]["logits"], dtype=float).reshape(-1)
        test_predictions = (test_logits >= 0.0).astype(int)
        return {
            "train_accuracy": float(metrics["train"]["accuracy"]),
            "val_accuracy": float(metrics["val"]["accuracy"]),
            "test_accuracy": float(metrics["test"]["accuracy"]),
            "test_predictions": test_predictions,
        }

    # Step 6: choose the best optimizer from validation accuracy, then expose
    # its locked test accuracy / predictions in the returned dictionary.
    summary_seed = int(seed) + 20000
    adam_summary = summarize(adam_weights, summary_seed)
    cobyla_summary = summarize(cobyla_weights, summary_seed)
    spsa_summary = summarize(spsa_weights, summary_seed)

    validation_scores = {
        "adam": ...,
        "cobyla": ...,
        "spsa": ...,
    }
    best_optimizer = ...
    best_test_accuracy = {
        "adam": adam_summary["test_accuracy"],
        "cobyla": cobyla_summary["test_accuracy"],
        "spsa": spsa_summary["test_accuracy"],
    }[best_optimizer]
    best_test_predictions = {
        "adam": adam_summary["test_predictions"],
        "cobyla": cobyla_summary["test_predictions"],
        "spsa": spsa_summary["test_predictions"],
    }[best_optimizer]

    # Step 7: when the PCA representation is 2D, optionally evaluate the best
    # noisy model on a coarse plane grid so the tests can draw decision regions.
    decision_payload: dict[str, np.ndarray] = {}
    if int(pca_components) == 2:
        X_reference = np.vstack([splits["X_train"], splits["X_val"], splits["X_test"]])
        grid_x, grid_y, grid_points = _make_decision_plane_grid(
            X_reference, resolution=28
        )
        best_weights = {
            "adam": ...,
            "cobyla": ...,
            "spsa": ...,
        }[best_optimizer]
        grid_logits = _forward_noisy_logits(
            prepared_circuit,
            input_params,
            weight_params,
            grid_points,
            best_weights,
            backend,
            np.random.default_rng(int(seed) + 30000),
            int(shots),
        )
        decision_payload = {
            "decision_grid_x": grid_x.astype(float),
            "decision_grid_y": grid_y.astype(float),
            "decision_grid_predictions": (grid_logits >= 0.0)
            .astype(int)
            .reshape(grid_x.shape),
        }

    # Step 8: return exactly one dictionary with per-optimizer histories and
    # summary metrics, plus the best locked test result and optional 2D grid.
    return {
        "adam": {
            **adam_history,
            **{
                key: float(value)
                for key, value in adam_summary.items()
                if key != "test_predictions"
            },
        },
        "cobyla": {
            **cobyla_history,
            **{
                key: float(value)
                for key, value in cobyla_summary.items()
                if key != "test_predictions"
            },
        },
        "spsa": {
            **spsa_history,
            **{
                key: float(value)
                for key, value in spsa_summary.items()
                if key != "test_predictions"
            },
        },
        "best_optimizer": ...,
        "best_test_accuracy": ...,
        "X_test": ...,
        "y_test": ...,
        "best_test_predictions": ...,
        **decision_payload,
    }
