"""
Shared test utilities for Assignment 3.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings
from typing import Any

import numpy as np
import pytest
import yaml


@dataclass
class StudentInfo:
    seed: int


# def load_student_info(path_seed: str | Path = "random_seed.yaml") -> StudentInfo:
#     with open(path_seed, "r", encoding="utf-8") as f:
#         data = yaml.safe_load(f)
#     return StudentInfo(seed=int(data["seed"]))

rng_seed = 42


def load_student_info(path_seed: str | Path = "random_seed.yaml") -> StudentInfo:
    global rng_seed
    return StudentInfo(seed=rng_seed)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def require_impl(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(str(exc))


def import_impl(module_name: str) -> Any:
    impl_root = os.environ.get("ASSIGNMENT3_IMPL_ROOT", "src")
    full_name = f"{impl_root}.{module_name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import '{full_name}'. Set ASSIGNMENT3_IMPL_ROOT to a valid package."
        ) from exc


def brute_force_best_qubo(qubo: np.ndarray) -> tuple[np.ndarray, float]:
    qubo = np.asarray(qubo, dtype=float)
    n = qubo.shape[0]
    best_x: np.ndarray | None = None
    best_e = np.inf

    for idx in range(1 << n):
        x = np.array([(idx >> i) & 1 for i in range(n)], dtype=float)
        e = float(np.real(x @ qubo @ x))
        if e < best_e:
            best_e = e
            best_x = x.copy()

    assert best_x is not None
    return best_x.astype(int), float(best_e)


def ensure_output_dir(path: str | Path = "output") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def maybe_publish_reference_artifact(
    source_path: str | Path,
    references_dir: str | Path = "references/",
) -> Path | None:
    if os.environ.get("ASSIGNMENT3_PUBLISH_Q2_REFERENCES", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    source = Path(source_path)
    target_dir = Path(references_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return target


def write_line_plot(
    path: str | Path,
    xs: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    title: str,
    x_label: str = "x",
    y_label: str = "value",
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.asarray(xs, dtype=float).reshape(-1)
    if xs.size < 2:
        raise ValueError("xs must contain at least two points")
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for label, values, color in series:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != xs.shape:
            raise ValueError("each series must have the same shape as xs")
        ax.plot(xs, values, label=label, color=color, linewidth=2.5)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    if y_min is not None or y_max is not None:
        ax.set_ylim(
            y_min if y_min is not None else ax.get_ylim()[0],
            y_max if y_max is not None else ax.get_ylim()[1],
        )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_quantum_circuit_image(
    circuit,
    path: str | Path,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    title = str(title) if title is not None else None
    width_units = max(int(circuit.depth() or 0), int(circuit.size() or 0), 1)
    figure_width = max(12.0, min(30.0, 2.5 + 0.38 * width_units))
    figure_height = max(2.8, 0.42 * circuit.num_qubits + 1.6)

    try:
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r".*qiskit\.circuit\.instruction\.Instruction\.condition.*"
                    r"deprecated as of qiskit 1\.3\.0.*"
                ),
                category=DeprecationWarning,
            )
            circuit.draw(
                output="mpl",
                idle_wires=False,
                fold=-1,
                ax=ax,
            )
        if title:
            fig.suptitle(title)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception:
        pass

    text_diagram = str(circuit.draw(output="text", idle_wires=False, fold=-1))
    longest_line = max((len(line) for line in text_diagram.splitlines()), default=40)
    text_width = max(12.0, min(30.0, 2.0 + 0.11 * longest_line))
    fig, ax = plt.subplots(figsize=(text_width, figure_height))
    ax.axis("off")
    if title:
        ax.set_title(title)
    ax.text(
        0.01,
        0.99,
        text_diagram,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_classification_prediction_plot(
    path: str | Path,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    decision_grid_x: np.ndarray | None = None,
    decision_grid_y: np.ndarray | None = None,
    decision_grid_predictions: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X, dtype=float)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2)")
    if X.shape[0] != y_true.shape[0] or X.shape[0] != y_pred.shape[0]:
        raise ValueError("X, y_true, and y_pred must have the same length")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    colors = ["#1f4e79", "#c44536", "#0b6e4f", "#7f3c8d", "#d17c00"]
    color_map = {
        int(label): colors[idx % len(colors)] for idx, label in enumerate(labels)
    }

    x_margin = 0.08 * max(1e-6, float(np.ptp(X[:, 0])))
    y_margin = 0.08 * max(1e-6, float(np.ptp(X[:, 1])))
    x_limits = (float(np.min(X[:, 0]) - x_margin), float(np.max(X[:, 0]) + x_margin))
    y_limits = (float(np.min(X[:, 1]) - y_margin), float(np.max(X[:, 1]) + y_margin))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    if (
        decision_grid_x is not None
        and decision_grid_y is not None
        and decision_grid_predictions is not None
    ):
        decision_grid_x = np.asarray(decision_grid_x, dtype=float)
        decision_grid_y = np.asarray(decision_grid_y, dtype=float)
        decision_grid_predictions = np.asarray(decision_grid_predictions, dtype=int)
        if (
            decision_grid_x.shape != decision_grid_y.shape
            or decision_grid_x.shape != decision_grid_predictions.shape
        ):
            raise ValueError("decision grid arrays must all have the same shape")
        grid_labels = np.unique(decision_grid_predictions.reshape(-1))
        all_labels = np.unique(np.concatenate([labels, grid_labels]))
    else:
        all_labels = labels

    background_colors = [
        "#a9cce8",
        "#efb0a6",
        "#abd8c4",
        "#d5b4e6",
        "#efc27a",
    ]
    background_cmap = ListedColormap(
        [background_colors[idx % len(background_colors)] for idx, _ in enumerate(all_labels)]
    )
    label_to_index = {int(label): idx for idx, label in enumerate(all_labels)}

    for ax, values, panel_title in (
        (axes[0], y_true, "True labels"),
        (axes[1], y_pred, "Predicted labels"),
    ):
        if (
            decision_grid_x is not None
            and decision_grid_y is not None
            and decision_grid_predictions is not None
        ):
            decision_indices = np.vectorize(lambda value: label_to_index[int(value)])(
                decision_grid_predictions
            )
            ax.contourf(
                decision_grid_x,
                decision_grid_y,
                decision_indices,
                levels=np.arange(len(all_labels) + 1) - 0.5,
                cmap=background_cmap,
                alpha=0.62,
                antialiased=True,
            )
            if len(all_labels) >= 2:
                ax.contour(
                    decision_grid_x,
                    decision_grid_y,
                    decision_indices,
                    levels=np.arange(len(all_labels) - 1) + 0.5,
                    colors="#202020",
                    linewidths=1.4,
                    alpha=0.95,
                )
        for label in labels:
            mask = values == label
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=58,
                color=color_map[int(label)],
                edgecolor="black",
                linewidth=0.65,
                alpha=0.97,
                label=f"class {int(label)}",
            )
        ax.set_title(panel_title)
        ax.set_xlabel("PCA component 1")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    axes[0].set_ylabel("PCA component 2")

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84, top=0.84)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_svg_bloch_frame(
    path: str | Path,
    point: np.ndarray,
    path_points: np.ndarray,
    title: str,
) -> None:
    point = np.asarray(point, dtype=float).reshape(3)
    path_points = np.asarray(path_points, dtype=float).reshape(-1, 3)

    width, height = 520, 520
    cx, cy = width / 2, height / 2 + 10
    radius = 170

    def proj(vec: np.ndarray) -> tuple[float, float]:
        x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
        px = cx + radius * (0.85 * x + 0.35 * y)
        py = cy - radius * (z + 0.25 * y)
        return px, py

    px, py = proj(point)
    path_svg = " ".join(f"{proj(v)[0]:.2f},{proj(v)[1]:.2f}" for v in path_points)
    x_end = proj(np.array([1.0, 0.0, 0.0]))
    y_end = proj(np.array([0.0, 1.0, 0.0]))
    z_end = proj(np.array([0.0, 0.0, 1.0]))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{cx:.1f}" y="32" font-size="24" text-anchor="middle">{title}</text>
<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{radius:.1f}" ry="{0.86 * radius:.1f}" fill="none" stroke="#777" stroke-width="2"/>
<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{0.42 * radius:.1f}" ry="{0.86 * radius:.1f}" fill="none" stroke="#ddd" stroke-width="1.5"/>
<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{radius:.1f}" ry="{0.22 * radius:.1f}" fill="none" stroke="#ddd" stroke-width="1.5"/>
<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x_end[0]:.1f}" y2="{x_end[1]:.1f}" stroke="#1f4e79" stroke-width="3"/>
<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{y_end[0]:.1f}" y2="{y_end[1]:.1f}" stroke="#c44536" stroke-width="3"/>
<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{z_end[0]:.1f}" y2="{z_end[1]:.1f}" stroke="#0b6e4f" stroke-width="3"/>
<text x="{x_end[0] + 12:.1f}" y="{x_end[1] + 4:.1f}" font-size="18" fill="#1f4e79">X</text>
<text x="{y_end[0] + 12:.1f}" y="{y_end[1] + 4:.1f}" font-size="18" fill="#c44536">Y</text>
<text x="{z_end[0] + 8:.1f}" y="{z_end[1] - 8:.1f}" font-size="18" fill="#0b6e4f">Z</text>
<polyline fill="none" stroke="#999" stroke-width="2" points="{path_svg}"/>
<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="#111"/>
</svg>"""
    Path(path).write_text(svg, encoding="utf-8")


def write_svg_bar_frame(
    path: str | Path,
    labels: list[str],
    values: np.ndarray,
    title: str,
) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(labels) != values.size:
        raise ValueError("labels and values must have the same length")

    width, height = 720, 480
    left, right, top, bottom = 70, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / max(values.size, 1) * 0.7
    colors = [
        "#1f4e79",
        "#c44536",
        "#0b6e4f",
        "#7f3c8d",
        "#d17c00",
        "#008080",
        "#555",
        "#999",
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" font-size="24" text-anchor="middle">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black" stroke-width="2"/>',
    ]

    for idx, value in enumerate(values):
        x = left + (idx + 0.15) * plot_w / values.size
        h = plot_h * float(value)
        y = top + plot_h - h
        color = colors[idx % len(colors)]
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" opacity="0.9"/>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24:.1f}" font-size="14" text-anchor="middle">{labels[idx]}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" font-size="12" text-anchor="middle">{value:.2f}</text>'
        )

    for tick in np.linspace(0.0, 1.0, 5):
        y = top + plot_h - plot_h * float(tick)
        lines.append(
            f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="12" text-anchor="end">{tick:.2f}</text>'
        )

    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_gif_from_svg_frames(
    output_path: str | Path,
    frame_writer,
    n_frames: int,
    delay_cs: int = 12,
) -> bool:
    convert = shutil.which("convert") or shutil.which("magick")
    if convert is None:
        return False

    with tempfile.TemporaryDirectory(prefix="assignment3_frames_") as tmpdir:
        frame_paths = []
        for idx in range(int(n_frames)):
            frame_path = Path(tmpdir) / f"frame_{idx:03d}.svg"
            frame_writer(frame_path, idx)
            frame_paths.append(str(frame_path))

        cmd = [
            convert,
            "-delay",
            str(int(delay_cs)),
            "-loop",
            "0",
            *frame_paths,
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    return True


def make_gif_from_matplotlib_frames(
    output_path: str | Path,
    frame_builder,
    n_frames: int,
    delay_cs: int = 12,
) -> bool:
    convert = shutil.which("convert") or shutil.which("magick")
    if convert is None:
        return False

    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory(prefix="assignment3_frames_") as tmpdir:
        frame_paths = []
        for idx in range(int(n_frames)):
            fig = frame_builder(idx)
            frame_path = Path(tmpdir) / f"frame_{idx:03d}.png"
            fig.savefig(frame_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            frame_paths.append(str(frame_path))

        cmd = [
            convert,
            "-delay",
            str(int(delay_cs)),
            "-loop",
            "0",
            *frame_paths,
            str(output_path),
        ]
        subprocess.run(cmd, check=True)
    return True
