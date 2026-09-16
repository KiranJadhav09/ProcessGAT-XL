#!/usr/bin/env python3
"""
Final deep training pipeline for BPIC12_W ProcessGAT-XL.

Run from project root:

    python scripts/train_final.py

Expected project structure:

    GNN/
    ├── data/
    │   └── BPIC12_W.xes
    ├── src/
    │   ├── graph_builder.py
    │   ├── model.py
    │   └── train.py
    ├── checkpoints/
    └── scripts/
        └── train_final.py

Outputs:

    artifacts/final_experiment/
    ├── checkpoints/
    │   ├── best_model.pt
    │   └── final_model.pt
    ├── metrics/
    │   ├── test_metrics.json
    │   ├── training_history.json
    │   ├── experiment_config.json
    │   └── dataset_statistics.json
    ├── plots/
    │   ├── training_loss.pdf
    │   ├── training_loss.png
    │   ├── validation_macro_f1.pdf
    │   ├── validation_macro_f1.png
    │   ├── validation_topk.pdf
    │   ├── validation_topk.png
    │   ├── confusion_matrix.pdf
    │   ├── confusion_matrix.png
    │   ├── normalized_confusion_matrix.pdf
    │   ├── normalized_confusion_matrix.png
    │   ├── per_class_f1.pdf
    │   └── per_class_f1.png
    └── data/
        └── vocab.json
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pm4py
import torch
from sklearn.metrics import confusion_matrix, f1_score

from torch_geometric.loader import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.graph_builder import build_vocab, log_to_prefix_graphs
from src.model import ProcessGAT
from src.train import train


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = ROOT / "data" / "BPIC12_W.xes"

OUTPUT_DIR = ROOT / "artifacts" / "final_experiment"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"
ARCHITECTURE_DIR = OUTPUT_DIR / "architecture"
DATA_DIR = OUTPUT_DIR / "data"


# ------------------------------------------------------------
# FINAL MODEL CONFIGURATION
# ------------------------------------------------------------

MODEL_CONFIG = {
    "in_channels": 6,
    "hidden": 64,
    "num_layers": 3,
    "heads": 8,
    "edge_dim": 8,
    "dropout": 0.25,
}


# ------------------------------------------------------------
# TRAINING CONFIGURATION
# ------------------------------------------------------------

TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 256,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "warmup_epochs": 5,
    "patience": 20,
    "seed": 42,
}


# ============================================================
# IEEE PLOT SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories() -> None:
    """Create final experiment directories."""

    for directory in [
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        ARCHITECTURE_DIR,
        DATA_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data) -> None:
    """Save JSON with readable formatting."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_figure(fig, name: str) -> None:
    """Save both vector PDF and high-resolution PNG."""

    pdf_path = PLOTS_DIR / f"{name}.pdf"
    png_path = PLOTS_DIR / f"{name}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        format="pdf",
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        format="png",
        dpi=300,
    )

    plt.close(fig)


def print_section(title: str) -> None:
    """Print a clear terminal section."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_dataset_statistics(graphs, activities):
    """Calculate useful graph-level statistics."""

    num_graphs = len(graphs)

    node_counts = []
    edge_counts = []
    labels = []
    case_ids = set()

    for graph in graphs:

        node_counts.append(int(graph.num_nodes))

        if graph.edge_index is not None:
            edge_counts.append(int(graph.edge_index.shape[1]))
        else:
            edge_counts.append(0)

        if hasattr(graph, "y"):
            labels.append(int(graph.y.item()))

        if hasattr(graph, "case_id"):
            case_value = graph.case_id

            if torch.is_tensor(case_value):
                case_value = case_value.item()

            case_ids.add(str(case_value))

    label_counts = {}

    for label in labels:
        activity = activities[label]
        label_counts[activity] = label_counts.get(activity, 0) + 1

    statistics = {
        "num_graphs": num_graphs,
        "num_unique_cases": len(case_ids),
        "num_activities": len(activities),
        "activities": activities,
        "nodes": {
            "mean": float(np.mean(node_counts)),
            "std": float(np.std(node_counts)),
            "min": int(np.min(node_counts)),
            "max": int(np.max(node_counts)),
        },
        "edges": {
            "mean": float(np.mean(edge_counts)),
            "std": float(np.std(edge_counts)),
            "min": int(np.min(edge_counts)),
            "max": int(np.max(edge_counts)),
        },
        "label_distribution": label_counts,
    }

    return statistics


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_loss(history):
    """Plot training loss."""

    epochs = [row["epoch"] for row in history]
    loss = [row["loss"] for row in history]

    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    ax.plot(
        epochs,
        loss,
        linewidth=1.5,
        label="Training Loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.35,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    save_figure(fig, "training_loss")


def plot_validation_macro_f1(history):
    """Plot validation Macro-F1."""

    epochs = [row["epoch"] for row in history]
    f1_values = [row["val_macro_f1"] for row in history]

    best_index = int(np.argmax(f1_values))

    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    ax.plot(
        epochs,
        f1_values,
        linewidth=1.5,
        label="Validation Macro-F1",
    )

    ax.scatter(
        [epochs[best_index]],
        [f1_values[best_index]],
        s=25,
        zorder=3,
        label=f"Best = {f1_values[best_index]:.3f}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation Macro-F1")

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.35,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    save_figure(fig, "validation_macro_f1")


def plot_validation_topk(history):
    """Plot Top-1 and Top-3 validation accuracy."""

    epochs = [row["epoch"] for row in history]

    top1 = [row["val_top1"] for row in history]
    top3 = [row["val_top3"] for row in history]

    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    ax.plot(
        epochs,
        top1,
        linewidth=1.5,
        label="Top-1",
    )

    ax.plot(
        epochs,
        top3,
        linewidth=1.5,
        linestyle="--",
        label="Top-3",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Top-k Accuracy")

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.35,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    save_figure(fig, "validation_topk")


# ============================================================
# TEST PREDICTIONS
# ============================================================

@torch.no_grad()
def collect_predictions(model, graphs):
    """Collect labels and predictions from graphs."""

    device = next(model.parameters()).device

    loader = DataLoader(
        graphs,
        batch_size=512,
        shuffle=False,
    )

    model.eval()

    all_true = []
    all_pred = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        predictions = logits.argmax(dim=1)

        all_true.extend(
            batch.y.view(-1).cpu().numpy().tolist()
        )

        all_pred.extend(
            predictions.cpu().numpy().tolist()
        )

    return np.asarray(all_true), np.asarray(all_pred)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, y_pred, activities):
    """Plot raw confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(activities))),
    )

    fig, ax = plt.subplots(figsize=(4.1, 3.45))

    image = ax.imshow(cm)

    ax.set_xticks(range(len(activities)))
    ax.set_yticks(range(len(activities)))

    ax.set_xticklabels(
        [f"{i}" for i in range(len(activities))]
    )

    ax.set_yticklabels(
        [f"{i}" for i in range(len(activities))]
    )

    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("True Activity")

    ax.set_title("Test Confusion Matrix")

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    save_figure(fig, "confusion_matrix")


def plot_normalized_confusion_matrix(
    y_true,
    y_pred,
    activities,
):
    """Plot row-normalized confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(activities))),
    ).astype(float)

    row_sums = cm.sum(axis=1, keepdims=True)

    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(4.1, 3.45))

    image = ax.imshow(normalized, vmin=0, vmax=1)

    ax.set_xticks(range(len(activities)))
    ax.set_yticks(range(len(activities)))

    ax.set_xticklabels(
        [f"{i}" for i in range(len(activities))]
    )

    ax.set_yticklabels(
        [f"{i}" for i in range(len(activities))]
    )

    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("True Activity")

    ax.set_title("Normalized Test Confusion Matrix")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):

            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    save_figure(
        fig,
        "normalized_confusion_matrix",
    )


# ============================================================
# PER-CLASS F1
# ============================================================

def plot_per_class_f1(
    y_true,
    y_pred,
    activities,
):
    """Plot per-class F1 scores."""

    scores = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(activities))),
        average=None,
        zero_division=0,
    )

    x = np.arange(len(activities))

    fig, ax = plt.subplots(figsize=(4.1, 2.8))

    ax.bar(
        x,
        scores,
        width=0.65,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in range(len(activities))]
    )

    ax.set_xlabel("Activity Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class Test F1 Score")

    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        linewidth=0.4,
        alpha=0.35,
    )

    for i, value in enumerate(scores):

        ax.text(
            i,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()

    save_figure(fig, "per_class_f1")


# ============================================================
# MODEL ARCHITECTURE FIGURE
# ============================================================

def create_architecture_figure():
    """
    Create a clean IEEE-style conceptual architecture diagram.

    This is a visual documentation figure, not a computation graph.
    """

    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    ax.axis("off")

    boxes = [
        (0.2, 1.25, 1.45, 1.25, "Process Prefix\nGraph"),
        (2.05, 1.25, 1.55, 1.25, "Node + Edge\nFeatures"),
        (4.00, 1.25, 1.55, 1.25, "GATv2\nBlock 1"),
        (5.95, 1.25, 1.55, 1.25, "GATv2\nBlock 2"),
        (7.90, 1.25, 1.55, 1.25, "GATv2\nBlock 3"),
        (9.85, 1.25, 1.55, 1.25, "Attention\nPooling"),
    ]

    # Because the last box extends beyond the nominal axis,
    # expand the canvas slightly.
    ax.set_xlim(0, 11.7)

    for x, y, w, h, label in boxes:

        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=1.1,
        )

        ax.add_patch(rectangle)

        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=8,
        )

    for i in range(len(boxes) - 1):

        x1, y1, w1, h1, _ = boxes[i]
        x2, y2, _, h2, _ = boxes[i + 1]

        ax.annotate(
            "",
            xy=(x2, y2 + h2 / 2),
            xytext=(x1 + w1, y1 + h1 / 2),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.0,
            ),
        )

    ax.text(
        5.85,
        3.25,
        "ProcessGAT-XL",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        5.85,
        0.55,
        "Residual GATv2 layers • LayerNorm • GELU • Multi-scale graph pooling",
        ha="center",
        va="center",
        fontsize=7.5,
    )

    fig.tight_layout()

    path_pdf = ARCHITECTURE_DIR / "processgat_xl_architecture.pdf"
    path_svg = ARCHITECTURE_DIR / "processgat_xl_architecture.svg"
    path_png = ARCHITECTURE_DIR / "processgat_xl_architecture.png"

    fig.savefig(
        path_pdf,
        bbox_inches="tight",
        format="pdf",
    )

    fig.savefig(
        path_svg,
        bbox_inches="tight",
        format="svg",
    )

    fig.savefig(
        path_png,
        bbox_inches="tight",
        format="png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print_section("FINAL PROCESSGAT-XL TRAINING")

    print(f"Project root : {ROOT}")
    print(f"Dataset      : {DATA_PATH}")
    print(f"Output       : {OUTPUT_DIR}")

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Make sure data/BPIC12_W.xes exists."
        )

    ensure_directories()

    # --------------------------------------------------------
    # SAVE EXPERIMENT CONFIG
    # --------------------------------------------------------

    experiment_config = {
        "experiment": "BPIC12_W_ProcessGAT_XL_Final",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "model": MODEL_CONFIG,
        "training": TRAIN_CONFIG,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "protocol": {
            "split": "case-level",
            "validation_selection_metric": "Macro-F1",
            "test_used_for_model_selection": False,
            "random_seed": TRAIN_CONFIG["seed"],
        },
        "note": (
            "The 64/8/8/0.25/2e-4 configuration was "
            "identified during exploratory tuning and is "
            "treated here as a candidate configuration."
        ),
    }

    save_json(
        METRICS_DIR / "experiment_config.json",
        experiment_config,
    )

    # --------------------------------------------------------
    # LOAD XES
    # --------------------------------------------------------

    print_section("1. LOADING BPIC12_W")

    print("Reading XES log...")

    log_df = pm4py.read_xes(
        str(DATA_PATH)
    )

    print(
        f"Events loaded: {len(log_df):,}"
    )

    # --------------------------------------------------------
    # BUILD VOCABULARY
    # --------------------------------------------------------

    print_section("2. BUILDING ACTIVITY VOCABULARY")

    activities, vocab = build_vocab(log_df)

    print(
        f"Activities: {len(activities)}"
    )

    for idx, activity in enumerate(activities):
        print(
            f"  {idx}: {activity}"
        )

    # Save vocabulary.

    vocab_output = {
        "activities": activities,
        "vocab": vocab,
    }

    save_json(
        DATA_DIR / "vocab.json",
        vocab_output,
    )

    # --------------------------------------------------------
    # BUILD PREFIX GRAPHS
    # --------------------------------------------------------

    print_section("3. BUILDING PREFIX GRAPHS")

    print(
        "Generating prefix-level process graphs..."
    )

    graph_start = time.time()

    graphs = log_to_prefix_graphs(
        log_df,
        vocab,
        max_cases=None,
    )

    graph_time = time.time() - graph_start

    print(
        f"Graphs generated: {len(graphs):,}"
    )

    print(
        f"Graph generation time: "
        f"{graph_time / 60:.2f} minutes"
    )

    if len(graphs) == 0:
        raise RuntimeError(
            "No graphs were generated."
        )

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    print_section("4. DATASET STATISTICS")

    statistics = calculate_dataset_statistics(
        graphs,
        activities,
    )

    save_json(
        METRICS_DIR / "dataset_statistics.json",
        statistics,
    )

    print(
        f"Total graphs : "
        f"{statistics['num_graphs']:,}"
    )

    print(
        f"Unique cases : "
        f"{statistics['num_unique_cases']:,}"
    )

    print(
        f"Activities   : "
        f"{statistics['num_activities']}"
    )

    print(
        f"Mean nodes   : "
        f"{statistics['nodes']['mean']:.2f}"
    )

    print(
        f"Mean edges   : "
        f"{statistics['edges']['mean']:.2f}"
    )

    print("\nNext-activity distribution:")

    for activity, count in statistics[
        "label_distribution"
    ].items():

        percentage = (
            100.0
            * count
            / len(graphs)
        )

        print(
            f"  {activity:<35} "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print_section("5. INITIALIZING PROCESSGAT-XL")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = ProcessGAT(
        in_channels=MODEL_CONFIG["in_channels"],
        hidden=MODEL_CONFIG["hidden"],
        num_classes=len(activities),
        heads=MODEL_CONFIG["heads"],
        edge_dim=MODEL_CONFIG["edge_dim"],
        dropout=MODEL_CONFIG["dropout"],
        num_layers=MODEL_CONFIG["num_layers"],
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # SAVE MODEL ARCHITECTURE INFO
    # --------------------------------------------------------

    architecture_info = {
        "model_name": "ProcessGAT-XL",
        "class": "ProcessGAT",
        "parameters": parameter_count,
        "configuration": MODEL_CONFIG,
        "input": {
            "node_features": 6,
            "edge_features": 1,
        },
        "graph_processing": {
            "convolution": "GATv2Conv",
            "layers": MODEL_CONFIG["num_layers"],
            "attention_heads": MODEL_CONFIG["heads"],
            "edge_dimension": MODEL_CONFIG["edge_dim"],
            "dropout": MODEL_CONFIG["dropout"],
            "normalization": "LayerNorm",
            "activation": "GELU",
            "pooling": (
                "learned graph attention + "
                "global mean + global max"
            ),
        },
        "optimizer": "AdamW",
        "scheduler": "warmup + cosine decay",
        "loss": (
            "cross entropy with label smoothing "
            "and effective-number class weighting"
        ),
    }

    save_json(
        ARCHITECTURE_DIR / "architecture.json",
        architecture_info,
    )

    create_architecture_figure()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print_section("6. DEEP TRAINING")

    print("Final configuration:")
    print(
        json.dumps(
            {
                "model": MODEL_CONFIG,
                "training": TRAIN_CONFIG,
            },
            indent=2,
        )
    )

    print()
    print(
        "Starting training..."
    )
    print(
        "This may take a substantial amount of time on CPU."
    )
    print(
        "Do not interrupt the terminal unless necessary."
    )
    print()

    training_start = time.time()

    history, test_metrics, metadata = train(
    graphs=graphs,
    model=model,
    num_classes=len(activities),
    epochs=TRAIN_CONFIG["epochs"],
    batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        label_smoothing=TRAIN_CONFIG["label_smoothing"],
        warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
        patience=TRAIN_CONFIG["patience"],
        checkpoint_dir=str(CHECKPOINT_DIR),
        seed=TRAIN_CONFIG["seed"],
    )

    training_time = time.time() - training_start

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    save_json(
        METRICS_DIR / "training_history.json",
        history,
    )

    # --------------------------------------------------------
    # SAVE TEST METRICS
    # --------------------------------------------------------

    save_json(
        METRICS_DIR / "test_metrics.json",
        test_metrics,
    )

    # --------------------------------------------------------
    # SAVE TRAINING METADATA
    # --------------------------------------------------------

    save_json(
        METRICS_DIR / "training_metadata.json",
        {
            **metadata,
            "training_time_seconds": training_time,
            "training_time_minutes": (
                training_time / 60.0
            ),
        },
    )

    # --------------------------------------------------------
    # COPY BEST MODEL TO FINAL MODEL
    # --------------------------------------------------------

    best_model = CHECKPOINT_DIR / "best_model.pt"

    final_model = CHECKPOINT_DIR / "final_model.pt"

    if best_model.exists():

        shutil.copy2(
            best_model,
            final_model,
        )

        print(
            f"\nFinal model saved to:\n"
            f"{final_model}"
        )

    else:

        print(
            "\nWARNING: best_model.pt was not found."
        )

    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    print_section("7. GENERATING TRAINING FIGURES")

    plot_training_loss(history)

    plot_validation_macro_f1(history)

    plot_validation_topk(history)

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print_section("8. EVALUATING BEST MODEL")

    if not best_model.exists():

        raise FileNotFoundError(
            "best_model.pt was not generated."
        )

    checkpoint = torch.load(
        best_model,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(device)

    # --------------------------------------------------------
    # RECREATE CASE-LEVEL TEST SPLIT
    # --------------------------------------------------------
    #
    # The training module uses the deterministic seed to
    # construct the case-level split. We recreate that split
    # here so the final figures use the exact held-out test
    # cases used by train().
    # --------------------------------------------------------

    from src.train import split_by_case

    train_graphs, val_graphs, test_graphs = split_by_case(
        graphs,
        seed=TRAIN_CONFIG["seed"],
    )

    print(
        f"Train graphs: "
        f"{len(train_graphs):,}"
    )

    print(
        f"Validation graphs: "
        f"{len(val_graphs):,}"
    )

    print(
        f"Test graphs: "
        f"{len(test_graphs):,}"
    )

    y_true, y_pred = collect_predictions(
        model,
        test_graphs,
    )

    # --------------------------------------------------------
    # TEST FIGURES
    # --------------------------------------------------------

    plot_confusion_matrix(
        y_true,
        y_pred,
        activities,
    )

    plot_normalized_confusion_matrix(
        y_true,
        y_pred,
        activities,
    )

    plot_per_class_f1(
        y_true,
        y_pred,
        activities,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_time = time.time() - total_start

    print_section("FINAL TRAINING COMPLETE")

    print(
        f"Best epoch : "
        f"{metadata.get('best_epoch', 'N/A')}"
    )

    print(
        f"Test Top-1 : "
        f"{test_metrics.get('top1', 'N/A')}"
    )

    print(
        f"Test Top-3 : "
        f"{test_metrics.get('top3', 'N/A')}"
    )

    print(
        f"Test Macro-F1 : "
        f"{test_metrics.get('macro_f1', 'N/A')}"
    )

    print(
        f"Test Weighted-F1 : "
        f"{test_metrics.get('weighted_f1', 'N/A')}"
    )

    print(
        f"\nTraining time: "
        f"{training_time / 60:.2f} minutes"
    )

    print(
        f"Total pipeline time: "
        f"{total_time / 60:.2f} minutes"
    )

    print("\nArtifacts generated in:")

    print(
        OUTPUT_DIR
    )

    print("\nImportant files:")

    print(
        f"  Model       : "
        f"{CHECKPOINT_DIR / 'final_model.pt'}"
    )

    print(
        f"  Metrics     : "
        f"{METRICS_DIR / 'test_metrics.json'}"
    )

    print(
        f"  History     : "
        f"{METRICS_DIR / 'training_history.json'}"
    )

    print(
        f"  Architecture: "
        f"{ARCHITECTURE_DIR / 'processgat_xl_architecture.pdf'}"
    )

    print(
        f"  Plots       : "
        f"{PLOTS_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
