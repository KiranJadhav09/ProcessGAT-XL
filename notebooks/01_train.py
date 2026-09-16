"""
Industry-grade ProcessGAT training pipeline.

Run from project root:
    python notebooks/01_train.py

The upgraded pipeline uses case-level splitting, so prefixes from the same
case cannot leak across train/validation/test.
"""

import json
import os
import sys
import warnings

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pm4py

from src.graph_builder import (
    build_vocab,
    log_to_prefix_graphs,
    normalise_amount,
    max_elapsed_from_log,
)
from src.model import ProcessGAT
from src.train import train, seed_everything


SEED = 42
LOG_PATH = "data/BPIC12_W.xes"
CHECKPOINT_DIR = "checkpoints"

# Strong candidate configuration. Tune before claiming these are optimal.
MODEL_CONFIG = {
    "in_channels": 6,
    "hidden": 64,
    "heads": 4,
    "edge_dim": 16,
    "dropout": 0.20,
    "num_layers": 3,
}

TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "warmup_epochs": 5,
    "patience": 15,
    "seed": SEED,
}


seed_everything(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Loading BPIC 2012 W log...")
log_df = pm4py.read_xes(LOG_PATH)

print(
    f"Cases: {log_df['case:concept:name'].nunique():,} | "
    f"Events: {len(log_df):,}"
)

activities, vocab = build_vocab(log_df)
num_classes = len(activities)

print(f"Activities ({num_classes}):")
for i, activity in enumerate(activities):
    print(f"  {i}: {activity}")

min_amount, max_amount = normalise_amount(log_df)
max_elapsed = max_elapsed_from_log(log_df)

with open(
    os.path.join(CHECKPOINT_DIR, "vocab.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "activities": activities,
            "vocab": vocab,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "max_elapsed": max_elapsed,
        },
        f,
        indent=2,
    )

print("\nBuilding prefix graphs...")
graphs = log_to_prefix_graphs(log_df, vocab)

print(f"Total prefix samples: {len(graphs):,}")

if not graphs:
    raise RuntimeError("No prefix graphs were generated.")

print(
    f"Sample graph | nodes={graphs[0].num_nodes} "
    f"edges={graphs[0].num_edges} "
    f"features={graphs[0].x.shape[1]} "
    f"label={activities[int(graphs[0].y.item())]} "
    f"case_id={int(graphs[0].case_id.item())}"
)

model = ProcessGAT(
    num_classes=num_classes,
    **MODEL_CONFIG,
)

print(
    f"\nProcessGAT-XL parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

history, test, split_info = train(
    model,
    graphs,
    num_classes=num_classes,
    checkpoint_dir=CHECKPOINT_DIR,
    **TRAIN_CONFIG,
)

results = {
    "dataset": "BPIC 2012 W",
    "model": "ProcessGAT-XL",
    "model_config": MODEL_CONFIG,
    "training_config": TRAIN_CONFIG,
    "split": "case-level",
    "split_info": split_info,
    "test_top1": test["top1"],
    "test_top3": test["top3"],
    "test_macro_f1": test["macro_f1"],
    "test_weighted_f1": test["weighted_f1"],
    "num_classes": num_classes,
    "activities": activities,
    "total_prefix_samples": len(graphs),
}

with open(
    os.path.join(CHECKPOINT_DIR, "results.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(results, f, indent=2)

epochs = [h["epoch"] for h in history]
losses = [h["loss"] for h in history]
top1 = [h["val_top1"] for h in history]
top3 = [h["val_top3"] for h in history]
f1 = [h["val_macro_f1"] for h in history]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, losses)
ax.set_title("ProcessGAT-XL Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    os.path.join(CHECKPOINT_DIR, "loss_curve.png"),
    dpi=180,
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, top1, label="Top-1")
ax.plot(epochs, top3, label="Top-3")
ax.plot(epochs, f1, label="Macro-F1")
ax.set_title("ProcessGAT-XL Validation Metrics")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    os.path.join(CHECKPOINT_DIR, "validation_metrics.png"),
    dpi=180,
)
plt.close(fig)

print("\nTraining complete.")
print(
    f"TEST | Top-1={test['top1']:.4f} | "
    f"Top-3={test['top3']:.4f} | "
    f"Macro-F1={test['macro_f1']:.4f} | "
    f"Weighted-F1={test['weighted_f1']:.4f}"
)
print(
    "\nNote: these metrics are not directly comparable with the old "
    "prefix-level split because the upgraded pipeline uses case-level "
    "evaluation to prevent cross-case prefix leakage."
)
