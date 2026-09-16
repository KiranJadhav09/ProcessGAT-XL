"""
Lightweight hyperparameter search for ProcessGAT-XL.

Usage:
    python scripts/tune.py

This intentionally tunes on a reduced subset and short runs first.
After selecting a configuration, run the full 100-epoch training pipeline.

The script reports the best validation Macro-F1 found; it does not claim
global optimality.
"""

import itertools
import os
import sys
import random

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

import pm4py

from src.graph_builder import build_vocab, log_to_prefix_graphs
from src.model import ProcessGAT
from src.train import train, seed_everything


LOG_PATH = "data/BPIC12_W.xes"
SEED = 42

# Keep the first search CPU-feasible. Expand only after this works.
SEARCH = {
    "hidden": [48, 64],
    "heads": [4, 8],
    "edge_dim": [8, 16],
    "dropout": [0.15, 0.25],
    "lr": [2e-4, 3e-4],
}

TRIAL_EPOCHS = 12
TRIAL_PATIENCE = 4
BATCH_SIZE = 256
MAX_CASES = 2500

seed_everything(SEED)

print("Loading data...")
log_df = pm4py.read_xes(LOG_PATH)
activities, vocab = build_vocab(log_df)

print(f"Building graphs from first {MAX_CASES} cases...")
graphs = log_to_prefix_graphs(
    log_df,
    vocab,
    max_cases=MAX_CASES,
)

print(f"Tuning graphs: {len(graphs):,}")

keys = list(SEARCH)
configs = [
    dict(zip(keys, values))
    for values in itertools.product(
        *(SEARCH[key] for key in keys)
    )
]

random.Random(SEED).shuffle(configs)

# Limit initial search; remove this cap for a larger sweep.
configs = configs[:12]

best = None

for trial_id, config in enumerate(configs, start=1):
    print("\n" + "=" * 72)
    print(f"TRIAL {trial_id}/{len(configs)}: {config}")

    model = ProcessGAT(
    in_channels=6,
    num_classes=len(activities),
    num_layers=3,
    hidden=config["hidden"],
    heads=config["heads"],
    edge_dim=config["edge_dim"],
    dropout=config["dropout"],
)

    _, test_like, _ = train(
        model,
        graphs,
        num_classes=len(activities),
        epochs=TRIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=config["lr"],
        weight_decay=1e-4,
        label_smoothing=0.05,
        warmup_epochs=2,
        patience=TRIAL_PATIENCE,
        checkpoint_dir=f"checkpoints/tuning_trial_{trial_id}",
        seed=SEED,
    )

    score = test_like["macro_f1"]

    print(
        f"TRIAL {trial_id} Macro-F1: {score:.4f}"
    )

    if best is None or score > best["score"]:
        best = {
            "score": score,
            "config": config,
        }

print("\nBEST CONFIG FOUND")
print(best)
print(
    "\nThis is the best configuration among the tested trials, "
    "not a proof of global optimality."
)
