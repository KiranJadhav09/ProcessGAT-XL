"""
Research/production training utilities for ProcessGAT.

Key improvements over the baseline:
  * case-level train/validation/test split to avoid prefix leakage
  * reproducible seeding
  * effective-number class weighting
  * label smoothing
  * AdamW + warmup/cosine schedule
  * gradient clipping
  * early stopping
  * best-checkpoint selection
  * per-class metrics and confusion matrix
  * JSON-serializable training history
"""

from __future__ import annotations

import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_case(graphs, test_size=0.15, val_size=0.15, seed=42):
    """Split graphs by case_id, never by individual prefixes."""

    groups = []
    for g in graphs:
        if not hasattr(g, "case_id"):
            raise ValueError(
                "Graphs do not contain case_id. "
                "Use the upgraded graph_builder.py to rebuild graphs."
            )
        groups.append(int(g.case_id.view(-1)[0].item()))

    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("Need at least 3 distinct cases.")

    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=seed,
    )

    relative_val = val_size / (1.0 - test_size)

    train_groups, val_groups = train_test_split(
        train_groups,
        test_size=relative_val,
        random_state=seed,
    )

    train_set = set(map(int, train_groups))
    val_set = set(map(int, val_groups))
    test_set = set(map(int, test_groups))

    train = [g for g, c in zip(graphs, groups) if c in train_set]
    val = [g for g, c in zip(graphs, groups) if c in val_set]
    test = [g for g, c in zip(graphs, groups) if c in test_set]

    return train, val, test


def effective_class_weights(graphs, num_classes, beta=0.9999):
    counts = np.zeros(num_classes, dtype=np.float64)
    for g in graphs:
        counts[int(g.y.view(-1)[0].item())] += 1

    effective = 1.0 - np.power(beta, counts)
    weights = np.where(
        counts > 0,
        (1.0 - beta) / np.maximum(effective, 1e-12),
        0.0,
    )

    weights = weights / max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def top_k_accuracy(logits, labels, k=3):
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    return topk.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.view(-1).cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=1)

    return {
        "top1": float((preds == labels).float().mean().item()),
        "top3": float(top_k_accuracy(logits, labels, 3)),
        "macro_f1": float(
            f1_score(
                labels.numpy(),
                preds.numpy(),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels.numpy(),
                preds.numpy(),
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            labels.numpy(),
            preds.numpy(),
            labels=list(range(num_classes)),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels.numpy(),
            preds.numpy(),
            labels=list(range(num_classes)),
        ).tolist(),
    }


def _scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(warmup_epochs, 1)

        progress = (epoch - warmup_epochs) / max(
            epochs - warmup_epochs - 1, 1
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


def train(
    model,
    graphs,
    num_classes,
    epochs=100,
    batch_size=256,
    lr=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.05,
    warmup_epochs=5,
    patience=15,
    checkpoint_dir="checkpoints",
    seed=42,
    device=None,
):
    seed_everything(seed)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_graphs, val_graphs, test_graphs = split_by_case(
        graphs,
        seed=seed,
    )

    print(
        f"Case-level split | train={len(train_graphs):,} "
        f"val={len(val_graphs):,} test={len(test_graphs):,}"
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    weights = effective_class_weights(
        train_graphs,
        num_classes,
    ).to(device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = _scheduler(
        optimizer,
        epochs,
        warmup_epochs,
    )

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            loss = F.cross_entropy(
                logits,
                batch.y.view(-1),
                weight=weights,
                label_smoothing=label_smoothing,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += float(loss.item())

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        val = evaluate(
            model,
            val_loader,
            device,
            num_classes,
        )

        # Primary selection metric: macro-F1; top-1 is retained as a
        # secondary metric because the task is next-event classification.
        score = val["macro_f1"]

        row = {
            "epoch": epoch,
            "loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "val_macro_f1": val["macro_f1"],
            "val_weighted_f1": val["weighted_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"top1={val['top1']:.4f} | "
            f"top3={val['top3']:.4f} | "
            f"macroF1={val['macro_f1']:.4f} | "
            f"lr={row['lr']:.2e}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0

            torch.save(
                {
                    "model_state": best_state,
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "val_macro_f1": val["macro_f1"],
                    "val_top1": val["top1"],
                    "model_config": {
                        "in_channels": model.in_channels,
                        "hidden": model.hidden,
                        "heads": model.heads,
                        "edge_dim": model.edge_dim,
                        "dropout": model.dropout,
                        "num_layers": model.num_layers,
                    },
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "label_smoothing": label_smoothing,
                        "warmup_epochs": warmup_epochs,
                        "patience": patience,
                        "seed": seed,
                        "split": "case-level",
                    },
                },
                os.path.join(
                    checkpoint_dir,
                    "best_model.pt",
                ),
            )
        else:
            stale += 1

        if stale >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")

    model.load_state_dict(best_state)

    test = evaluate(
        model,
        test_loader,
        device,
        num_classes,
    )

    with open(
        os.path.join(checkpoint_dir, "history.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(history, f, indent=2)

    with open(
        os.path.join(checkpoint_dir, "test_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(test, f, indent=2)

    return history, test, {
        "best_epoch": best_epoch,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
    }
