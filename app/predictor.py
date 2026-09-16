"""Production inference wrapper for ProcessGAT-XL."""

import json
from pathlib import Path

import torch

from src.graph_builder import trace_to_graph
from src.model import ProcessGAT


class NextActivityPredictor:
    def __init__(self, checkpoint_dir="checkpoints", device=None):
        self.checkpoint_dir = Path(checkpoint_dir)

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = device

        vocab_path = self.checkpoint_dir / "vocab.json"
        checkpoint_path = self.checkpoint_dir / "best_model.pt"

        if not vocab_path.exists():
            raise FileNotFoundError(vocab_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        with open(vocab_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.activities = metadata["activities"]
        self.vocab = {
            str(k): int(v)
            for k, v in metadata["vocab"].items()
        }

        self.min_amount = float(
            metadata.get("min_amount", 0.0)
        )
        self.max_amount = float(
            metadata.get("max_amount", 1.0)
        )
        self.max_elapsed = float(
            metadata.get("max_elapsed", 1.0)
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        config = checkpoint["model_config"]

        self.model = ProcessGAT(
            in_channels=int(config["in_channels"]),
            hidden=int(config["hidden"]),
            num_classes=int(checkpoint["num_classes"]),
            heads=int(config["heads"]),
            edge_dim=int(config["edge_dim"]),
            dropout=float(config["dropout"]),
            num_layers=int(config.get("num_layers", 3)),
        )

        self.model.load_state_dict(
            checkpoint["model_state"]
        )
        self.model.to(self.device)
        self.model.eval()

        self.metadata = {
            "checkpoint_epoch": checkpoint["epoch"],
            "val_macro_f1": checkpoint.get(
                "val_macro_f1"
            ),
            "val_top1": checkpoint.get(
                "val_top1"
            ),
            "model_config": config,
        }

    @torch.inference_mode()
    def predict(self, trace, top_k=3):
        if not isinstance(trace, list) or not trace:
            raise ValueError(
                "trace must be a non-empty list."
            )

        graph = trace_to_graph(
            trace,
            self.vocab,
            max_amount=self.max_amount,
            min_amount=self.min_amount,
            max_elapsed=self.max_elapsed,
        ).to(self.device)

        batch = torch.zeros(
            graph.num_nodes,
            dtype=torch.long,
            device=self.device,
        )

        logits = self.model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            batch,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]

        k = min(
            int(top_k),
            len(self.activities),
        )

        values, indices = torch.topk(
            probabilities,
            k=k,
        )

        return [
            {
                "activity": self.activities[int(index)],
                "probability": round(
                    float(value),
                    6,
                ),
            }
            for value, index in zip(
                values.cpu(),
                indices.cpu(),
            )
        ]
