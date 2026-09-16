"""Minimal smoke tests for the inference stack."""

import json
from pathlib import Path

import torch

from src.model import ProcessGAT


def test_model_forward_shape():
    model = ProcessGAT(
        in_channels=6,
        hidden=64,
        num_classes=6,
        heads=4,
        edge_dim=16,
        dropout=0.2,
        num_layers=3,
    )

    x = torch.randn(2, 6)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.ones(2, 1)
    batch = torch.zeros(2, dtype=torch.long)

    out = model(x, edge_index, edge_attr, batch)

    assert out.shape == (1, 6)
