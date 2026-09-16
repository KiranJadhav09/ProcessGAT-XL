"""
ProcessGAT-XL: stronger graph-attention architecture for process forecasting.

The model consumes prefix graphs with:
  node features: 6
  edge features: 1 transition-frequency weight

Architecture:
  EdgeEncoder
      -> 3 x GATv2Conv blocks with residual connections + LayerNorm
      -> learned graph attention pooling + mean/max pooling
      -> MLP classifier

This is a candidate production/research configuration. "Best" hyperparameters
must be established empirically through the supplied tuning script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


class GraphAttentionPooling(nn.Module):
    """Learned attention pooling over nodes in each graph."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, batch):
        scores = self.score(x).squeeze(-1)
        weights = torch.zeros_like(scores)

        for graph_id in batch.unique(sorted=True):
            mask = batch == graph_id
            weights[mask] = torch.softmax(scores[mask], dim=0)

        pooled = x * weights.unsqueeze(-1)
        return global_add_pool(pooled, batch)


class ResidualGATv2Block(nn.Module):
    def __init__(self, in_channels, out_channels, heads, edge_dim, dropout):
        super().__init__()

        self.conv = GATv2Conv(
            in_channels,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = F.gelu(h)
        h = self.dropout(h)
        return self.norm(h + self.residual(x))


class ProcessGAT(nn.Module):
    """Configurable ProcessGAT-XL model."""

    def __init__(
        self,
        in_channels=6,
        hidden=64,
        num_classes=6,
        heads=4,
        edge_dim=16,
        dropout=0.20,
        num_layers=3,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.in_channels = in_channels
        self.hidden = hidden
        self.num_classes = num_classes
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.edge_encoder = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        blocks = []
        blocks.append(
            ResidualGATv2Block(
                in_channels,
                hidden,
                heads,
                edge_dim,
                dropout,
            )
        )

        for _ in range(num_layers - 1):
            blocks.append(
                ResidualGATv2Block(
                    hidden,
                    hidden,
                    heads,
                    edge_dim,
                    dropout,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.pool_attention = GraphAttentionPooling(hidden)

        # mean + max + learned attention pooling
        pooled_dim = hidden * 3

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_attr is None:
            edge_attr = x.new_ones((edge_index.size(1), 1))
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        edge_attr = self.edge_encoder(edge_attr)

        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        attn_pool = self.pool_attention(x, batch)

        graph_embedding = torch.cat(
            [mean_pool, max_pool, attn_pool],
            dim=-1,
        )

        return self.classifier(graph_embedding)
