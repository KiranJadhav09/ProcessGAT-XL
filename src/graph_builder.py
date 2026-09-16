"""
Graph construction utilities for BPIC 2012 W next-activity prediction.

Training graph:
  - Nodes: unique activities in the observed prefix
  - Edges: directly-follows transitions with normalized frequency weights
  - Node features (6):
      0 activity index
      1 first occurrence position
      2 occurrence frequency
      3 mean elapsed time from case start
      4 resource diversity
      5 case-level amount requested
  - Label: next activity index
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ACTIVITY_COL = "concept:name"
CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
RESOURCE_COL = "org:resource"
AMOUNT_COL = "case:AMOUNT_REQ"


def build_vocab(log_df):
    acts = sorted(log_df[ACTIVITY_COL].dropna().unique().tolist())
    vocab = {activity: i for i, activity in enumerate(acts)}
    return acts, vocab


def normalise_amount(log_df):
    amounts = pd.to_numeric(
        log_df.groupby(CASE_COL)[AMOUNT_COL].first(),
        errors="coerce",
    ).fillna(0.0)

    mn = float(amounts.min())
    mx = float(amounts.max())
    return mn, mx


def max_elapsed_from_log(log_df):
    """Global maximum case duration in seconds."""
    max_elapsed = 1.0

    for _, case_df in log_df.groupby(CASE_COL):
        times = pd.to_datetime(case_df[TIME_COL]).sort_values()
        if len(times) > 1:
            duration = (times.iloc[-1] - times.iloc[0]).total_seconds()
            max_elapsed = max(max_elapsed, float(duration))

    return max_elapsed


def case_to_prefixes(case_df, vocab, max_amount, min_amount, max_elapsed, case_id=None):
    """Generate one PyG graph for every non-terminal prefix of one case."""

    case_df = case_df.sort_values(TIME_COL).reset_index(drop=True)

    events = case_df[ACTIVITY_COL].tolist()
    times = pd.to_datetime(case_df[TIME_COL]).tolist()

    # Missing resources are represented by a stable placeholder.
    resources = (
        case_df[RESOURCE_COL]
        .fillna("__UNKNOWN_RESOURCE__")
        .astype(str)
        .tolist()
    )

    amount_value = pd.to_numeric(
        case_df[AMOUNT_COL].iloc[0],
        errors="coerce",
    )
    amount = 0.0 if pd.isna(amount_value) else float(amount_value)

    norm_amount = (amount - min_amount) / (
        max_amount - min_amount + 1e-8
    )

    n = len(events)
    if n < 2:
        return []

    t0 = times[0]
    elapsed = [(t - t0).total_seconds() for t in times]
    max_e = max_elapsed + 1e-8

    graphs = []
    vocab_size = len(vocab)

    for k in range(1, n):
        prefix_events = events[:k]
        prefix_times = elapsed[:k]
        prefix_resources = resources[:k]
        label = vocab[events[k]]

        seen_pos = {}
        act_times = defaultdict(list)
        act_res = defaultdict(set)
        act_count = defaultdict(int)

        for pos, (act, t, resource) in enumerate(
            zip(prefix_events, prefix_times, prefix_resources)
        ):
            if act not in seen_pos:
                seen_pos[act] = pos
            act_times[act].append(t)
            act_res[act].add(resource)
            act_count[act] += 1

        unique_acts = sorted(
            seen_pos.keys(),
            key=lambda activity: seen_pos[activity],
        )
        local_idx = {
            activity: i for i, activity in enumerate(unique_acts)
        }

        num_nodes = len(unique_acts)
        max_res = max(
            len(act_res[activity]) for activity in unique_acts
        )

        features = []
        for activity in unique_acts:
            f0 = vocab[activity] / (vocab_size - 1 + 1e-8)
            f1 = seen_pos[activity] / (k - 1 + 1e-8)
            f2 = act_count[activity] / (k + 1e-8)
            f3 = np.mean(act_times[activity]) / max_e
            f4 = len(act_res[activity]) / (max_res + 1e-8)
            f5 = norm_amount

            features.append([f0, f1, f2, f3, f4, f5])

        x = torch.tensor(features, dtype=torch.float32)

        edge_counts = defaultdict(int)
        for i in range(len(prefix_events) - 1):
            source = local_idx[prefix_events[i]]
            target = local_idx[prefix_events[i + 1]]
            edge_counts[(source, target)] += 1

        if edge_counts:
            srcs, dsts, weights = [], [], []
            max_weight = max(edge_counts.values())

            for (source, target), weight in edge_counts.items():
                srcs.append(source)
                dsts.append(target)
                weights.append(weight / max_weight)

            edge_index = torch.tensor(
                [srcs, dsts],
                dtype=torch.long,
            )
            edge_attr = torch.tensor(
                weights,
                dtype=torch.float32,
            ).view(-1, 1)
        else:
            # k=1: no transition exists yet. A self-loop makes the
            # graph valid for the GAT and provides an edge feature.
            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long,
            )
            edge_attr = torch.tensor(
                [[1.0]],
                dtype=torch.float32,
            )

        y = torch.tensor([label], dtype=torch.long)

        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                case_id=torch.tensor([int(case_id)], dtype=torch.long)
                if case_id is not None else torch.tensor([-1], dtype=torch.long),
            )
        )

    return graphs


def log_to_prefix_graphs(log_df, vocab, max_cases=None):
    """Convert the event log into all non-terminal prefix graphs."""

    min_amount, max_amount = normalise_amount(log_df)
    max_elapsed = max_elapsed_from_log(log_df)

    case_ids = log_df[CASE_COL].dropna().unique()
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    all_graphs = []

    # groupby once rather than repeatedly filtering the complete dataframe.
    grouped = log_df.groupby(CASE_COL, sort=False)

    for group_id, case_id in enumerate(case_ids):
        case_df = grouped.get_group(case_id)
        all_graphs.extend(
            case_to_prefixes(
                case_df,
                vocab,
                max_amount,
                min_amount,
                max_elapsed,
                case_id=group_id,
            )
        )

    return all_graphs


def trace_to_graph(
    trace,
    vocab,
    max_amount=1.0,
    min_amount=0.0,
    max_elapsed=1.0,
):
    """
    Build an unlabeled graph from a user-entered activity trace.

    The web demo only receives activity names, so inference uses:
      - elapsed time based on trace position,
      - one placeholder resource per activity occurrence,
      - amount = 0.

    This keeps the six-dimensional feature schema identical to training.
    """

    if not isinstance(trace, (list, tuple)) or len(trace) == 0:
        raise ValueError("Trace must contain at least one activity.")

    unknown = [activity for activity in trace if activity not in vocab]
    if unknown:
        raise ValueError(
            "Unknown activity name(s): " + ", ".join(map(str, unknown))
        )

    # Build synthetic inference-time timestamps in one-second increments.
    base_time = pd.Timestamp("2000-01-01")
    case_df = pd.DataFrame(
        {
            ACTIVITY_COL: list(trace),
            TIME_COL: [
                base_time + pd.Timedelta(seconds=i)
                for i in range(len(trace))
            ],
            RESOURCE_COL: ["__DEMO_RESOURCE__"] * len(trace),
            AMOUNT_COL: [0.0] * len(trace),
        }
    )

    # There is no next-activity label at inference time. Build the graph
    # directly from the full observed trace.
    events = case_df[ACTIVITY_COL].tolist()
    times = pd.to_datetime(case_df[TIME_COL]).tolist()
    resources = case_df[RESOURCE_COL].tolist()

    elapsed = [
        (t - times[0]).total_seconds()
        for t in times
    ]

    vocab_size = len(vocab)
    seen_pos = {}
    act_times = defaultdict(list)
    act_res = defaultdict(set)
    act_count = defaultdict(int)

    for pos, (activity, t, resource) in enumerate(
        zip(events, elapsed, resources)
    ):
        if activity not in seen_pos:
            seen_pos[activity] = pos
        act_times[activity].append(t)
        act_res[activity].add(resource)
        act_count[activity] += 1

    unique_acts = sorted(
        seen_pos.keys(),
        key=lambda activity: seen_pos[activity],
    )
    local_idx = {
        activity: i for i, activity in enumerate(unique_acts)
    }

    max_res = max(
        len(act_res[activity])
        for activity in unique_acts
    )

    amount = 0.0
    norm_amount = (amount - min_amount) / (
        max_amount - min_amount + 1e-8
    )

    features = []
    k = len(events)

    for activity in unique_acts:
        f0 = vocab[activity] / (vocab_size - 1 + 1e-8)
        f1 = seen_pos[activity] / (max(k - 1, 1) + 1e-8)
        f2 = act_count[activity] / (k + 1e-8)
        f3 = np.mean(act_times[activity]) / (
            max_elapsed + 1e-8
        )
        f4 = len(act_res[activity]) / (
            max_res + 1e-8
        )
        f5 = norm_amount

        features.append([f0, f1, f2, f3, f4, f5])

    x = torch.tensor(features, dtype=torch.float32)

    edge_counts = defaultdict(int)
    for i in range(len(events) - 1):
        source = local_idx[events[i]]
        target = local_idx[events[i + 1]]
        edge_counts[(source, target)] += 1

    if edge_counts:
        srcs, dsts, weights = [], [], []
        max_weight = max(edge_counts.values())

        for (source, target), weight in edge_counts.items():
            srcs.append(source)
            dsts.append(target)
            weights.append(weight / max_weight)

        edge_index = torch.tensor(
            [srcs, dsts],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            weights,
            dtype=torch.float32,
        ).view(-1, 1)
    else:
        edge_index = torch.tensor(
            [[0], [0]],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [[1.0]],
            dtype=torch.float32,
        )

    # Dummy y is required only to create a standard PyG Data object.
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
