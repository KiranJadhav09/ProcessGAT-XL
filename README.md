---
title: ProcessGAT-XL Next-Activity Intelligence
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ProcessGAT-XL — Next-Activity Intelligence

Industry-style research demonstrator for next-activity prediction in
business processes using graph neural networks.

## Problem

Given a partial process trace, predict the next event/activity.

## Data

**BPIC 2012 W** loan-process event log.

The raw XES file is intentionally excluded from version control.

## Graph representation

Each non-terminal prefix is converted into a directed graph:

- **Nodes:** unique activities observed in the prefix
- **Edges:** directly-follows transitions
- **Edge feature:** normalized transition frequency
- **Node features:**
  1. activity identity
  2. first occurrence position
  3. occurrence frequency
  4. elapsed time from case start
  5. resource diversity
  6. requested amount

## Model

ProcessGAT-XL uses:

- 3 residual GATv2 layers
- 4 attention heads
- 64-dimensional node representation
- learned edge encoder
- LayerNorm + GELU
- learned graph attention pooling
- global mean/max pooling
- MLP classifier

## Evaluation protocol

The upgraded training pipeline uses a **case-level split**.
This is important because splitting individual prefixes can place prefixes
from the same case in train and test, which can produce an optimistic estimate.

Metrics:

- Top-1 accuracy
- Top-3 accuracy
- Macro-F1
- Weighted-F1
- per-class precision/recall/F1
- confusion matrix

## Training

```bash
source venv/bin/activate
python notebooks/01_train.py
```

The default configuration is a strong candidate, not a claim of globally
optimal hyperparameters.

For a lightweight hyperparameter sweep:

```bash
python scripts/tune.py
```

After tuning, update `MODEL_CONFIG` and `TRAIN_CONFIG` in
`notebooks/01_train.py`, then perform the full training run.

## Local application

```bash
python -m uvicorn app.main:app --port 7860
```

Open:

`http://127.0.0.1:7860`

## API

### Health

`GET /health`

### Metadata

`GET /metadata`

### Prediction

`POST /predict`

Example:

```json
{
  "trace": [
    "W_Beoordelen fraude",
    "W_Nabellen incomplete dossiers"
  ]
}
```

## Research caveat

The browser demo accepts activity names only. Consequently, inference-time
elapsed time, resource diversity, and amount are represented using neutral
demo assumptions. A research-grade extension should expose these contextual
attributes as model inputs.
