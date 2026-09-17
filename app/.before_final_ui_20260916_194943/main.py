from pathlib import Path
import time
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from app.predictor import NextActivityPredictor
except Exception:
    NextActivityPredictor = None

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts" / "final_experiment"
CKPT = FINAL / "checkpoints"
PLOTS = FINAL / "plots"
ARCH = FINAL / "architecture"
STATIC = Path(__file__).resolve().parent / "static"

MODEL_CONFIG = {
    "in_channels": 6,
    "hidden": 64,
    "num_layers": 3,
    "heads": 8,
    "edge_dim": 8,
    "dropout": 0.25,
}
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 256,
    "lr": 0.0002,
    "weight_decay": 0.0001,
    "label_smoothing": 0.05,
    "warmup_epochs": 5,
    "patience": 20,
    "seed": 42,
}
METRICS = {
    "top1": 0.7603225708007812,
    "top3": 0.9988172054290771,
    "macro_f1": 0.6755298499943678,
    "weighted_f1": 0.754352805921297,
}
# BPIC 2012 W source log: 9,658 cases. 7,469 cases are represented in the
# generated prefix-graph corpus used for this final experiment.
DATASET = {
    "source_cases": 9658,
    "unique_cases": 7469,
    "total_graphs": 62755,
    "mean_nodes": 2.36,
    "mean_edges": 2.94,
    "train_graphs": 44295,
    "validation_graphs": 9160,
    "test_graphs": 9300,
}
CLASSES = [
    "W_Afhandelen leads",
    "W_Beoordelen fraude",
    "W_Completeren aanvraag",
    "W_Nabellen incomplete dossiers",
    "W_Nabellen offertes",
    "W_Valideren aanvraag",
]
F1 = {
    "W_Afhandelen leads": 0.47376,
    "W_Beoordelen fraude": 0.77273,
    "W_Completeren aanvraag": 0.81775,
    "W_Nabellen incomplete dossiers": 0.81713,
    "W_Nabellen offertes": 0.81376,
    "W_Valideren aanvraag": 0.35805,
}
SUPPORT = {
    "W_Afhandelen leads": 175,
    "W_Beoordelen fraude": 38,
    "W_Completeren aanvraag": 2850,
    "W_Nabellen incomplete dossiers": 1712,
    "W_Nabellen offertes": 3409,
    "W_Valideren aanvraag": 1116,
}

predictor = None
if NextActivityPredictor and CKPT.exists():
    try:
        predictor = NextActivityPredictor(str(CKPT))
    except Exception:
        predictor = None

app = FastAPI(
    title="ProcessGAT",
    version="2.1.0",
    description="Process intelligence and next-activity prediction.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PLOTS.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS)), name="plots")
if ARCH.exists():
    app.mount("/architecture", StaticFiles(directory=str(ARCH)), name="architecture")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class PredictRequest(BaseModel):
    trace: list[str] = Field(..., min_length=1)


@app.get("/", response_class=HTMLResponse)
def home():
    html = (STATIC / "index.html").read_text()
    html = html.replace("__CLASSES__", json.dumps(CLASSES))
    html = html.replace("__F1__", json.dumps(F1))
    html = html.replace("__SUPPORT__", json.dumps(SUPPORT))
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ProcessGAT",
        "version": "2.1.0",
        "model_loaded": predictor is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "ProcessGAT-XL",
        "version": "2.1.0",
        "parameters": 190895,
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "evaluation_protocol": {
            "split": "case-level",
            "checkpoint_selection": "validation Macro-F1",
            "test_usage": "held-out evaluation after model selection",
        },
        "metrics": METRICS,
        "dataset": DATASET,
    }


@app.get("/metadata")
def metadata():
    return model_info()


@app.get("/analytics")
def analytics():
    return {
        "metrics": METRICS,
        "dataset": DATASET,
        "classes": CLASSES,
        "class_f1": F1,
        "class_support": SUPPORT,
    }


@app.get("/predict")
def predict_help():
    return {"message": "POST /predict with JSON {'trace':['activity 1','activity 2']}"}


@app.post("/predict")
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model predictor is not available.")
    trace = [str(x).strip() for x in req.trace if str(x).strip()]
    t = time.perf_counter()
    try:
        result = predictor.predict(trace)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    ms = (time.perf_counter() - t) * 1000
    if isinstance(result, dict):
        result["inference_ms"] = round(ms, 3)
        return result
    return {"trace": trace, "predictions": result, "inference_ms": round(ms, 3)}
