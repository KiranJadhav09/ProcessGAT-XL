from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.predictor import NextActivityPredictor


ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = ROOT / "artifacts" / "final_experiment"
CHECKPOINT_DIR = FINAL_DIR / "checkpoints"
METRICS_DIR = FINAL_DIR / "metrics"
PLOTS_DIR = FINAL_DIR / "plots"
ARCHITECTURE_DIR = FINAL_DIR / "architecture"
STATIC_DIR = ROOT / "app" / "static"


app = FastAPI(
    title="Process Intelligence Platform",
    description="ProcessGAT-XL next-activity prediction platform",
    version="1.0.0",
)


predictor = NextActivityPredictor(
    str(CHECKPOINT_DIR)
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


app.mount(
    "/plots",
    StaticFiles(directory=str(PLOTS_DIR)),
    name="plots",
)


app.mount(
    "/architecture",
    StaticFiles(directory=str(ARCHITECTURE_DIR)),
    name="architecture",
)


class PredictionRequest(BaseModel):
    trace: list[str]


def read_json(filename: str):
    path = METRICS_DIR / filename

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ProcessGAT-XL",
        "checkpoint": str(
            CHECKPOINT_DIR / "final_model.pt"
        ),
    }


@app.get("/model-info")
def model_info():

    config = read_json("experiment_config.json")
    metrics = read_json("test_metrics.json")
    architecture = read_json(
        "architecture.json"
    )

    return {
        "model": "ProcessGAT-XL",
        "parameters": architecture.get(
            "parameters",
            190895,
        ),
        "model_config": config.get(
            "model",
            {},
        ),
        "training_config": config.get(
            "training",
            {},
        ),
        "metrics": {
            "top1": metrics.get("top1"),
            "top3": metrics.get("top3"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get(
                "weighted_f1"
            ),
        },
    }


@app.get("/analytics")
def analytics():

    return {
        "test_metrics": read_json(
            "test_metrics.json"
        ),
        "training_history": read_json(
            "training_history.json"
        ),
        "training_metadata": read_json(
            "training_metadata.json"
        ),
        "dataset_statistics": read_json(
            "dataset_statistics.json"
        ),
        "experiment_config": read_json(
            "experiment_config.json"
        ),
    }


@app.get("/predict")
def predict_get():
    return JSONResponse(
        {
            "message": (
                "Use POST /predict with "
                "{'trace': ['activity', ...]}"
            )
        }
    )


@app.post("/predict")
def predict(request: PredictionRequest):

    if not request.trace:
        return {
            "error": "Trace cannot be empty."
        }

    try:

        predictions = predictor.predict(
            request.trace
        )

        return {
            "trace": request.trace,
            "predictions": predictions,
        }

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc)
            },
        )
