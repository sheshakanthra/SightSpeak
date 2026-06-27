"""SightSpeak — YOLOv8-nano training with MLflow tracking.

Trains the on-device detector that powers SightSpeak's real-time guidance.
Reads every hyperparameter from params.yaml so the run is reproducible and
DVC-trackable. Metrics, params, and the best checkpoint are logged to MLflow.

Usage:
    python train/train.py --params params.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano for SightSpeak")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    m = params["model"]
    t = params["train"]
    ds = params["dataset"]
    mlf = params["mlflow"]

    mlflow.set_tracking_uri(mlf["tracking_uri"])
    mlflow.set_experiment(mlf["experiment"])

    with mlflow.start_run(run_name=t["name"]):
        # Log the flattened config so runs are comparable in the MLflow UI.
        mlflow.log_params(
            {
                "arch": m["arch"],
                "num_classes": m["num_classes"],
                **{f"train.{k}": v for k, v in t.items()},
            }
        )

        model = YOLO(m["arch"])
        results = model.train(
            data=ds["config"],
            epochs=t["epochs"],
            batch=t["batch"],
            imgsz=t["imgsz"],
            optimizer=t["optimizer"],
            lr0=t["lr0"],
            lrf=t["lrf"],
            momentum=t["momentum"],
            weight_decay=t["weight_decay"],
            warmup_epochs=t["warmup_epochs"],
            patience=t["patience"],
            seed=t["seed"],
            device=t["device"],
            workers=t["workers"],
            project=t["project"],
            name=t["name"],
            exist_ok=True,
        )

        # ultralytics exposes final validation metrics on results.results_dict.
        metrics = getattr(results, "results_dict", {}) or {}
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key.replace("(", "_").replace(")", ""), float(value))
            except (TypeError, ValueError):
                continue

        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(t["project"]) / t["name"]
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            mlflow.log_artifact(str(best), artifact_path="weights")
            print(f"[train] best checkpoint: {best}")
        else:
            print(f"[train] WARNING: expected checkpoint not found at {best}")


if __name__ == "__main__":
    main()
