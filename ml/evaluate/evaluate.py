"""SightSpeak — mAP + inference-latency evaluation with gate metrics.

Computes detection accuracy (mAP@0.5) on the validation set and benchmarks
per-image inference latency. Writes metrics.json, which the CI eval gate reads
to decide whether a model is allowed to merge.

Usage:
    python evaluate/evaluate.py --params params.yaml --weights runs/.../best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from ultralytics import YOLO


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def measure_latency(model: YOLO, imgsz: int, device, runs: int, warmup: int) -> float:
    """Return median per-image inference latency in milliseconds."""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        timings.append((time.perf_counter() - start) * 1000.0)

    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SightSpeak model")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()

    params = load_params(args.params)
    ev = params["evaluate"]
    ds = params["dataset"]
    mlf = params["mlflow"]

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    mlflow.set_tracking_uri(mlf["tracking_uri"])
    mlflow.set_experiment(mlf["experiment"])

    with mlflow.start_run(run_name="evaluate"):
        model = YOLO(str(weights))

        val = model.val(
            data=ds["config"],
            imgsz=ev["imgsz"],
            device=ev["device"],
            verbose=False,
        )
        map50 = float(val.box.map50)      # mAP@0.5
        map5095 = float(val.box.map)      # mAP@0.5:0.95

        latency_ms = measure_latency(
            model,
            imgsz=ev["imgsz"],
            device=ev["device"],
            runs=ev["latency_runs"],
            warmup=ev["latency_warmup"],
        )

        passed = map50 >= ev["min_map"] and latency_ms <= ev["max_latency_ms"]

        metrics = {
            "map50": round(map50, 4),
            "map50_95": round(map5095, 4),
            "latency_ms": round(latency_ms, 3),
            "min_map": ev["min_map"],
            "max_latency_ms": ev["max_latency_ms"],
            "gate_passed": passed,
        }

        for key in ("map50", "map50_95", "latency_ms"):
            mlflow.log_metric(key, metrics[key])
        mlflow.log_metric("gate_passed", int(passed))

        out = Path(args.out)
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))

        print(json.dumps(metrics, indent=2))
        print(f"[evaluate] gate {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
