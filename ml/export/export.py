"""SightSpeak — TFLite INT8 export with post-training quantization.

Converts a trained YOLOv8-nano checkpoint into an INT8-quantized TFLite model
suitable for on-device, real-time inference. ultralytics handles the
ONNX -> TensorFlow -> TFLite conversion and uses a representative dataset
(drawn from the validation images) to calibrate the INT8 quantization.

Usage:
    python export/export.py --params params.yaml --weights runs/.../best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SightSpeak model to TFLite INT8")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    args = parser.parse_args()

    params = load_params(args.params)
    ex = params["export"]
    mlf = params["mlflow"]

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    mlflow.set_tracking_uri(mlf["tracking_uri"])
    mlflow.set_experiment(mlf["experiment"])

    with mlflow.start_run(run_name="export-tflite-int8"):
        mlflow.log_params(
            {
                "export.format": ex["format"],
                "export.int8": ex["int8"],
                "export.imgsz": ex["imgsz"],
                "export.calib_data": ex["calib_data"],
                "weights": str(weights),
            }
        )

        model = YOLO(str(weights))
        # int8=True triggers PTQ; `data` supplies the representative calibration set.
        exported = model.export(
            format=ex["format"],
            int8=ex["int8"],
            imgsz=ex["imgsz"],
            data=params["dataset"]["config"],
        )

        # `exported` is the path string to the produced .tflite (or its dir).
        src = Path(exported)
        dst = Path("export") / "sightspeak_int8.tflite"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copyfile(src, dst)
        else:
            # Some ultralytics versions return a dir; find the int8 tflite inside.
            candidates = list(src.glob("*int8*.tflite")) or list(src.glob("*.tflite"))
            if not candidates:
                raise FileNotFoundError(f"No .tflite produced under {src}")
            shutil.copyfile(candidates[0], dst)

        size_mb = dst.stat().st_size / (1024 * 1024)
        mlflow.log_metric("model_size_mb", size_mb)
        mlflow.log_artifact(str(dst), artifact_path="tflite")
        print(f"[export] wrote {dst} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
