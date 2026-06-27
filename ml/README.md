# SightSpeak ML

On-device, real-time object detection for visually impaired users.
YOLOv8-nano → TFLite INT8, with a DVC pipeline, MLflow tracking, and a
CI eval gate.

```
ml/
├── data/          # dataset (DVC tracked) + data.yaml descriptor
├── train/         # YOLOv8-nano training + MLflow logging
├── export/        # TFLite INT8 export & quantization
├── evaluate/      # mAP + latency eval → metrics.json
├── guidance/      # class → spoken guidance mapping (RAG layer)
├── dvc.yaml       # train → export → evaluate pipeline
├── params.yaml    # all hyperparameters / thresholds
└── MLproject      # MLflow entry points
```

## Getting Started

```bash
# 1. Install dependencies
cd ml
pip install -r requirements.txt

# 2. Initialize DVC — ONE TIME ONLY.
#    Use --subdir because .dvc/ lives in ml/, not the git repo root.
#    This creates .dvc/cache and internals; it preserves the existing
#    .dvc/config (remote settings).
dvc init --subdir

# 3. Track your dataset (once you have images/ + labels/ in place)
dvc add data/images data/labels
git add data/images.dvc data/labels.dvc data/.gitignore

# 4. Run the full pipeline: train → export → evaluate
dvc repro
```

> **Note:** `dvc init --subdir` must be run once locally — the committed
> `.dvc/config` alone does not initialize the cache. Without `--subdir`,
> DVC expects `.dvc/` at the git root and will error, since ours lives in `ml/`.

Configure the storage remote in `.dvc/config` (a `local` remote is set by
default). For a shared remote, point it at your bucket and provide credentials
via env vars or `dvc remote modify --local`.

## Eval Gate & Branch Protection

Every PR that touches `ml/**` triggers the **SightSpeak Eval Gate**
(`.github/workflows/eval_gate.yml`), which runs `dvc repro` (retrain → export →
evaluate) and enforces two thresholds from `params.yaml`:

| Metric | Threshold | Fails if |
| --- | --- | --- |
| mAP@0.5 | `min_map: 0.45` | mAP **< 0.45** |
| Inference latency | `max_latency_ms: 65` | latency **> 65 ms** |

If either check fails, the gate job exits non-zero and the PR check goes red.

**To make the gate actually block merges**, enable branch protection on `main`
(GitHub → Settings → Branches → Add rule):

- Require status checks to pass before merging → select **`eval-gate`**.
- Require branches to be up to date before merging (recommended).

Without branch protection the workflow still runs and reports pass/fail, but
GitHub will not prevent a merge on failure. The thresholds live in
`params.yaml` (`evaluate.min_map`, `evaluate.max_latency_ms`) — update them
there and the gate stays in sync, since `evaluate.py` writes them into
`metrics.json` which the workflow reads.
