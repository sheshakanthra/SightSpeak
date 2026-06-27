<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&pause=1000&color=00D4FF&center=true&vCenter=true&width=600&lines=SightSpeak+%F0%9F%91%81%EF%B8%8F%E2%80%8D%F0%9F%97%A8%F0%9F%94%8A;On-Device+AI+for+the+Visually+Impaired;Real-Time+%C2%B7+Offline+%C2%B7+Privacy-First" alt="Typing SVG" />

<br/>

![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Android](https://img.shields.io/badge/Android-API%2024+-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)
![Python](https://img.shields.io/badge/Python%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8--nano-00FFFF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)
![ONNX](https://img.shields.io/badge/ONNX%20Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)

<br/>

> **SightSpeak** is a production-grade, fully offline Android application that provides real-time object detection and spoken audio guidance for visually impaired users — powered by YOLOv8-nano, ONNX Runtime, and a complete MLOps lifecycle. No cloud. No subscription. No privacy risk.

<br/>

[📱 Demo](#demo) • [🏗️ Architecture](#architecture) • [📊 Results](#results) • [🚀 Getting Started](#getting-started) • [🔬 MLOps Pipeline](#mlops-pipeline) • [🗺️ Roadmap](#roadmap)

</div>

---

## 🎯 Problem Statement

Over **285 million people** worldwide live with visual impairment. Existing assistive tools either require constant internet connectivity, expensive hardware, or sacrifice user privacy by sending camera footage to cloud servers. SightSpeak eliminates all three barriers.

| Existing Solutions | SightSpeak |
|---|---|
| ❌ Cloud-dependent | ✅ 100% on-device |
| ❌ Privacy risk (camera → server) | ✅ Zero data leaves the device |
| ❌ Expensive hardware | ✅ Works on any Android 7.0+ phone |
| ❌ No MLOps — static models | ✅ Full DVC + MLflow + CI/CD pipeline |
| ❌ English only | ✅ Multi-language TTS ready |

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────┐

│                    ANDROID DEVICE                        │

│                                                          │

│  📷 CameraX          🧠 YOLOv8-nano       🔊 TTS        │

│  Live Frames   →    ONNX Runtime    →   Audio Output    │

│  (320×320)         (25.6ms/frame)      (Offline TTS)    │

│                         ↑                               │

│              Priority & Guidance Logic                   │

│              (RAG-style label mapping)                   │

└─────────────────────────────────────────────────────────┘

↑

┌──────────────────────────────┐

│       MLOps PIPELINE         │

│                              │

│  DVC → MLflow → GitHub CI   │

│  Data  Track   Eval Gate     │

│  Version Runs  mAP ≥ 0.45   │

│                Lat ≤ 65ms   │

└──────────────────────────────┘

### Core Components

| Component | Technology | Purpose |
|---|---|---|
| Object Detection | YOLOv8-nano (Ultralytics) | Real-time multi-class detection |
| Mobile Inference | ONNX Runtime Android 1.16.3 | On-device model execution |
| Camera Pipeline | CameraX 1.3.4 | Live frame capture & analysis |
| Audio Guidance | Android TextToSpeech | Spoken object announcements |
| Experiment Tracking | MLflow 2.14 | Training metrics & artifact logging |
| Data Versioning | DVC 3.51 | Dataset & model version control |
| CI/CD Gate | GitHub Actions | Automated eval-gated deployment |
| Language | Kotlin + Python 3.11 | App + ML pipeline |

---

## 📊 Results

### Detection Performance (50-epoch COCO128 baseline)

| Class | mAP@0.5 | Precision | Recall |
|---|---|---|---|
| TV | 0.995 | 1.0 | 0.995 |
| Microwave | 0.995 | 0.939 | 0.995 |
| Refrigerator | 0.995 | 0.897 | 1.0 |
| Laptop | 0.789 | 0.897 | 0.667 |
| Clock | 0.888 | 0.859 | 0.889 |
| Teddy bear | 0.831 | 0.872 | 0.648 |
| Dining table | 0.800 | 0.948 | 0.615 |

### On-Device Performance (Poco X2 — Snapdragon 730G)

| Metric | Value | Target |
|---|---|---|
| Inference latency | **25.6ms/frame** | ≤ 65ms ✅ |
| Preprocessing | 0.3ms | — |
| Postprocessing | 1.3ms | — |
| Model size (ONNX) | **12.4 MB** | — |
| Min Android SDK | API 24 (Android 7.0) | — |
| Offline capability | **100%** | ✅ |

---

## 🔬 MLOps Pipeline

SightSpeak's key research contribution is its **production-grade MLOps lifecycle** — the gap identified in all surveyed prior works.
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────┐

│   DVC   │ →  │ Training │ →  │  MLflow  │ →  │ GitHub      │

│ Dataset │    │ YOLOv8n  │    │ Tracking │    │ Actions     │

│ Version │    │ 50 epochs│    │ mAP/Loss │    │ Eval Gate   │

│ Control │    │ COCO128  │    │ Artifacts│    │ Auto-deploy │

└─────────┘    └──────────┘    └──────────┘    └─────────────┘

### Eval Gate Logic (GitHub Actions)

Every model push is automatically evaluated. Promotion is **blocked** if:
- `mAP@0.5 < 0.45` — accuracy regression
- `Inference latency > 65ms` — performance regression

This ensures only verified, production-quality models reach the Android app.

### Pipeline Stages (DVC)

```yaml
stages:
  train  →  export  →  evaluate
```

Run the full pipeline with a single command:
```bash
cd ml && dvc repro
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Android Studio (Hedgehog or later)
- Android device (API 24+, USB debugging enabled)
- Git + DVC

### ML Pipeline Setup

```bash
# Clone the repo
git clone https://github.com/sheshakanthra/SightSpeak.git
cd SightSpeak/ml

# Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Train the model
python train/train.py

# Export to ONNX
python export/export.py

# Evaluate
python evaluate/evaluate.py
```

### Android App Setup

```bash
# Open android/ folder in Android Studio
# Connect your Android device via USB
# Enable Developer Options + USB Debugging
# Click Run ▶ in Android Studio
```

Or install the APK directly:
```bash
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

---

## 📁 Repository Structure
SightSpeak/

├── android/                          # Kotlin Android application

│   ├── app/src/main/

│   │   ├── java/com/example/sightspeak/

│   │   │   └── MainActivity.kt       # CameraX + ONNX + TTS integration

│   │   ├── AndroidManifest.xml

│   │   └── res/layout/activity_main.xml

│   └── build.gradle.kts

│

├── ml/                               # ML pipeline

│   ├── train/train.py                # YOLOv8 training + MLflow logging

│   ├── export/export.py              # ONNX export + quantization

│   ├── evaluate/evaluate.py          # mAP + latency evaluation

│   ├── guidance/labels.json          # Object → spoken cue mapping

│   ├── dvc.yaml                      # DVC pipeline definition

│   ├── params.yaml                   # Hyperparameters

│   ├── MLproject                     # MLflow project file

│   ├── requirements.txt

│   └── .github/workflows/

│       └── eval_gate.yml             # CI/CD evaluation gate

│

└── runs/                             # Training artifacts & ONNX models

---

## 🗺️ Roadmap

- [x] YOLOv8-nano baseline training
- [x] ONNX Runtime Android integration
- [x] CameraX live detection pipeline
- [x] Offline TTS audio guidance
- [x] MLflow experiment tracking
- [x] DVC data versioning
- [x] GitHub Actions eval gate
- [ ] INT8 quantization for 2× speed improvement
- [ ] Tamil / Hindi TTS language support
- [ ] Custom obstacle dataset (stairs, doors, vehicles)
- [ ] Distance estimation from bounding box size
- [ ] Haptic feedback for urgent detections
- [ ] YOLOv8s upgrade for higher accuracy

---

## 🧑‍💻 About the Developer

**Sheshakanth RA** — AI Developer & B.Tech Information Technology Student  
Chennai Institute of Technology (2024–2028) | CGPA: 8.0

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/sheshakanthra)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/sheshakanthra)

**Certifications:** AWS Cloud Practitioner · NPTEL IoT (Elite + Silver) · IBM AI Fundamentals · Cisco CCNA · MongoDB Basics

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for accessibility · Powered by on-device AI**

*"Technology should remove barriers, not create them."*

⭐ Star this repo if SightSpeak inspires you

</div>
