# clinical-tta-edge

**Test-Time Adaptation for Edge-Deployed Clinical Vision Models**

> Status: **Active — baseline in progress** (started September 2026)

---

## Research Question

Can a lightweight test-time adaptation (TTA) mechanism make an edge-deployed clinical
vision model robust to distribution shift without requiring labeled target-domain data,
and without retraining?

---

## Motivation

My IEEE publication on YOLOv8-based patient safety monitoring
(DOI: 10.1109/INSTCon69741.2026.11691839) achieved F1=0.947 on its training distribution.
However, it was trained and evaluated on a single controlled dataset. Deploying the same
model across hospitals with different cameras, lighting, or patient demographics produces
systematic accuracy degradation — the same distribution shift problem addressed in
pathology by PathTTT (Hosseini et al., IPMI 2025) and cooperative perception by
V2X-TTA (IEEE IoT 2026).

The question: can lightweight adaptation at test time recover this accuracy loss without
labeled target data and without the compute budget of a foundation model?

---

## Domain Shift Setup

| | Dataset | Description |
|---|---|---|
| **Source** | YOLOv8n pretrained (COCO person) | General person detection |
| **Target 1** | MOT17 indoor sequences | Surveillance-style, different angles |
| **Target 2** | HDA Person Dataset (hospital corridors) | Clinical-environment domain |
| **Target 3** | Synthetic brightness/contrast shift on COCO | Controlled quantified shift |

All datasets are publicly available without IRB requirements.

---

## Experimental Plan

### Phase 1 — Baseline (Completed)
- [x] Run YOLOv8n (COCO-pretrained) on source domain, verify mAP
- [x] Implement synthetic domain shifts (contrast, brightness, blur, noise)
- [x] Document confidence entropy distribution shift (`baseline/entropy_analysis.py`)
- [x] Calculate distribution divergence metrics (Wasserstein distance & JS divergence)
- [x] Record inference latency baseline on CPU (edge proxy: 38ms / frame)

### Phase 2 — TTA Methods (Implemented & Bench-Ready)
- [x] TENT: detection entropy minimization on BatchNorm affine statistics (`methods/tent.py`)
- [x] SHOT: source-free information maximization with proposal entropy & class diversity (`methods/shot.py`)
- [x] Entropy-gating: adapt only when entropy exceeds threshold to save edge compute (`methods/entropy_gating.py`)
- [x] Auxiliary-task TTT: self-supervised 4-way rotation prediction (`methods/ttt_aux.py`)
- [x] Comprehensive test suite with 100% pass rate (`tests/test_tta_pipeline.py`)

### Phase 3 — Analysis & Benchmark Suite
- [x] Unified CLI experiment runner (`run_experiments.py`)
- [ ] MOT17 sequence video streaming evaluation
- [ ] 4-page workshop abstract

---

## Why This Is Non-Trivial for Detection

Standard TTA assumes image classifiers with softmax entropy.
YOLOv8 is a single-stage anchor-free detector:
- Multi-scale spatially-structured detection heads (not a global class distribution)
- Temporal correlation between video frames (non-i.i.d. assumption violated)
- Quantized edge deployments often remove batch normalization

These constraints mean TENT cannot be applied directly.
This project explores how to formulate the adaptation signal for detection models.

---

## Project Structure

```
clinical-tta-edge/
  README.md
  requirements.txt
  run_experiments.py     # Unified CLI experiment runner
  baseline/
    evaluate.py          # YOLOv8 inference + mAP evaluation
    dataset_setup.py     # Automated shift generator (contrast, brightness, blur, noise)
    entropy_analysis.py  # Confidence entropy distribution analysis & plotting
  methods/
    tent.py              # Detection-entropy TENT on normalization affine params
    shot.py              # SHOT adaptation (entropy + class diversity maximization)
    entropy_gating.py    # Gated TTA (selective compute-saving adaptation)
    ttt_aux.py           # Auxiliary-task self-supervised TTT
  experiments/
    configs/             # Dataset & experiment YAML configurations
    results/             # Evaluation logs, JSON summaries, and distribution plots
  tests/
    test_tta_pipeline.py # Unit and integration test suite
```

---

## Requirements

```
ultralytics>=8.0
torch>=2.0
torchvision
numpy
opencv-python
pandas
matplotlib
tqdm
pyyaml
scipy>=1.10.0
pytest>=7.0.0
```

Install: `pip install -r requirements.txt`

---

## Current Status (September 2026)

| Component | Status |
|---|---|
| Repo initialized | Completed |
| Source model (YOLOv8n COCO) | Ready & Evaluated |
| Baseline evaluation script (`evaluate.py`) | Completed |
| Synthetic shift generator (`dataset_setup.py`) | Completed (4 shifts) |
| Entropy shift analysis (`entropy_analysis.py`) | Completed & Plotted |
| TENT adapter (`methods/tent.py`) | Completed & Verified (10,592 params / 99.7% frozen) |
| SHOT adapter (`methods/shot.py`) | Completed & Verified |
| Entropy-Gated adapter (`methods/entropy_gating.py`) | Completed & Verified |
| Auxiliary-task TTT adapter (`methods/ttt_aux.py`) | Completed & Verified |
| Automated test suite (`tests/`) | 6/6 tests passing |
| Unified CLI runner (`run_experiments.py`) | Completed & Verified |

---

## Baseline Benchmark Results

Evaluated on source domain benchmark (COCO-128 val subset, 128 images) on local CPU hardware:

- **Hardware**: Intel Core Ultra 5 225U (CPU execution, edge proxy)
- **Model**: YOLOv8n (pre-trained, input resolution 640x640)
- **Metrics**:
  - **mAP@50**: 0.6054
  - **mAP@50-95**: 0.4454
  - **Precision**: 0.6385
  - **Recall**: 0.5361
  - **Latency**: 38.0 ms / image (single-thread edge proxy inference)

This establishes the clean in-distribution baseline against which distribution shifts and test-time adaptation recovery will be benchmarked.


---

Puluputturi Nithin Reddy — github.com/NithinReddy22
