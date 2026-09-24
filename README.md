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

### Phase 1 — Baseline (current)
- [ ] Run YOLOv8n (COCO-pretrained) on source domain, verify mAP
- [ ] Run same model on Target 1 (MOT17), measure mAP degradation
- [ ] Run same model on Target 2 (HDA), measure mAP degradation
- [ ] Document confidence entropy distribution shift
- [ ] Record inference latency baseline on CPU (edge proxy)

### Phase 2 — TTA Methods
- [ ] TENT: entropy minimization on batch norm / layer norm statistics
- [ ] Auxiliary-task TTT (PathTTT-inspired)
- [ ] SHOT: source-free information maximization
- [ ] Entropy-gating: adapt only when entropy exceeds threshold

### Phase 3 — Analysis
- [ ] Compare mAP recovery across methods
- [ ] Measure per-method adaptation latency overhead
- [ ] Test for catastrophic adaptation on in-distribution data
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
  baseline/
    evaluate.py          # YOLOv8 inference + mAP on any COCO-format dataset
    dataset_setup.py     # Instructions and scripts to prepare datasets
    entropy_analysis.py  # Confidence entropy distribution analysis
  methods/
    tent.py              # TENT adaptation for detection
    ttt_aux.py           # Auxiliary-task TTT (planned)
    shot.py              # SHOT adaptation (planned)
  experiments/
    configs/             # YAML per experiment
    results/             # CSV result logs
  notebooks/
    exploratory.ipynb    # EDA and visualization
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
```

Install: `pip install -r requirements.txt`

---

## Current Status (September 2026)

| Component | Status |
|---|---|
| Repo initialized | Completed |
| Source model (YOLOv8n COCO) | Ready & Evaluated |
| Baseline evaluation script (`evaluate.py`) | Completed |
| Initial baseline experiment (COCO128 source) | Completed |
| TENT implementation (`tent.py`) | Initial implementation completed |
| Target domain datasets (MOT17 / synthetic shift) | Setup script ready |
| Shifted domain evaluations & adaptation runs | In progress |

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
