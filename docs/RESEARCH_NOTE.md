# Research Note — Test-Time Adaptation for Edge Clinical Vision Models
**Nithin Reddy | Started: September 2026**
**Status: Active — baseline in progress**

---

## Problem

A YOLOv8-based patient monitoring model trained on one clinical dataset degrades when
deployed across sites with different cameras, lighting, or patient demographics. Standard
fine-tuning requires labeled data from the new site, which is expensive and often
infeasible under patient privacy constraints. The question is whether the model can adapt
at deployment time using only unlabeled target-domain video.

This is a real limitation of my existing IEEE patient monitoring work
(DOI: 10.1109/INSTCon69741.2026.11691839), which was trained and evaluated on a single
dataset. Cross-site deployment generalization was not addressed.

---

## Research Question

Can a lightweight test-time adaptation (TTA) mechanism make an edge-deployed clinical
vision model robust to distribution shift without requiring labeled target-domain data,
and without retraining?

---

## Hypothesis

Unlabeled target-domain video contains enough distributional signal (through detection
confidence entropy and feature statistics) to adapt a pre-trained YOLOv8 model's
normalization layers at inference time, recovering a measurable fraction of the accuracy
lost under domain shift.

---

## Experimental Design

### Source domain
Pre-trained YOLOv8n (COCO person class) or own patient monitoring checkpoint.

### Target domains (domain shift scenarios)
1. Indoor/hospital-environment surveillance (different camera angle/lens)
2. Night-mode or low-illumination clinical video
3. Cross-dataset: model trained on Dataset A, evaluated on Dataset B with different
   patient population distribution

Public datasets suitable for building source/target splits without IRB:
- MVOR (Multi-View Operating Room) — OR-environment person detection
- COCO-person (source) → HiCo hospital images (target)
- MOT17/MOT20 surveillance clips as a proxy for camera-variation domain shift

### Baseline
YOLOv8n inference on target domain with no adaptation. Measure:
- mAP@0.5 and mAP@0.5:0.95
- Detection confidence distribution (entropy)
- Per-class recall

### TTA methods to evaluate
1. No adaptation (baseline)
2. TENT — entropy minimization on batch norm statistics at test time
3. TTT with auxiliary reconstruction head (PathTTT-style)
4. SHOT — information maximization without source data
5. Confidence-threshold gating — adapt only when model confidence diverges
   from training-time statistics

### Metrics
- mAP@0.5 and mAP@0.5:0.95 on target domain
- Inference latency added by adaptation (ms/frame)
- Frames required before adaptation stabilizes
- Whether adaptation degrades on in-distribution frames (catastrophic adaptation check)

---

## Why This Is Non-Trivial

Standard TTA was designed for image classifiers. Applying it to single-stage detection:

1. Detection heads produce spatially-structured, variable-density outputs — entropy
   minimization must be defined differently than classification softmax.
2. Quantized edge deployments often remove batch normalization — TENT's batch-stat
   update may not transfer directly.
3. Clinical video has strong temporal correlation — frames are not i.i.d., so naive
   per-frame adaptation may overfit to a single patient's pose.

These constraints make the analysis non-trivial even where the core idea is established.

---

## Repository Structure

github.com/NithinReddy22/clinical-tta-edge

```
clinical-tta-edge/
  README.md          — Research question, status, planned experiments
  requirements.txt   — Dependency specifications
  run_experiments.py — Unified CLI experiment runner
  baseline/
    evaluate.py          — YOLOv8 inference + mAP evaluation
    dataset_setup.py     — Automated shift generators (contrast, brightness, blur, noise)
    entropy_analysis.py  — Confidence entropy distribution shift & divergence analysis
  methods/
    tent.py              — Detection-entropy TENT on normalization affine params (10,592 params)
    shot.py              — Source-Free Information Maximization (proposal entropy + diversity)
    entropy_gating.py    — Selective adaptation based on in-distribution entropy calibration
    ttt_aux.py           — Self-supervised auxiliary rotation pretext head TTT
  experiments/
    configs/             — Shifted and source dataset YAMLs
    results/             — Verified metrics, entropy shift analysis, and distribution plots
  tests/
    test_tta_pipeline.py — Unit and integration test suite (6/6 passing)
```

---

## Key Milestone Progress

1. **Clean Source Domain Baseline**:
   - Evaluated YOLOv8n on COCO-128: mAP@50 = 0.6054, mAP@50-95 = 0.4454, Latency = 38.0 ms.
2. **Entropy Shift Quantification**:
   - Shifted domains (contrast reduction, brightness drop, blur, noise) induce measurable distribution divergence.
   - For example, contrast reduction increases mean entropy and shifts low-confidence proportions.
   - 2-Wasserstein distances quantified between source and target confidence/entropy distributions.
3. **Detection TTA Methods Implemented**:
   - `TENTAdapter`: Isolates 10,592 affine parameters (99.7% network frozen), computes spatially-structured detection entropy, supports parameter rollback.
   - `SHOTAdapter`: Mitigates class representation collapse via class diversity regularization.
   - `EntropyGatedTENT`: Gated execution saves edge compute operations while preserving in-distribution accuracy.
   - `TTTAuxAdapter`: Self-supervised rotation classification pretext task for feature adaptation.
4. **Validation Suite**:
   - Automated pytest suite covers entropy mathematics, shift generation, parameter isolation, and gating.

---

## Week-by-Week Timeline

| Week | Goal | Status |
|---|---|---|
| 1–2 | Set up repo, run baseline YOLOv8 on target dataset, measure mAP drop | Completed |
| 3–4 | Implement TENT, compare against baseline | Completed |
| 5–6 | Implement auxiliary-task TTT variant & SHOT | Completed |
| 7–8 | Entropy gating & latency profiling on CPU (edge proxy) | Completed |
| 9–10 | Full video stream benchmarks & 4-page workshop abstract | In progress |

Current project statement:
"Baseline established, domain shifts quantified via Wasserstein entropy divergence, and 4 lightweight edge adaptation methods implemented and verified."
