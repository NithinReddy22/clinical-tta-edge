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

## Repository

github.com/NithinReddy22/clinical-tta-edge

```
clinical-tta-edge/
  README.md          — research question, status, planned experiments
  baseline/          — YOLOv8 inference + evaluation on target domain
  methods/
    tent.py
    ttt_aux.py
    shot.py
  experiments/       — config files + result CSVs
  notebooks/         — exploratory analysis
```

---

## Week-by-Week Timeline (working full-time, ~8h/week)

| Week | Goal |
|---|---|
| 1–2 | Set up repo, run baseline YOLOv8 on target dataset, measure mAP drop |
| 3–4 | Implement TENT, compare against baseline |
| 5–6 | Implement auxiliary-task TTT variant |
| 7–8 | SHOT comparison, latency profiling on CPU (edge proxy) |
| 9–10 | Write 4-page workshop abstract |

By week 2 the honest statement in emails becomes:
"I have established a baseline and am comparing adaptation methods."
