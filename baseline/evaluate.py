"""
baseline/evaluate.py
====================
YOLOv8 inference and mAP evaluation on any COCO-format dataset.

Usage:
    python baseline/evaluate.py --model yolov8n.pt --data path/to/data.yaml --split val

The script evaluates a pretrained YOLOv8 model on a specified dataset split and
records mAP@0.5, mAP@0.5:0.95, per-class recall, and confidence entropy statistics.

This is Phase 1 of the clinical-tta-edge project.
No adaptation is applied here. This is the baseline.
"""

import argparse
import json
import csv
import os
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Runtime imports — requires: pip install ultralytics
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    import torch
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Run: pip install ultralytics"
    ) from e


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, data_yaml: str, split: str = "val", device: str = "cpu"):
    """
    Run YOLOv8 validation on the specified dataset split.

    Args:
        model_path: Path to .pt weights file (e.g. yolov8n.pt).
        data_yaml:  Path to COCO-format data.yaml file.
        split:      Dataset split to evaluate ("val" or "test").
        device:     "cpu" or "cuda:0".

    Returns:
        dict with mAP50, mAP50_95, per-class metrics, and entropy stats.
    """
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        split=split,
        device=device,
        verbose=True,
        save_json=True,       # saves per-image predictions to JSON
        conf=0.001,           # low threshold to capture full precision-recall curve
        iou=0.6,
    )

    metrics = {
        "model": model_path,
        "data": data_yaml,
        "split": split,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_ap50": results.box.ap50.tolist() if hasattr(results.box, "ap50") else [],
    }

    return metrics, results


def entropy_from_confidences(pred_confs: list[float]) -> float:
    """
    Compute mean prediction entropy from a list of detection confidences.
    Uses binary entropy: H = -p*log(p) - (1-p)*log(1-p)
    This is an approximation for the adaptation signal.
    """
    if not pred_confs:
        return float("nan")
    confs = np.clip(np.array(pred_confs, dtype=np.float32), 1e-6, 1 - 1e-6)
    entropy = -(confs * np.log(confs) + (1 - confs) * np.log(1 - confs))
    return float(np.mean(entropy))


def collect_confidence_entropy(model_path: str, data_yaml: str,
                               split: str = "val", device: str = "cpu",
                               max_images: int = 500) -> dict:
    """
    Run inference on up to max_images images and collect confidence distributions
    without running full mAP evaluation. Useful for quick entropy diagnostics.

    Returns:
        dict with mean_entropy, median_conf, low_conf_fraction
    """
    from ultralytics import YOLO
    import glob

    model = YOLO(model_path)

    # Parse data.yaml for image directory
    import yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    img_dir = data_cfg.get(split, data_cfg.get("val", ""))
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(os.path.dirname(data_yaml), img_dir)

    image_paths = (
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )[:max_images]

    all_confs = []
    for img_path in image_paths:
        preds = model(img_path, verbose=False, conf=0.01)
        for result in preds:
            if result.boxes is not None and len(result.boxes):
                confs = result.boxes.conf.cpu().numpy().tolist()
                all_confs.extend(confs)

    return {
        "n_images": len(image_paths),
        "n_detections": len(all_confs),
        "mean_conf": float(np.mean(all_confs)) if all_confs else float("nan"),
        "mean_entropy": entropy_from_confidences(all_confs),
        "low_conf_fraction": float(
            np.mean(np.array(all_confs) < 0.3)
        ) if all_confs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------

def save_results(metrics: dict, output_dir: str = "experiments/results"):
    """Save metrics to a CSV log and a JSON snapshot."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON snapshot
    json_path = os.path.join(output_dir, f"eval_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to master CSV
    csv_path = os.path.join(output_dir, "results_log.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"\nResults saved:\n  JSON: {json_path}\n  CSV:  {csv_path}")
    return json_path, csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline YOLOv8 evaluation — clinical-tta-edge Phase 1"
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 weights (default: yolov8n.pt, auto-downloaded from Ultralytics)"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to COCO-format data.yaml"
    )
    parser.add_argument(
        "--split", default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu or cuda:0"
    )
    parser.add_argument(
        "--entropy-only", action="store_true",
        help="Only collect entropy statistics (fast, no full mAP)"
    )
    parser.add_argument(
        "--max-images", type=int, default=500,
        help="Max images for entropy-only mode"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("clinical-tta-edge | Phase 1 Baseline Evaluation")
    print(f"Model:  {args.model}")
    print(f"Data:   {args.data}")
    print(f"Split:  {args.split}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    if args.entropy_only:
        print("Running entropy-only diagnostics...")
        entropy_stats = collect_confidence_entropy(
            args.model, args.data, args.split, args.device, args.max_images
        )
        print(json.dumps(entropy_stats, indent=2))
        save_results(entropy_stats)
    else:
        print("Running full mAP evaluation (baseline — no adaptation)...")
        metrics, _ = evaluate(args.model, args.data, args.split, args.device)
        print("\n--- Results ---")
        for k, v in metrics.items():
            if k != "per_class_ap50":
                print(f"  {k}: {v}")
        save_results(metrics)


if __name__ == "__main__":
    main()
