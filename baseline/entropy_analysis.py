"""
baseline/entropy_analysis.py
=============================
Quantifies and visualizes confidence and entropy distribution shift
between in-distribution source domain and out-of-distribution clinical target domains.

Phase 1 deliverable:
  - Document confidence entropy distribution shift under domain perturbations.
  - Calculate distribution divergence metrics (Wasserstein distance, JS divergence).
  - Generate publication-quality comparison figures and JSON summary reports.

Usage:
    python baseline/entropy_analysis.py --source coco128.yaml --targets experiments/configs/coco128_synthetic_contrast.yaml experiments/configs/coco128_synthetic_brightness.yaml
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(u, v):
        if len(u) == 0 or len(v) == 0:
            return 0.0
        qs = np.linspace(0.01, 0.99, 100)
        return float(np.mean(np.abs(np.quantile(u, qs) - np.quantile(v, qs))))
from ultralytics import YOLO


def compute_binary_entropy(confs: np.ndarray) -> np.ndarray:
    """
    Computes Shannon binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    Clipping to avoid numerical instability at extremes.
    """
    p = np.clip(confs, 1e-6, 1.0 - 1e-6)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def resolve_image_paths(data_yaml_path: str, split: str = "val", max_images: int = 128) -> list[str]:
    """Extracts image paths from a YOLO data.yaml configuration."""
    from ultralytics.data.utils import check_det_dataset
    if not os.path.isfile(data_yaml_path):
        meta = check_det_dataset(data_yaml_path)
        img_dir = str(meta.get(split, meta.get("val", "")))
    else:
        with open(data_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        root = cfg.get("path", "")
        split_dir = cfg.get(split, cfg.get("val", ""))
        if not os.path.isabs(split_dir) and root:
            img_dir = os.path.join(root, split_dir)
        elif not os.path.isabs(split_dir):
            img_dir = os.path.join(os.path.dirname(os.path.abspath(data_yaml_path)), split_dir)
        else:
            img_dir = split_dir

    paths = sorted(
        list(Path(img_dir).glob("*.jpg")) +
        list(Path(img_dir).glob("*.png")) +
        list(Path(img_dir).glob("*.jpeg"))
    )[:max_images]
    return [str(p) for p in paths]


def collect_domain_confs_and_entropy(
    model: YOLO,
    data_yaml: str,
    domain_name: str,
    split: str = "val",
    max_images: int = 128,
    conf_thresh: float = 0.01,
) -> dict:
    """
    Runs model inference on the domain and collects detection confidences and entropies.
    """
    image_paths = resolve_image_paths(data_yaml, split=split, max_images=max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset YAML: {data_yaml}")

    all_confs = []
    per_image_counts = []

    for img_path in image_paths:
        preds = model(img_path, verbose=False, conf=conf_thresh)
        count = 0
        for res in preds:
            if res.boxes is not None and len(res.boxes):
                confs = res.boxes.conf.cpu().numpy().tolist()
                all_confs.extend(confs)
                count += len(confs)
        per_image_counts.append(count)

    confs_arr = np.array(all_confs, dtype=np.float32)
    entropy_arr = compute_binary_entropy(confs_arr) if len(confs_arr) > 0 else np.array([])

    stats = {
        "domain": domain_name,
        "yaml_path": data_yaml,
        "n_images": len(image_paths),
        "total_detections": int(len(confs_arr)),
        "mean_detections_per_image": float(np.mean(per_image_counts)) if per_image_counts else 0.0,
        "mean_conf": float(np.mean(confs_arr)) if len(confs_arr) else float("nan"),
        "median_conf": float(np.median(confs_arr)) if len(confs_arr) else float("nan"),
        "std_conf": float(np.std(confs_arr)) if len(confs_arr) else float("nan"),
        "mean_entropy": float(np.mean(entropy_arr)) if len(entropy_arr) else float("nan"),
        "median_entropy": float(np.median(entropy_arr)) if len(entropy_arr) else float("nan"),
        "std_entropy": float(np.std(entropy_arr)) if len(entropy_arr) else float("nan"),
        "low_conf_fraction": float(np.mean(confs_arr < 0.3)) if len(confs_arr) else float("nan"),
        "high_entropy_fraction": float(np.mean(entropy_arr > 0.4)) if len(entropy_arr) else float("nan"),
    }

    return stats, confs_arr, entropy_arr


def run_entropy_shift_analysis(
    model_path: str = "yolov8n.pt",
    source_yaml: str = "coco128.yaml",
    target_yamls: list[str] = None,
    output_dir: str = "experiments/results",
    max_images: int = 128,
) -> dict:
    """
    Compares source vs. one or more target domains, computes Wasserstein shift distance,
    and plots confidence and entropy distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    # 1. Evaluate Source Domain
    source_name = "Source (Clean in-distribution)"
    print(f"Profiling {source_name} on {source_yaml}...")
    src_stats, src_confs, src_entropy = collect_domain_confs_and_entropy(
        model, source_yaml, domain_name="source_clean", max_images=max_images
    )

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "source": src_stats,
        "targets": {},
        "divergences": {},
    }

    target_data = []

    # 2. Evaluate Target Domains
    if target_yamls is None:
        target_yamls = []

    for t_yaml in target_yamls:
        t_name = Path(t_yaml).stem
        print(f"Profiling Target Domain: {t_name} on {t_yaml}...")
        t_stats, t_confs, t_entropy = collect_domain_confs_and_entropy(
            model, t_yaml, domain_name=t_name, max_images=max_images
        )
        results["targets"][t_name] = t_stats

        # Compute Wasserstein distance on confidences and entropies
        w_dist_conf = float(wasserstein_distance(src_confs, t_confs)) if (len(src_confs) and len(t_confs)) else float("nan")
        w_dist_ent = float(wasserstein_distance(src_entropy, t_entropy)) if (len(src_entropy) and len(t_entropy)) else float("nan")

        results["divergences"][t_name] = {
            "wasserstein_distance_confidence": w_dist_conf,
            "wasserstein_distance_entropy": w_dist_ent,
            "entropy_increase_pct": float(
                ((t_stats["mean_entropy"] - src_stats["mean_entropy"]) / src_stats["mean_entropy"]) * 100.0
            ) if src_stats["mean_entropy"] > 0 else 0.0,
        }

        target_data.append((t_name, t_confs, t_entropy, t_stats))

    # 3. Generate Comparative Plot
    plot_path = os.path.join(output_dir, "entropy_distribution_comparison.png")
    generate_comparison_plots(
        src_name="Source Clean",
        src_confs=src_confs,
        src_entropy=src_entropy,
        targets=target_data,
        save_path=plot_path,
    )
    results["visualization"] = plot_path

    # 4. Save JSON Report
    json_path = os.path.join(output_dir, "entropy_shift_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Entropy Shift Analysis completed:")
    print(f"  Report: {json_path}")
    print(f"  Figure: {plot_path}")
    return results


def generate_comparison_plots(
    src_name: str,
    src_confs: np.ndarray,
    src_entropy: np.ndarray,
    targets: list[tuple[str, np.ndarray, np.ndarray, dict]],
    save_path: str,
):
    """
    Renders clean dual-panel distribution comparison figure (Confidence KDE & Entropy CDF/KDE).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Style
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Left: Confidence Distribution
    ax_conf = axes[0]
    ax_conf.hist(
        src_confs, bins=30, density=True, alpha=0.35, color=palette[0],
        label=f"{src_name} (μ={np.mean(src_confs):.2f})"
    )

    for i, (t_name, t_confs, _, _) in enumerate(targets):
        c = palette[(i + 1) % len(palette)]
        clean_label = t_name.replace("coco128_", "").replace("synthetic_", "")
        ax_conf.hist(
            t_confs, bins=30, density=True, alpha=0.35, color=c,
            label=f"{clean_label} (μ={np.mean(t_confs):.2f})"
        )

    ax_conf.set_title("Detection Confidence Distribution", fontsize=12, fontweight="bold")
    ax_conf.set_xlabel("Prediction Confidence p", fontsize=11)
    ax_conf.set_ylabel("Probability Density", fontsize=11)
    ax_conf.grid(True, linestyle="--", alpha=0.5)
    ax_conf.legend(fontsize=9, loc="upper right")

    # Right: Entropy Distribution
    ax_ent = axes[1]
    ax_ent.hist(
        src_entropy, bins=30, density=True, alpha=0.35, color=palette[0],
        label=f"{src_name} (H={np.mean(src_entropy):.3f})"
    )

    for i, (t_name, _, t_ent, _) in enumerate(targets):
        c = palette[(i + 1) % len(palette)]
        clean_label = t_name.replace("coco128_", "").replace("synthetic_", "")
        ax_ent.hist(
            t_ent, bins=30, density=True, alpha=0.35, color=c,
            label=f"{clean_label} (H={np.mean(t_ent):.3f})"
        )

    ax_ent.set_title("Prediction Entropy Distribution H(p)", fontsize=12, fontweight="bold")
    ax_ent.set_xlabel("Entropy H(p) = -p log(p) - (1-p) log(1-p)", fontsize=11)
    ax_ent.set_ylabel("Probability Density", fontsize=11)
    ax_ent.grid(True, linestyle="--", alpha=0.5)
    ax_ent.legend(fontsize=9, loc="upper right")

    fig.suptitle("Clinical-TTA-Edge | Domain Shift vs Entropy Divergence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Confidence Entropy Shift Analysis")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--source", default="coco128.yaml", help="Source domain dataset YAML")
    parser.add_argument("--targets", nargs="+", default=[
        "experiments/configs/coco128_synthetic_contrast.yaml",
        "experiments/configs/coco128_synthetic_brightness.yaml",
        "experiments/configs/coco128_synthetic_blur.yaml",
        "experiments/configs/coco128_synthetic_noise.yaml",
    ], help="Target domain dataset YAMLs")
    parser.add_argument("--max-images", type=int, default=128, help="Max images to evaluate")
    parser.add_argument("--output-dir", default="experiments/results", help="Output directory")
    args = parser.parse_args()

    # Filter targets that actually exist
    valid_targets = [t for t in args.targets if os.path.exists(t)]
    if not valid_targets:
        print("[WARN] None of the specified target YAMLs exist. Run dataset_setup.py --generate-shifts first.")
        sys.exit(1)

    run_entropy_shift_analysis(
        model_path=args.model,
        source_yaml=args.source,
        target_yamls=valid_targets,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
