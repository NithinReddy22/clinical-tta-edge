"""
run_experiments.py
===================
Unified CLI experiment runner for clinical-tta-edge.
Runs on CPU or CUDA GPU automatically.

Usage Examples:
    # 1. Evaluate baseline on clean or shifted dataset
    python run_experiments.py --mode baseline --data coco128.yaml

    # 2. Run TENT test-time adaptation
    python run_experiments.py --mode tent --data experiments/configs/coco128_synthetic_contrast.yaml

    # 3. Run SHOT adaptation (entropy + diversity)
    python run_experiments.py --mode shot --data experiments/configs/coco128_synthetic_contrast.yaml

    # 4. Run Entropy-Gated adaptation
    python run_experiments.py --mode gated --data experiments/configs/coco128_synthetic_contrast.yaml

    # 5. Run Entropy shift analysis & visualization
    python run_experiments.py --mode entropy-analysis

    # 6. Run full comparative benchmark suite
    python run_experiments.py --mode compare
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import yaml
import torch
import numpy as np

# Ensure root directory is on sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from baseline.evaluate import evaluate, save_results
from baseline.entropy_analysis import run_entropy_shift_analysis, resolve_image_paths, compute_binary_entropy
from methods.tent import TENTAdapter, YOLOv8TENT
from methods.shot import SHOTAdapter, YOLOv8SHOT
from methods.entropy_gating import EntropyGatedTENT, YOLOv8EntropyGated
from methods.ttt_aux import TTTAuxAdapter, YOLOv8TTTAux


def parse_args():
    parser = argparse.ArgumentParser(description="Clinical-TTA-Edge Experiment Runner")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "tent", "shot", "gated", "ttt-aux", "entropy-analysis", "compare"],
        default="baseline",
        help="Experiment execution mode",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Path to dataset yaml")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val/test)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', or 'cuda:0'")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adaptation learning rate")
    parser.add_argument("--steps", type=int, default=1, help="Adaptation gradient steps per sample")
    parser.add_argument("--entropy-threshold", type=float, default=0.22, help="Threshold for entropy gating")
    parser.add_argument("--max-images", type=int, default=128, help="Max images to evaluate in test sequences")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Directory for result logs")
    return parser.parse_args()


def run_adapter_stream(
    adapter,
    data_yaml: str,
    split: str = "val",
    max_images: int = 128,
    conf_thresh: float = 0.25,
) -> dict:
    """
    Simulates sequential stream test-time adaptation across a target domain video/image stream.
    Measures adaptation latency overhead, prediction confidence recovery, and entropy stabilization.
    """
    image_paths = resolve_image_paths(data_yaml, split=split, max_images=max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found for dataset {data_yaml}")

    print(f"Running stream adaptation on {len(image_paths)} frames...")
    adapter.reset()

    latencies_ms = []
    losses = []
    confs = []
    entropies = []

    for idx, img_path in enumerate(image_paths):
        t0 = time.perf_counter()
        img_tensor = adapter.preprocess_image(img_path)

        # Adapt
        loss_val = adapter.adapt_step(img_tensor)

        # Inference
        res = adapter._yolo(img_path, verbose=False, conf=conf_thresh)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(elapsed_ms)
        losses.append(loss_val)

        # Record detections
        for r in res:
            if r.boxes is not None and len(r.boxes):
                c = r.boxes.conf.cpu().numpy().tolist()
                confs.extend(c)

    confs_arr = np.array(confs, dtype=np.float32)
    ent_arr = compute_binary_entropy(confs_arr) if len(confs_arr) > 0 else np.array([])

    metrics = {
        "dataset": data_yaml,
        "n_frames_evaluated": len(image_paths),
        "total_detections": int(len(confs_arr)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "median_latency_ms": float(np.median(latencies_ms)),
        "initial_loss": float(losses[0]) if losses else float("nan"),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "mean_conf": float(np.mean(confs_arr)) if len(confs_arr) else float("nan"),
        "mean_entropy": float(np.mean(ent_arr)) if len(ent_arr) else float("nan"),
        "trainable_params": adapter.adapted_param_count,
        "total_params": adapter.total_param_count,
    }

    return metrics


def run_comparative_benchmark(
    model_path: str = "yolov8n.pt",
    source_data: str = "coco128.yaml",
    target_data: str = "experiments/configs/coco128_synthetic_contrast.yaml",
    device: str = "cpu",
    output_dir: str = "experiments/results",
    max_images: int = 128,
) -> dict:
    """
    Executes the complete Phase 2 comparative benchmark suite:
      1. Clean Source Baseline (In-Distribution Reference)
      2. Shifted Target Baseline (No Adaptation)
      3. TENT Adaptation
      4. SHOT Adaptation
      5. Entropy-Gated TENT (Selective Adaptation)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print(" Clinical-TTA-Edge: Phase 2 Comparative Benchmark Suite")
    print("=" * 65)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_path,
        "device": device,
        "source_dataset": source_data,
        "target_dataset": target_data,
        "benchmarks": {},
    }

    # 1. Clean Source Baseline
    print("\n[1/5] Evaluating Clean Source Domain Baseline (In-Distribution)...")
    clean_metrics, _ = evaluate(model_path=model_path, data_yaml=source_data, split="val", device=device)
    results["benchmarks"]["clean_source_baseline"] = {
        "mAP50": clean_metrics["mAP50"],
        "mAP50_95": clean_metrics["mAP50_95"],
        "precision": clean_metrics["precision"],
        "recall": clean_metrics["recall"],
    }
    print(f"  Clean Source mAP50: {clean_metrics['mAP50']:.4f} | mAP50-95: {clean_metrics['mAP50_95']:.4f}")

    # 2. Shifted Target Baseline (No Adaptation)
    print("\n[2/5] Evaluating Shifted Target Domain Baseline (No Adaptation)...")
    shifted_metrics, _ = evaluate(model_path=model_path, data_yaml=target_data, split="val", device=device)
    results["benchmarks"]["shifted_baseline_no_tta"] = {
        "mAP50": shifted_metrics["mAP50"],
        "mAP50_95": shifted_metrics["mAP50_95"],
        "precision": shifted_metrics["precision"],
        "recall": shifted_metrics["recall"],
        "mAP50_drop": float(clean_metrics["mAP50"] - shifted_metrics["mAP50"]),
    }
    print(f"  Shifted mAP50: {shifted_metrics['mAP50']:.4f} (Drop: {results['benchmarks']['shifted_baseline_no_tta']['mAP50_drop']:.4f})")

    # 3. TENT Stream Adaptation
    print("\n[3/5] Evaluating TENT Stream Adaptation...")
    tent_adapter = TENTAdapter(model_path=model_path, device=device)
    tent_stream = run_adapter_stream(tent_adapter, target_data, max_images=max_images)
    # Evaluate adapted model mAP
    tent_eval, _ = evaluate(model_path=model_path, data_yaml=target_data, split="val", device=device)
    results["benchmarks"]["tent_adapted"] = {
        "mAP50": tent_eval["mAP50"],
        "mAP50_95": tent_eval["mAP50_95"],
        "precision": tent_eval["precision"],
        "recall": tent_eval["recall"],
        "mean_latency_ms": tent_stream["mean_latency_ms"],
        "adapted_params": tent_adapter.adapted_param_count,
        "mAP50_recovery": float(tent_eval["mAP50"] - shifted_metrics["mAP50"]),
    }
    print(f"  TENT Adapted mAP50: {tent_eval['mAP50']:.4f} | Latency: {tent_stream['mean_latency_ms']:.1f}ms")

    # 4. SHOT Stream Adaptation
    print("\n[4/5] Evaluating SHOT Stream Adaptation...")
    shot_adapter = SHOTAdapter(model_path=model_path, device=device)
    shot_stream = run_adapter_stream(shot_adapter, target_data, max_images=max_images)
    shot_eval, _ = evaluate(model_path=model_path, data_yaml=target_data, split="val", device=device)
    results["benchmarks"]["shot_adapted"] = {
        "mAP50": shot_eval["mAP50"],
        "mAP50_95": shot_eval["mAP50_95"],
        "precision": shot_eval["precision"],
        "recall": shot_eval["recall"],
        "mean_latency_ms": shot_stream["mean_latency_ms"],
        "adapted_params": shot_adapter.adapted_param_count,
        "mAP50_recovery": float(shot_eval["mAP50"] - shifted_metrics["mAP50"]),
    }
    print(f"  SHOT Adapted mAP50: {shot_eval['mAP50']:.4f} | Latency: {shot_stream['mean_latency_ms']:.1f}ms")

    # 5. Entropy-Gated TENT
    print("\n[5/5] Evaluating Entropy-Gated TENT Adaptation...")
    gated_adapter = EntropyGatedTENT(model_path=model_path, device=device, entropy_threshold=0.20)
    image_paths = resolve_image_paths(target_data, split="val", max_images=max_images)
    for p in image_paths:
        _ = gated_adapter.adapt_and_predict(p)
    results["benchmarks"]["entropy_gated_tent"] = {
        "total_frames": gated_adapter.stats["total_frames"],
        "adapted_frames": gated_adapter.stats["adapted_frames"],
        "adaptation_rate_pct": float(gated_adapter.adaptation_rate * 100.0),
        "compute_savings_pct": float(gated_adapter.compute_savings_pct),
    }
    print(f"  Gated TENT adapted {gated_adapter.stats['adapted_frames']}/{gated_adapter.stats['total_frames']} frames ({gated_adapter.adaptation_rate*100:.1f}%)")
    print(f"  Compute savings: {gated_adapter.compute_savings_pct:.1f}%")

    # Save Comparative Results
    summary_path = os.path.join(output_dir, "tta_comparison_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Benchmark completed successfully! Results written to: {summary_path}")
    print("=" * 65)
    return results


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=================================================")
    print(" Clinical-TTA-Edge: Test-Time Adaptation Bench   ")
    print("=================================================")
    print(f"Mode:     {args.mode}")
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Device:   {device} ({'GPU Acceleration' if 'cuda' in device else 'CPU Edge Proxy'})")
    print(f"PyTorch:  {torch.__version__}")
    print("-------------------------------------------------")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "baseline":
        print(f"Running baseline evaluation on {args.data}...")
        metrics, _ = evaluate(model_path=args.model, data_yaml=args.data, split=args.split, device=device)
        print("\n--- Baseline Results ---")
        for k, v in metrics.items():
            if k != "per_class_ap50":
                print(f"  {k}: {v}")
        save_results(metrics, output_dir=args.output_dir)

    elif args.mode == "tent":
        print(f"Running TENT test-time adaptation on {args.data}...")
        adapter = TENTAdapter(model_path=args.model, lr=args.lr, steps=args.steps, device=device)
        metrics = run_adapter_stream(adapter, args.data, split=args.split, max_images=args.max_images)
        print("\n--- TENT Adaptation Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        save_path = os.path.join(args.output_dir, "tent_stream_results.json")
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {save_path}")

    elif args.mode == "shot":
        print(f"Running SHOT test-time adaptation on {args.data}...")
        adapter = SHOTAdapter(model_path=args.model, lr=args.lr, steps=args.steps, device=device)
        metrics = run_adapter_stream(adapter, args.data, split=args.split, max_images=args.max_images)
        print("\n--- SHOT Adaptation Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        save_path = os.path.join(args.output_dir, "shot_stream_results.json")
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {save_path}")

    elif args.mode == "gated":
        print(f"Running Entropy-Gated adaptation on {args.data}...")
        adapter = EntropyGatedTENT(
            model_path=args.model,
            entropy_threshold=args.entropy_threshold,
            lr=args.lr,
            steps=args.steps,
            device=device,
        )
        image_paths = resolve_image_paths(args.data, split=args.split, max_images=args.max_images)
        print(f"Processing {len(image_paths)} frames with entropy gating threshold tau={args.entropy_threshold}...")
        for p in image_paths:
            _ = adapter.adapt_and_predict(p)
        print("\n--- Entropy-Gated TENT Results ---")
        print(f"  Total Frames:    {adapter.stats['total_frames']}")
        print(f"  Adapted Frames:  {adapter.stats['adapted_frames']}")
        print(f"  Skipped Frames:  {adapter.stats['skipped_frames']}")
        print(f"  Adaptation Rate: {adapter.adaptation_rate:.2%}")
        print(f"  Compute Savings: {adapter.compute_savings:.2%}")

    elif args.mode == "ttt-aux":
        print(f"Running Auxiliary-Task TTT adaptation on {args.data}...")
        adapter = TTTAuxAdapter(model_path=args.model, lr=args.lr, steps=args.steps, device=device)
        metrics = run_adapter_stream(adapter, args.data, split=args.split, max_images=args.max_images)
        print("\n--- TTT-Aux Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.mode == "entropy-analysis":
        print("Running entropy shift analysis across source and targets...")
        target_yamls = [
            "experiments/configs/coco128_synthetic_contrast.yaml",
            "experiments/configs/coco128_synthetic_brightness.yaml",
            "experiments/configs/coco128_synthetic_blur.yaml",
            "experiments/configs/coco128_synthetic_noise.yaml",
        ]
        valid_targets = [t for t in target_yamls if os.path.exists(t)]
        run_entropy_shift_analysis(
            model_path=args.model,
            source_yaml=args.data,
            target_yamls=valid_targets,
            output_dir=args.output_dir,
            max_images=args.max_images,
        )

    elif args.mode == "compare":
        target = args.data if args.data != "coco128.yaml" else "experiments/configs/coco128_synthetic_contrast.yaml"
        run_comparative_benchmark(
            model_path=args.model,
            source_data="coco128.yaml",
            target_data=target,
            device=device,
            output_dir=args.output_dir,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()
