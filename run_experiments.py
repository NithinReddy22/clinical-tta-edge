\"\"\"
Main CLI experiment runner for clinical-tta-edge.
Runs on CPU or CUDA GPU automatically.

Usage:
    python run_experiments.py --mode baseline --data coco128.yaml --device auto
\"\"\"

import argparse
import sys
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description=\"Clinical-TTA-Edge Experiment Runner\")
    parser.add_argument(\"--mode\", type=str, choices=[\"baseline\", \"tent\"], default=\"baseline\",
                        help=\"Experiment mode (baseline or tent adaptation)\")
    parser.add_argument(\"--model\", type=str, default=\"yolov8n.pt\", help=\"Path to YOLO model\")
    parser.add_argument(\"--data\", type=str, default=\"coco128.yaml\", help=\"Path to dataset yaml\")
    parser.add_argument(\"--split\", type=str, default=\"val\", help=\"Dataset split (val/test)\")
    parser.add_argument(\"--device\", type=str, default=\"auto\",
                        help=\"Device: 'auto', 'cpu', 'cuda', or 'cuda:0'\")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.device == \"auto\":
        device = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"
    else:
        device = args.device

    print(\"=================================================\")
    print(\" Clinical-TTA-Edge: Test-Time Adaptation Bench   \")
    print(\"=================================================\")
    print(f\"Mode:     {args.mode}\")
    print(f\"Model:    {args.model}\")
    print(f\"Data:     {args.data}\")
    print(f\"Device:   {device} ({'GPU Acceleration' if 'cuda' in device else 'CPU Proxy'})\")
    print(f\"PyTorch:  {torch.__version__}\")
    print(\"-------------------------------------------------\")

    if args.mode == \"baseline\":
        from baseline.evaluate import evaluate
        print(\"Running baseline evaluation...\")
        metrics = evaluate(model_path=args.model, data_yaml=args.data, split=args.split, device=device)
        print(\"Evaluation completed successfully.\")
    elif args.mode == \"tent\":
        print(\"Running TENT adaptation benchmark...\")
        from methods.tent import YOLOv8TENT
        print(\"TENT module loaded. Running adaptation pipeline...\")

if __name__ == \"__main__\":
    main()
