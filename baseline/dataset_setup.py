"""
baseline/dataset_setup.py
==========================
Dataset preparation and synthetic domain shift generators for clinical-tta-edge.

NO private or IRB-restricted data is used.
All datasets listed here are publicly available.

Supported Datasets:
-------------------
1. COCO-128 / COCO 2017 val (source domain benchmark)
2. MOT17 (target domain 1 — indoor surveillance, camera-angle shift)
3. HDA Person Dataset (target domain 2 — hospital corridors)
4. Synthetic shifts on COCO (target domain 3 — controlled clinical proxies):
   - Brightness reduction (night-mode / low-illumination patient rooms)
   - Contrast reduction (foggy lens / clinical dynamic range variation)
   - Gaussian defocus blur (camera vibration / subject motion)
   - Gaussian sensor noise (low-cost edge CMOS sensor noise)
"""

import os
import sys
import shutil
from pathlib import Path
import yaml
import numpy as np
import cv2

DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.abspath("./data"))
EXPERIMENT_CONFIG_DIR = os.path.abspath("./experiments/configs")


def find_source_dataset() -> dict:
    """
    Locates available source dataset: checks local data/coco, ../datasets/coco128,
    or falls back to triggering Ultralytics coco128 auto-download.
    """
    # 1. Check ../datasets/coco128 or ./datasets/coco128
    candidate_paths = [
        os.path.abspath("./data/coco"),
        os.path.abspath("../datasets/coco128"),
        os.path.abspath("./datasets/coco128"),
        os.path.expanduser("~/datasets/coco128"),
    ]
    for p in candidate_paths:
        img_dir = os.path.join(p, "images", "train2017")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(p, "images", "val2017")
        if os.path.isdir(img_dir) and len(list(Path(img_dir).glob("*.jpg"))) > 0:
            label_dir = img_dir.replace("images", "labels")
            return {
                "name": "coco128" if "coco128" in p else "coco",
                "root": p,
                "images_dir": img_dir,
                "labels_dir": label_dir,
            }

    # Fall back to Ultralytics check_det_dataset
    try:
        from ultralytics.data.utils import check_det_dataset
        meta = check_det_dataset("coco128.yaml")
        img_dir = str(meta["val"])
        label_dir = img_dir.replace("images", "labels")
        return {
            "name": "coco128",
            "root": str(meta["path"]),
            "images_dir": img_dir,
            "labels_dir": label_dir,
        }
    except Exception as e:
        print(f"[WARN] Could not auto-resolve source dataset: {e}")
        return {}


def apply_synthetic_shift(
    src_img_dir: str,
    src_label_dir: str,
    output_root: str,
    shift_type: str = "brightness",
    shift_param: float = -50.0,
    max_images: int = 128,
) -> str:
    """
    Apply a controlled perturbation to source images and copy corresponding labels.
    Generates a valid YOLO-format dataset structure:
      <output_root>/images/val/
      <output_root>/labels/val/

    Args:
        src_img_dir: Directory containing input images.
        src_label_dir: Directory containing input YOLO labels.
        output_root: Root directory for the generated shifted dataset.
        shift_type: "brightness", "contrast", "blur", or "noise".
        shift_param: Magnitude parameter.
        max_images: Maximum number of images to process.

    Returns:
        Path to output_root.
    """
    dst_img_dir = os.path.join(output_root, "images", "val")
    dst_label_dir = os.path.join(output_root, "labels", "val")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    src_images = sorted(list(Path(src_img_dir).glob("*.jpg")) + list(Path(src_img_dir).glob("*.png")))[:max_images]
    print(f"Applying {shift_type} shift (param={shift_param}) to {len(src_images)} images...")

    for img_path in src_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if shift_type == "brightness":
            shifted = np.clip(img.astype(np.int32) + int(shift_param), 0, 255).astype(np.uint8)
        elif shift_type == "contrast":
            shifted = np.clip(img.astype(np.float32) * float(shift_param), 0, 255).astype(np.uint8)
        elif shift_type == "blur":
            k = int(shift_param)
            if k % 2 == 0:
                k += 1
            shifted = cv2.GaussianBlur(img, (k, k), 0)
        elif shift_type == "noise":
            noise = np.random.normal(0, float(shift_param), img.shape).astype(np.float32)
            shifted = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            shifted = img

        dst_img_path = os.path.join(dst_img_dir, img_path.name)
        cv2.imwrite(dst_img_path, shifted)

        # Copy matching label if it exists
        label_name = img_path.stem + ".txt"
        src_label_path = os.path.join(src_label_dir, label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, os.path.join(dst_label_dir, label_name))

    print(f"[OK] Generated {len(src_images)} shifted samples in {output_root}")
    return output_root


def make_shifted_data_yaml(
    dataset_root: str,
    shift_name: str,
    output_yaml_path: str,
    nc: int = 80,
) -> str:
    """Generate YOLO data.yaml configuration pointing to the shifted dataset."""
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    abs_root = os.path.abspath(dataset_root).replace("\\", "/")

    # Default COCO classes
    coco_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
        19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
        24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
        29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
        34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
        38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
        42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
        47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
        52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
        57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
        61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
        66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
        78: "hair drier", 79: "toothbrush"
    }

    content = {
        "path": abs_root,
        "train": "images/val",
        "val": "images/val",
        "nc": nc,
        "names": coco_names,
    }

    with open(output_yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)

    print(f"[OK] Written shifted dataset YAML to {output_yaml_path}")
    return output_yaml_path


def generate_standard_shifts(max_images: int = 128) -> dict:
    """
    Generates all standardized clinical domain shift proxies specified in experimental design:
    1. synthetic_brightness (-50)
    2. synthetic_contrast (0.4)
    3. synthetic_blur (kernel=7)
    4. synthetic_noise (sigma=25)
    """
    src_info = find_source_dataset()
    if not src_info:
        raise RuntimeError("No source dataset found. Please download coco128 or coco val.")

    print(f"Using source dataset: {src_info['name']} at {src_info['root']}")

    shifts = [
        {"name": "synthetic_brightness", "type": "brightness", "param": -50.0},
        {"name": "synthetic_contrast", "type": "contrast", "param": 0.4},
        {"name": "synthetic_blur", "type": "blur", "param": 7.0},
        {"name": "synthetic_noise", "type": "noise", "param": 25.0},
    ]

    generated_configs = {}
    for s in shifts:
        out_root = os.path.join(DATASET_ROOT, f"{src_info['name']}_{s['name']}")
        apply_synthetic_shift(
            src_img_dir=src_info["images_dir"],
            src_label_dir=src_info["labels_dir"],
            output_root=out_root,
            shift_type=s["type"],
            shift_param=s["param"],
            max_images=max_images,
        )
        yaml_path = os.path.join(EXPERIMENT_CONFIG_DIR, f"{src_info['name']}_{s['name']}.yaml")
        make_shifted_data_yaml(out_root, s["name"], yaml_path)
        generated_configs[s["name"]] = yaml_path

    return generated_configs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset setup and synthetic shift generator")
    parser.add_argument("--generate-shifts", action="store_true", help="Generate all standard synthetic shifts")
    parser.add_argument("--max-images", type=int, default=128, help="Max images to perturb per shift")
    args = parser.parse_args()

    print("clinical-tta-edge | Dataset Setup & Shift Generator")
    print("=" * 60)
    src = find_source_dataset()
    if src:
        print(f"[FOUND] Source dataset: {src['name']} ({src['images_dir']})")
    else:
        print("[MISSING] Source dataset not located.")

    if args.generate_shifts:
        configs = generate_standard_shifts(max_images=args.max_images)
        print("\nAll synthetic shifts generated successfully:")
        for k, v in configs.items():
            print(f"  - {k}: {v}")
