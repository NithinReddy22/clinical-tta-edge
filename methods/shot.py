"""
methods/shot.py
===============
SHOT (Source Hypothesis Transfer) adaptation for YOLOv8 object detection.

Reference:
  Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis
  Transfer for Unsupervised Domain Adaptation", ICML 2020.
  https://arxiv.org/abs/2002.08546

Adapted for single-stage anchor-free detectors:
  In addition to proposal-level entropy minimization, SHOT introduces an
  information-theoretic diversity term:
      L_SHOT = L_entropy - beta * L_diversity

  Where:
    - L_entropy: minimizes uncertainty on detected proposals.
    - L_diversity: maximizes entropy of the average class prediction distribution
      over detected candidate anchors:
          L_div = - sum_{c} p_bar_c * log(p_bar_c + eps)
      This prevents representation collapse where all detections collapse to
      a single dominant class under severe shift.
"""

from typing import Union, List, Tuple
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO
import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from methods.tent import TENTAdapter


def detection_shot_loss(
    predictions: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    conf_threshold: float = 0.05,
    diversity_weight: float = 0.4,
) -> Tuple[torch.Tensor, float, float]:
    """
    Computes SHOT objective for YOLOv8 detections:
      Loss = Entropy(p) - diversity_weight * Entropy(mean_p)

    Returns:
        (total_loss, entropy_val, diversity_val)
    """
    if isinstance(predictions, (list, tuple)):
        preds = predictions[0]
    else:
        preds = predictions

    if preds.dim() == 3 and preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(1, 2)  # [B, N, C+4]

    class_probs = preds[..., 4:]
    obj_conf, _ = class_probs.max(dim=-1)

    mask = obj_conf > conf_threshold
    if not mask.any():
        k = min(50, obj_conf.shape[-1])
        top_confs, top_indices = torch.topk(obj_conf.view(-1), k=k)
        selected_confs = top_confs
        selected_classes = class_probs.view(-1, class_probs.shape[-1])[top_indices]
    else:
        selected_confs = obj_conf[mask]
        selected_classes = class_probs[mask]

    # 1. Proposal Entropy Loss
    p = torch.clamp(selected_confs, 1e-6, 1.0 - 1e-6)
    entropy_loss = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()

    # 2. Diversity Loss across class distribution
    # Normalized class probabilities per box
    class_sum = selected_classes.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p_norm = selected_classes / class_sum
    # Average class distribution across all candidate boxes
    p_bar = p_norm.mean(dim=0).clamp(min=1e-6)
    # Shannon entropy of the mean distribution (higher = more diverse classes)
    diversity = -(p_bar * torch.log(p_bar)).sum()

    total_loss = entropy_loss - diversity_weight * diversity
    return total_loss, float(entropy_loss.item()), float(diversity.item())


class SHOTAdapter(TENTAdapter):
    """
    SHOT Adapter for YOLOv8.
    Inherits normalization-layer adaptation and parameter rollback from TENTAdapter,
    while overriding the adaptation objective with the Information Maximization loss.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        lr: float = 1e-4,
        steps: int = 1,
        conf_threshold: float = 0.05,
        diversity_weight: float = 0.4,
        device: str = "cpu",
        optimizer_type: str = "Adam",
    ):
        super().__init__(
            model_path=model_path,
            lr=lr,
            steps=steps,
            conf_threshold=conf_threshold,
            device=device,
            optimizer_type=optimizer_type,
        )
        self.diversity_weight = diversity_weight

    def adapt_step(self, image_tensor: torch.Tensor) -> float:
        """
        Executes gradient descent step on the SHOT objective.
        """
        initial_loss_val = 0.0

        for s in range(self.steps):
            self.optimizer.zero_grad()
            with torch.enable_grad():
                out = self.model(image_tensor)
                loss, ent, div = detection_shot_loss(
                    out,
                    conf_threshold=self.conf_threshold,
                    diversity_weight=self.diversity_weight,
                )

                if s == 0:
                    initial_loss_val = float(loss.item())

                loss.backward()
                self.optimizer.step()

        return initial_loss_val


YOLOv8SHOT = SHOTAdapter


if __name__ == "__main__":
    print("SHOT adapter verification...")
    adapter = SHOTAdapter(model_path="yolov8n.pt", device="cpu")
    print(f"  Trainable Parameters: {adapter.adapted_param_count:,}")
    dummy_input = torch.randn(1, 3, 640, 640)
    loss = adapter.adapt_step(dummy_input)
    print(f"  SHOT dummy adapt step successful. Loss: {loss:.4f}")
    adapter.reset()
    print("  Model reset successful.")
