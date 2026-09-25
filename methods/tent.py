"""
methods/tent.py
===============
TENT adaptation for YOLOv8 object detection.

Reference:
  Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization"
  ICLR 2021 — https://arxiv.org/abs/2006.10726

Standard TENT is designed for image classifiers with global softmax.
Adapting it to single-stage anchor-free detectors (like YOLOv8) requires:
  1. Spatially-structured detection entropy: computed over objectness confidences
     and normalized candidate class distributions across multi-scale feature grids.
  2. Edge-efficient parameter selection: adapting ONLY BatchNorm2d / LayerNorm
     affine scale (gamma) and bias (beta) parameters (99.7% of network parameters frozen).
  3. Running statistics frozen: keep running mean and running variance fixed to
     prevent batch-size-1 statistic instability.
  4. Proper parameter rollback / reset support to prevent unbounded drift across sessions.
"""

import time
import copy
from typing import Optional, Union, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
from ultralytics import YOLO


def detection_entropy(
    predictions: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    conf_threshold: float = 0.05,
    class_entropy_weight: float = 0.5,
) -> torch.Tensor:
    """
    Computes surrogate detection entropy objective for YOLOv8.

    Args:
        predictions: Raw output tensor from YOLOv8 eval forward pass.
                     Shape: [batch, 84, 8400] (for 80 classes + 4 box coords)
        conf_threshold: Minimum objectness confidence to consider.
        class_entropy_weight: Weight for normalized multi-class distribution entropy.

    Returns:
        Scalar entropy loss tensor with requires_grad=True.
    """
    if isinstance(predictions, (list, tuple)):
        preds = predictions[0]
    else:
        preds = predictions

    # Expected shape: [B, C+4, N_anchors] -> Transpose to [B, N_anchors, C+4]
    if preds.dim() == 3 and preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(1, 2)  # [B, 8400, 84]

    # Class probabilities / confidences: preds[..., 4:] (shape [B, N, num_classes])
    class_probs = preds[..., 4:]

    # Max confidence per anchor
    obj_conf, _ = class_probs.max(dim=-1)  # [B, N]

    mask = obj_conf > conf_threshold
    if not mask.any():
        # Fall back to top-50 candidate anchors if no anchor exceeds threshold
        k = min(50, obj_conf.shape[-1])
        top_confs, top_indices = torch.topk(obj_conf.view(-1), k=k)
        selected_confs = top_confs
        selected_classes = class_probs.view(-1, class_probs.shape[-1])[top_indices]
    else:
        selected_confs = obj_conf[mask]
        selected_classes = class_probs[mask]

    # Binary objectness entropy: H(p) = -p log(p) - (1-p) log(1-p)
    p = torch.clamp(selected_confs, 1e-6, 1.0 - 1e-6)
    obj_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()

    # Normalized class entropy for candidate boxes
    class_sum = selected_classes.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    p_cls = torch.clamp(selected_classes / class_sum, 1e-6, 1.0)
    cls_entropy = -(p_cls * torch.log(p_cls)).sum(dim=-1).mean()

    total_loss = obj_entropy + class_entropy_weight * cls_entropy
    return total_loss


class TENTAdapter:
    """
    Test-time entropy minimization adapter for YOLOv8.
    Only updates normalization layer affine parameters (scale and shift).
    All backbone, neck, and detection head weights remain frozen.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        lr: float = 1e-4,
        steps: int = 1,
        conf_threshold: float = 0.05,
        device: str = "cpu",
        optimizer_type: str = "Adam",
    ):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.steps = steps
        self.lr = lr
        self.model_path = model_path

        # Load YOLO model
        self._yolo = YOLO(model_path)
        self.model = self._yolo.model.to(self.device)

        # Configure trainable parameters (BN affine only)
        self._configure_parameters()

        # Cache initial clean parameter state for lightweight rollback
        self._clean_param_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Optimizer targeting only active normalization params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        else:
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9)

    def _configure_parameters(self):
        """
        Freeze all parameters except BatchNorm2d affine parameters.
        Set eval mode so running mean and running variance remain constant.
        """
        self.model.eval()
        self.model.requires_grad_(False)

        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                # Enable gradients only for scale (weight) and shift (bias)
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    @property
    def adapted_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def reset(self):
        """Cleanly restore initial pretrained weights without reloading from disk."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._clean_param_state:
                    param.copy_(self._clean_param_state[name])
        # Reset optimizer state
        self.optimizer.state.clear()

    def preprocess_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Converts input image to normalized [1, 3, 640, 640] PyTorch float tensor."""
        if isinstance(image, torch.Tensor):
            t = image.to(self.device)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            return t.float()

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to read image at: {image}")
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Resize to 640x640 standard YOLO input resolution
        img = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def adapt_step(self, image_tensor: torch.Tensor) -> float:
        """
        Performs one or more gradient descent steps on detection entropy.

        Returns:
            Scalar float value of entropy loss before adaptation.
        """
        initial_loss_val = 0.0

        for s in range(self.steps):
            self.optimizer.zero_grad()
            with torch.enable_grad():
                out = self.model(image_tensor)
                loss = detection_entropy(out, conf_threshold=self.conf_threshold)

                if s == 0:
                    initial_loss_val = float(loss.item())

                loss.backward()
                self.optimizer.step()

        return initial_loss_val

    def adapt_and_predict(self, image: Union[str, np.ndarray], conf: float = 0.25):
        """
        Adapts model on test image and returns Ultralytics Results object.
        """
        img_tensor = self.preprocess_image(image)

        # Adaptation step
        loss = self.adapt_step(img_tensor)

        # Inference through standard predictor
        results = self._yolo(image, verbose=False, conf=conf)
        return results, loss

    def profile_latency(
        self,
        sample_images: List[str],
        n_warmup: int = 2,
    ) -> dict:
        """
        Profiles inference latency: Baseline inference vs TENT Adaptation + Inference.
        """
        if not sample_images:
            return {}

        self.reset()
        tensors = [self.preprocess_image(p) for p in sample_images[:10]]

        # Warmup
        for t in tensors[:n_warmup]:
            _ = self.model(t)

        # Measure baseline inference
        baseline_times = []
        for t in tensors:
            t0 = time.perf_counter()
            _ = self.model(t)
            baseline_times.append((time.perf_counter() - t0) * 1000.0)

        # Measure TENT adaptation + inference
        self.reset()
        tent_times = []
        for t in tensors:
            t0 = time.perf_counter()
            _ = self.adapt_step(t)
            _ = self.model(t)
            tent_times.append((time.perf_counter() - t0) * 1000.0)

        self.reset()
        return {
            "baseline_mean_ms": float(np.mean(baseline_times)),
            "baseline_median_ms": float(np.median(baseline_times)),
            "tent_mean_ms": float(np.mean(tent_times)),
            "tent_median_ms": float(np.median(tent_times)),
            "overhead_ms": float(np.mean(tent_times) - np.mean(baseline_times)),
            "overhead_pct": float(((np.mean(tent_times) - np.mean(baseline_times)) / np.mean(baseline_times)) * 100.0),
        }


# Export alias matching run_experiments.py expectation
YOLOv8TENT = TENTAdapter


if __name__ == "__main__":
    print("TENT adapter verification...")
    adapter = TENTAdapter(model_path="yolov8n.pt", device="cpu")
    n_adapt = adapter.adapted_param_count
    n_total = adapter.total_param_count
    print(f"  Trainable Parameters: {n_adapt:,} / {n_total:,} ({(n_adapt/n_total):.2%})")

    dummy_input = torch.randn(1, 3, 640, 640)
    initial_loss = adapter.adapt_step(dummy_input)
    print(f"  Single dummy adapt step successful. Loss: {initial_loss:.4f}")
    adapter.reset()
    print("  Model reset successful.")
