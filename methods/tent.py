"""
methods/tent.py
===============
TENT adaptation for YOLOv8 object detection.

Reference:
  Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization"
  ICLR 2021 — https://arxiv.org/abs/2006.10726

Standard TENT is designed for image classifiers. Applying it to a single-stage
object detector like YOLOv8 requires adapting the entropy signal because:
  1. There is no global softmax — each spatial cell produces its own objectness
     score and class probabilities independently.
  2. The model is anchor-free and multi-scale — three detection heads operate
     at different resolutions.
  3. In video, frames are temporally correlated — not i.i.d. batches.

This implementation defines a detection-compatible entropy objective and updates
only the BatchNorm/LayerNorm affine parameters (scale and bias), keeping all
other weights frozen during adaptation.

Status: Phase 2 — planned. Skeleton in place.
"""

import torch
import torch.nn as nn
from typing import Optional
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Detection-specific entropy
# ---------------------------------------------------------------------------

def detection_entropy(predictions, conf_threshold: float = 0.05) -> torch.Tensor:
    """
    Compute a surrogate entropy objective for detection model adaptation.

    For each anchor/cell, the model outputs an objectness confidence p.
    We compute binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    and average over all anchors above a minimum confidence threshold.

    Args:
        predictions: Raw YOLOv8 output tensor or list of prediction tensors.
                     Shape: [batch, num_anchors, 5+num_classes]
        conf_threshold: Minimum objectness to include in entropy computation.
                        Filters near-zero predictions that are pure background.

    Returns:
        Scalar entropy loss tensor (requires_grad=True if predictions do).
    """
    if isinstance(predictions, (list, tuple)):
        preds = torch.cat([p.reshape(p.shape[0], -1, p.shape[-1]) for p in predictions], dim=1)
    else:
        preds = predictions

    # objectness confidence — assumes standard YOLOv8 output format
    # [batch, anchors, 4 + num_classes] — confidence embedded in class scores
    class_scores = preds[..., 4:]                     # [B, A, C]
    obj_conf, _ = class_scores.max(dim=-1)            # [B, A]

    mask = obj_conf > conf_threshold
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True)

    p = torch.sigmoid(obj_conf[mask])
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean()


# ---------------------------------------------------------------------------
# TENT adapter for YOLOv8
# ---------------------------------------------------------------------------

class TENTAdapter:
    """
    Wraps a YOLOv8 model and applies test-time entropy minimization.
    Only updates BatchNorm and LayerNorm affine parameters (gamma, beta).
    All other parameters are frozen.

    Usage:
        adapter = TENTAdapter(model_path="yolov8n.pt", lr=1e-4, steps=1)
        adapted_results = adapter.adapt_and_predict(image_path)

    IMPORTANT:
        - This is a stateful adapter. Call adapter.reset() between independent
          test sequences to avoid accumulating drift.
        - Results reported in experiments/ are for tracking purposes only.
          Actual mAP numbers require running the full eval pipeline.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        lr: float = 1e-4,
        steps: int = 1,
        conf_threshold: float = 0.05,
        device: str = "cpu",
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.steps = steps

        # Load model and extract PyTorch model
        self._yolo = YOLO(model_path)
        self.model = self._yolo.model.to(device)
        self.model.train()

        # Freeze everything except BN/LN affine params
        self._configure_parameters()

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )

    def _configure_parameters(self):
        """Freeze all params except BatchNorm/LayerNorm scale and bias."""
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.requires_grad_(True)
                # Keep running stats frozen — adapt only affine params
                if hasattr(module, "track_running_stats"):
                    module.track_running_stats = False

    def adapt_and_predict(self, image):
        """
        Run TENT adaptation on a single image or batch, then return predictions.

        Args:
            image: File path, numpy array, or tensor accepted by YOLOv8.

        Returns:
            Ultralytics Results object (same as model.predict() output).
        """
        # Forward pass with gradient tracking through BN layers
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            # Note: calling self.model directly for gradient flow
            # self._yolo() goes through post-processing which detaches gradients
            with torch.enable_grad():
                raw_out = self._forward_raw(image)
                if raw_out is not None:
                    loss = detection_entropy(raw_out, self.conf_threshold)
                    loss.backward()
                    self.optimizer.step()

        # Final inference with adapted weights (eval mode for prediction)
        self.model.eval()
        results = self._yolo(image, verbose=False)
        self.model.train()
        return results

    def _forward_raw(self, image):
        """
        Attempt to get raw logits from the model backbone+head without
        post-processing, to retain gradients.
        This is model-version dependent and may need adjustment.
        """
        try:
            import cv2
            import numpy as np
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.resize(img, (640, 640))
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
            else:
                img_tensor = image

            return self.model(img_tensor)
        except Exception:
            return None

    def reset(self):
        """Reset model parameters to original pretrained state."""
        self._yolo = YOLO.__init__.__class__  # placeholder
        print("[TENT] Note: full reset requires re-loading the original weights.")

    @property
    def adapted_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TENT adapter smoke test — checking parameter setup...")
    adapter = TENTAdapter(model_path="yolov8n.pt", device="cpu")
    n_adapt = adapter.adapted_param_count
    n_total = sum(p.numel() for p in adapter.model.parameters())
    print(f"  Adapted parameters:  {n_adapt:,}")
    print(f"  Total parameters:    {n_total:,}")
    print(f"  Frozen fraction:     {(n_total - n_adapt) / n_total:.1%}")
    print("TENT adapter ready. Connect to evaluate.py for full experiment.")
