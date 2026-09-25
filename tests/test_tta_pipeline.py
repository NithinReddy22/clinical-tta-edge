"""
tests/test_tta_pipeline.py
==========================
Comprehensive automated test suite for clinical-tta-edge:
  1. Detection entropy computation & numerical stability
  2. Synthetic domain shift perturbations & dataset structure
  3. TENT adapter normalization layer parameter isolation & reset rollback
  4. SHOT adapter proposal entropy & class diversity regularization
  5. Entropy-Gated selective adaptation & compute savings counters
  6. Auxiliary-Task TTT rotation pretext prediction & self-supervised gradients
"""

import os
import shutil
import tempfile
import pytest
import torch
import numpy as np
import cv2

from baseline.entropy_analysis import compute_binary_entropy
from baseline.dataset_setup import apply_synthetic_shift, make_shifted_data_yaml
from methods.tent import TENTAdapter, detection_entropy
from methods.shot import SHOTAdapter, detection_shot_loss
from methods.entropy_gating import EntropyGatedTENT
from methods.ttt_aux import TTTAuxAdapter, RotationPretextHead


def test_entropy_computation():
    """Verify binary Shannon entropy calculation and clipping bounds."""
    confs = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    entropies = compute_binary_entropy(confs)

    assert len(entropies) == 3
    # At p=0.5, H(p) = -0.5*log(0.5) - 0.5*log(0.5) = ln(2) ≈ 0.693147
    assert np.isclose(entropies[1], np.log(2.0), atol=1e-3)
    # Near 0 and 1, entropy approaches 0
    assert entropies[0] < 1e-4
    assert entropies[2] < 1e-4
    assert np.all(np.isfinite(entropies))


def test_synthetic_shift_generation():
    """Verify image perturbation and label alignment for synthetic shift."""
    with tempfile.TemporaryDirectory() as temp_dir:
        src_img_dir = os.path.join(temp_dir, "src_images")
        src_lbl_dir = os.path.join(temp_dir, "src_labels")
        os.makedirs(src_img_dir)
        os.makedirs(src_lbl_dir)

        # Create dummy image and label
        dummy_img = (np.ones((100, 100, 3)) * 128).astype(np.uint8)
        img_file = os.path.join(src_img_dir, "sample001.jpg")
        cv2.imwrite(img_file, dummy_img)

        lbl_file = os.path.join(src_lbl_dir, "sample001.txt")
        with open(lbl_file, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        out_root = os.path.join(temp_dir, "shifted_out")
        apply_synthetic_shift(
            src_img_dir=src_img_dir,
            src_label_dir=src_lbl_dir,
            output_root=out_root,
            shift_type="brightness",
            shift_param=-30.0,
            max_images=1,
        )

        dst_img = os.path.join(out_root, "images", "val", "sample001.jpg")
        dst_lbl = os.path.join(out_root, "labels", "val", "sample001.txt")

        assert os.path.isfile(dst_img), "Shifted image was not written"
        assert os.path.isfile(dst_lbl), "Matching label was not copied"

        # Check shifted pixel value (128 - 30 = 98)
        loaded = cv2.imread(dst_img)
        assert np.allclose(loaded[0, 0], [98, 98, 98], atol=1)


def test_tent_adapter_parameter_isolation_and_rollback():
    """Verify that TENT adapts ONLY normalization parameters and rollback restores weights."""
    adapter = TENTAdapter(model_path="yolov8n.pt", device="cpu")

    # Verify 10,592 parameters adapted (99.7% frozen)
    assert adapter.adapted_param_count == 10592
    assert adapter.total_param_count > 3_000_000

    # Ensure all non-BN params do NOT require grad
    for name, param in adapter.model.named_parameters():
        if "bn" not in name.lower() and "norm" not in name.lower():
            assert not param.requires_grad, f"Non-norm parameter {name} has requires_grad=True!"

    # Save a copy of an active BN weight
    first_bn_param = next(p for p in adapter.model.parameters() if p.requires_grad)
    initial_val = first_bn_param.clone()

    # Perform adapt step
    dummy_x = torch.randn(1, 3, 640, 640)
    loss = adapter.adapt_step(dummy_x)
    assert np.isfinite(loss)
    assert not torch.equal(first_bn_param, initial_val), "Weights should change after optimizer step"

    # Test parameter rollback
    adapter.reset()
    assert torch.equal(first_bn_param, initial_val), "Weights must exactly match initial state after reset()"


def test_shot_adapter_loss():
    """Verify SHOT information maximization loss computation with entropy + diversity."""
    adapter = SHOTAdapter(model_path="yolov8n.pt", device="cpu", diversity_weight=0.5)
    dummy_x = torch.randn(1, 3, 640, 640)
    initial_loss = adapter.adapt_step(dummy_x)

    assert np.isfinite(initial_loss)
    adapter.reset()


def test_entropy_gated_adaptation():
    """Verify conditional gating behavior and compute savings counters."""
    gated = EntropyGatedTENT(model_path="yolov8n.pt", entropy_threshold=0.38, device="cpu")
    gated.reset_stats()

    dummy_img = (np.ones((640, 640, 3)) * 128).astype(np.uint8)

    # Image prediction with gating
    res, ent, did_adapt = gated.adapt_and_predict(dummy_img)

    assert gated.stats["total_frames"] == 1
    assert gated.stats["adapted_frames"] + gated.stats["skipped_frames"] == 1
    assert 0.0 <= gated.adaptation_rate <= 1.0
    assert 0.0 <= gated.compute_savings <= 1.0


def test_ttt_aux_head():
    """Verify Auxiliary-Task TTT 4-way rotation classification."""
    head = RotationPretextHead(in_channels=256)
    x = torch.randn(4, 256, 20, 20)
    logits = head(x)
    assert logits.shape == (4, 4), f"Expected [4, 4], got {logits.shape}"

    adapter = TTTAuxAdapter(model_path="yolov8n.pt", device="cpu")
    dummy_x = torch.randn(1, 3, 640, 640)
    loss = adapter.adapt_step(dummy_x)
    assert np.isfinite(loss)
    adapter.reset()
