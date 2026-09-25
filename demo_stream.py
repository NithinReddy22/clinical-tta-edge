"""
demo_stream.py
==============
Real-time Video and Webcam Stream Demonstration for Clinical-TTA-Edge.

Test test-time adaptation live on:
  - Live Webcam feed: python demo_stream.py --source 0
  - Video file:       python demo_stream.py --source hospital_clip.mp4 --save output.mp4

Interactive Live Controls:
  - Press 'm' to cycle modes: [Baseline (No TTA)] <-> [TENT (Continuous)] <-> [Entropy-Gated TTA]
  - Press 's' to inject live clinical shifts: [None] <-> [Dim Ward / Night Mode] <-> [Low Contrast] <-> [Defocus Blur]
  - Press 'r' to reset model weights back to clean pretrained state
  - Press 'q' to quit

Displays live on-screen HUD:
  - Real-time detection bounding boxes and confidence
  - Frame detection entropy gauge H(p) vs gating threshold tau
  - Adaptation state: [ADAPTING (Grad Step)] vs [GATED (Skipped, In-Distribution)]
  - Real-time FPS and edge processing latency (ms)
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure repo root is in sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from methods.tent import TENTAdapter
from methods.entropy_gating import EntropyGatedTENT


def apply_live_synthetic_shift(frame: np.ndarray, shift_mode: str) -> np.ndarray:
    """Injects simulated real-time clinical domain shifts into the video stream."""
    if shift_mode == "night_mode":
        # Simulates low-illumination hospital room / dimmed night ward
        return np.clip(frame.astype(np.int32) - 65, 0, 255).astype(np.uint8)
    elif shift_mode == "low_contrast":
        # Simulates milky lens / foggy camera exposure
        return np.clip(frame.astype(np.float32) * 0.38 + 20, 0, 255).astype(np.uint8)
    elif shift_mode == "defocus_blur":
        # Simulates camera vibration or dirty lens defocus
        return cv2.GaussianBlur(frame, (11, 11), 0)
    return frame


def draw_hud(
    frame: np.ndarray,
    mode_name: str,
    shift_name: str,
    entropy: float,
    threshold: float,
    did_adapt: bool,
    fps: float,
    latency_ms: float,
    n_detections: int,
):
    """Renders a sleek, publication-grade edge telemetry HUD onto the video frame."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay box on top-left
    overlay = frame.copy()
    box_w = min(420, w - 20)
    box_h = 160
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (60, 70, 85), 1)

    # Status color
    if mode_name == "Baseline (No TTA)":
        mode_color = (180, 180, 180)
        status_text = "STATIC WEIGHTS (NO TTA)"
        status_color = (150, 150, 150)
    elif did_adapt:
        mode_color = (0, 215, 255)  # Amber / Gold
        status_text = "ADAPTING [BN AFFINE GRAD STEP]"
        status_color = (0, 165, 255)  # Orange
    else:
        mode_color = (80, 220, 100)  # Green
        status_text = "GATED [IN-DISTRIBUTION / PASS-THROUGH]"
        status_color = (80, 220, 100)

    # Text headers
    cv2.putText(frame, "CLINICAL-TTA-EDGE TELEMETRY", (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode_name}", (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mode_color, 1)
    cv2.putText(frame, f"Shift: {shift_name.upper()}", (22, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(frame, f"Status: {status_text}", (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

    # Entropy gauge bar
    bar_x = 22
    bar_y = 104
    bar_w = box_w - 44
    bar_h = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)

    # Fill gauge (normalized 0 to 0.7)
    norm_ent = np.clip(entropy / 0.65, 0.0, 1.0)
    fill_w = int(norm_ent * bar_w)
    fill_color = (80, 220, 100) if entropy <= threshold else (0, 140, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    # Threshold marker
    thresh_x = int(bar_x + np.clip(threshold / 0.65, 0.0, 1.0) * bar_w)
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 255), 2)

    cv2.putText(frame, f"H(p)={entropy:.3f} (tau={threshold:.2f})", (bar_x, bar_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Footer stats
    cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Objects: {n_detections}", (bar_x, bar_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Quick Help banner on bottom
    cv2.putText(frame, "[M] Mode  [S] Shift  [R] Reset  [Q] Quit", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def run_stream(
    source: str = "0",
    model_path: str = "yolov8n.pt",
    device: str = "cpu",
    entropy_threshold: float = 0.38,
    conf_thresh: float = 0.25,
    save_path: str = None,
    headless: bool = False,
):
    """Main live video / webcam streaming loop with test-time adaptation."""
    # Convert numeric webcam string to integer index
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        if source == "0":
            print("Tip: If you do not have a connected webcam, pass a video file: --source path/to/video.mp4")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0

    print("=" * 60)
    print(" Clinical-TTA-Edge: Live Video Adaptation Stream")
    print("=" * 60)
    print(f"Source:     {source} ({width}x{height} @ {fps_in:.1f} FPS)")
    print(f"Model:      {model_path} on {device}")
    print(f"Threshold:  tau={entropy_threshold}")
    print("Keyboard Controls:")
    print("  'm' -> Toggle Mode (Baseline -> TENT -> Gated)")
    print("  's' -> Inject Synthetic Shift (None -> Night -> Contrast -> Blur)")
    print("  'r' -> Reset Model to clean pretrained state")
    print("  'q' -> Exit")
    print("=" * 60)

    # Initialize adapters
    gated_adapter = EntropyGatedTENT(
        model_path=model_path,
        entropy_threshold=entropy_threshold,
        device=device,
    )

    # Modes and shifts lists
    modes = ["Entropy-Gated TTA", "TENT (Continuous)", "Baseline (No TTA)"]
    mode_idx = 0

    shifts = ["none", "night_mode", "low_contrast", "defocus_blur"]
    shift_idx = 0

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, min(fps_in, 30.0), (width, height))
        print(f"Recording output to: {save_path}")

    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video stream reached.")
                break

            t0 = time.perf_counter()

            # Apply user-toggled synthetic shift
            active_shift = shifts[shift_idx]
            shifted_frame = apply_live_synthetic_shift(frame, active_shift)

            active_mode = modes[mode_idx]
            did_adapt = False
            curr_entropy = 0.0

            # Preprocess tensor for entropy check / adaptation
            img_tensor = gated_adapter.preprocess_image(shifted_frame)

            if active_mode == "Baseline (No TTA)":
                # Pure inference without adaptation
                curr_entropy = gated_adapter.evaluate_entropy_only(img_tensor)
                results = gated_adapter._yolo(shifted_frame, verbose=False, conf=conf_thresh)
            elif active_mode == "TENT (Continuous)":
                # Continuous adaptation on every frame
                curr_entropy = gated_adapter.evaluate_entropy_only(img_tensor)
                gated_adapter.adapt_step(img_tensor)
                did_adapt = True
                results = gated_adapter._yolo(shifted_frame, verbose=False, conf=conf_thresh)
            else:  # Entropy-Gated TTA
                results, curr_entropy, did_adapt = gated_adapter.adapt_and_predict(shifted_frame, conf=conf_thresh)

            # Draw detections
            annotated_frame = results[0].plot() if results and len(results) else shifted_frame.copy()
            n_dets = len(results[0].boxes) if results and results[0].boxes is not None else 0

            # Latency and FPS
            t_elapsed = time.perf_counter() - t0
            latency_ms = t_elapsed * 1000.0
            fps_real = 1.0 / (time.perf_counter() - prev_time)
            prev_time = time.perf_counter()

            # Draw Telemetry HUD
            draw_hud(
                frame=annotated_frame,
                mode_name=active_mode,
                shift_name=active_shift,
                entropy=curr_entropy,
                threshold=entropy_threshold,
                did_adapt=did_adapt,
                fps=fps_real,
                latency_ms=latency_ms,
                n_detections=n_dets,
            )

            if writer:
                writer.write(annotated_frame)

            # Display window (if not in headless mode)
            if not headless:
                try:
                    cv2.imshow("Clinical-TTA-Edge | Real-Time Adaptation Stream", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("Exit requested by user.")
                        break
                    elif key == ord("m"):
                        mode_idx = (mode_idx + 1) % len(modes)
                        print(f"[MODE CHANGED] -> {modes[mode_idx]}")
                    elif key == ord("s"):
                        shift_idx = (shift_idx + 1) % len(shifts)
                        print(f"[SHIFT CHANGED] -> {shifts[shift_idx]}")
                    elif key == ord("r"):
                        gated_adapter.reset()
                        print("[RESET] Model restored to clean pretrained weights.")
                except Exception:
                    pass

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Stream closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Live Video & Webcam Adaptation Demo")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. '0') or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda:0'")
    parser.add_argument("--threshold", type=float, default=0.38, help="Calibrated entropy gating threshold")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--save", type=str, default=None, help="Optional output MP4 path to save recorded stream")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window")
    args = parser.parse_args()

    run_stream(
        source=args.source,
        model_path=args.model,
        device=args.device,
        entropy_threshold=args.threshold,
        conf_thresh=args.conf,
        save_path=args.save,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
