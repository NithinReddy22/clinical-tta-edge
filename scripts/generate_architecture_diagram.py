"""
scripts/generate_architecture_diagram.py
========================================
Generates publication-quality pipeline architecture diagram for Clinical-TTA-Edge.
Produces high-resolution PNG and vector SVG formats for README and paper submission.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch


def generate_pipeline_diagram(output_png: str, output_svg: str):
    fig = plt.figure(figsize=(18, 10.5), dpi=300, facecolor="#0B0F19")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title and Subtitle Header
    ax.text(
        50, 96.5,
        "Clinical-TTA-Edge: Test-Time Adaptation Architecture & Pipeline",
        ha="center", va="center",
        fontsize=21, fontweight="bold", color="#FFFFFF", fontfamily="sans-serif"
    )
    ax.text(
        50, 93.2,
        "Real-Time Edge Domain Generalization for Clinical Vision Models via Gated Normalization Adaptation",
        ha="center", va="center",
        fontsize=12, color="#94A3B8", fontfamily="sans-serif", style="italic"
    )

    def draw_card(x, y, w, h, title, subtitle="", bg="#131B2E", border="#1E293B", title_color="#38BDF8"):
        bbox = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2
        )
        ax.add_patch(bbox)
        if title:
            ax.text(
                x + w / 2, y + h - 2.8, title,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=title_color, zorder=3, fontfamily="sans-serif"
            )
        if subtitle:
            ax.text(
                x + w / 2, y + h - 5.0, subtitle,
                ha="center", va="center", fontsize=8.5,
                color="#64748B", zorder=3, fontfamily="sans-serif"
            )

    def draw_badge(x, y, text, bg="#1E293B", text_color="#E2E8F0", fontsize=8.5, w=None):
        pad_x = 1.0 if w is None else w / 2
        ax.text(
            x, y, text,
            ha="center", va="center", fontsize=fontsize, color=text_color,
            fontweight="bold", fontfamily="sans-serif", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", facecolor=bg, edgecolor="none", alpha=0.95)
        )

    # -------------------------------------------------------------
    # STAGE 1: Clinical Domain Shift Stream (Left)
    # -------------------------------------------------------------
    draw_card(3, 30, 18, 55, "1. UNLABELED STREAM", "Clinical Video Feed", bg="#0F172A", border="#38BDF8", title_color="#38BDF8")

    # Shift types items inside card 1
    shifts = [
        ("Input Frame x_t", "Raw Video Stream (640x640)", "#38BDF8", "#0369A1"),
        ("Night Ward Shift", "Dimmed Lighting (delta L < 0)", "#F59E0B", "#78350F"),
        ("Low Contrast", "Foggy lens / High glare", "#F59E0B", "#78350F"),
        ("Defocus / Vibration", "Gaussian blur (sigma > 2.0)", "#EF4444", "#7F1D1D"),
        ("Edge Sensor Noise", "Gaussian CMOS noise", "#EF4444", "#7F1D1D"),
    ]
    cur_y = 74
    for name, desc, color, tag_bg in shifts:
        bbox = FancyBboxPatch((4.5, cur_y - 5.5), 15, 6.2, boxstyle="round,pad=0.3,rounding_size=0.8", facecolor="#1E293B", edgecolor=color, linewidth=1.2, zorder=3)
        ax.add_patch(bbox)
        ax.text(5.5, cur_y - 2.5, name, fontsize=9.2, fontweight="bold", color=color, zorder=4)
        ax.text(5.5, cur_y - 4.5, desc, fontsize=7.2, color="#94A3B8", zorder=4)
        cur_y -= 8.5

    ax.text(12, 33.5, "No Private Patient Labels Required", ha="center", fontsize=8, color="#10B981", fontweight="bold", zorder=4)

    # -------------------------------------------------------------
    # STAGE 2: Edge YOLOv8 Detector (Frozen vs Active Normalization)
    # -------------------------------------------------------------
    draw_card(25, 30, 21, 55, "2. EDGE DETECTOR", "YOLOv8n Single-Stage Architecture", bg="#0F172A", border="#6366F1", title_color="#818CF8")

    # Backbone & Neck Box
    bbox_frozen = FancyBboxPatch((26.5, 60), 18, 19, boxstyle="round,pad=0.4,rounding_size=0.8", facecolor="#181825", edgecolor="#4B5563", linewidth=1.2, zorder=3)
    ax.add_patch(bbox_frozen)
    ax.text(35.5, 76.5, "CSPDarknet Backbone & PANet", ha="center", fontsize=9.5, fontweight="bold", color="#E2E8F0", zorder=4)
    ax.text(35.5, 73.5, "Conv + C2f Blocks + SPPF", ha="center", fontsize=8.2, color="#94A3B8", zorder=4)
    draw_badge(35.5, 69.5, "FROZEN WEIGHTS: 99.7%", bg="#374151", text_color="#FCA5A5", fontsize=8)
    ax.text(35.5, 65.5, "2.99M parameters frozen\nPreserves core spatial representations", ha="center", fontsize=7.5, color="#6B7280", zorder=4)

    # Normalization trainable parameters box
    bbox_affine = FancyBboxPatch((26.5, 37), 18, 19.5, boxstyle="round,pad=0.4,rounding_size=0.8", facecolor="#1E1B4B", edgecolor="#818CF8", linewidth=1.6, zorder=3)
    ax.add_patch(bbox_affine)
    ax.text(35.5, 53.5, "BatchNorm2d / LayerNorm", ha="center", fontsize=10, fontweight="bold", color="#A5B4FC", zorder=4)
    ax.text(35.5, 50.5, "Affine Parameters ONLY", ha="center", fontsize=8.5, color="#C7D2FE", zorder=4)
    draw_badge(35.5, 46.5, "ADAPTIVE: gamma, beta in R^10,592", bg="#4338CA", text_color="#EEF2FF", fontsize=8.2)
    ax.text(35.5, 41.5, "Running Mean & Var Frozen\nPrevents B=1 statistic collapse", ha="center", fontsize=7.8, color="#93C5FD", zorder=4)

    ax.text(35.5, 33.0, "Checkpoint & Rollback Buffer", ha="center", fontsize=8, color="#F59E0B", fontweight="bold", zorder=4)

    # -------------------------------------------------------------
    # STAGE 3: Proposal Entropy & Gating Engine
    # -------------------------------------------------------------
    draw_card(50, 48, 22, 37, "3. ENTROPY GATING", "Conditional Edge Execution", bg="#0F172A", border="#10B981", title_color="#34D399")

    # Shannon Entropy formula box
    bbox_ent = FancyBboxPatch((51.5, 68), 19, 11.5, boxstyle="round,pad=0.4,rounding_size=0.8", facecolor="#064E3B", edgecolor="#059669", linewidth=1.2, zorder=3)
    ax.add_patch(bbox_ent)
    ax.text(61, 77, "Binary Shannon Detection Entropy", ha="center", fontsize=8.8, fontweight="bold", color="#A7F3D0", zorder=4)
    ax.text(61, 73, r"$H(p_t) = -\frac{1}{M}\sum_{i=1}^M [p_i \log p_i + (1-p_i)\log(1-p_i)]$", ha="center", fontsize=7.8, color="#ECFDF5", zorder=4)
    ax.text(61, 69.5, "Evaluated on multi-scale detection grid (8,400 anchors)", ha="center", fontsize=7.2, color="#6EE7B7", zorder=4)

    # Decision Diamond / Gating Box
    bbox_decision = FancyBboxPatch((51.5, 51.5), 19, 13, boxstyle="round,pad=0.4,rounding_size=0.8", facecolor="#1E293B", edgecolor="#10B981", linewidth=1.5, zorder=3)
    ax.add_patch(bbox_decision)
    ax.text(61, 62, "Edge Gating Decision Gate", ha="center", fontsize=9.2, fontweight="bold", color="#F8FAFC", zorder=4)
    ax.text(61, 58, "Condition: H(p_t) > tau_gate (0.38)", ha="center", fontsize=8.2, color="#FCD34D", zorder=4)

    draw_badge(56, 54, "NO (H <= tau)", bg="#065F46", text_color="#A7F3D0", fontsize=7.5)
    draw_badge(66, 54, "YES (H > tau)", bg="#991B1B", text_color="#FECACA", fontsize=7.5)

    # -------------------------------------------------------------
    # STAGE 4: TTA Optimization Objectives (Bottom Middle)
    # -------------------------------------------------------------
    draw_card(50, 4, 46, 38, "4. TTA OBJECTIVE FORMULATIONS & ADAPTATION", "Multi-Paradigm Edge Optimization", bg="#0F172A", border="#F59E0B", title_color="#FBBF24")

    # Three objective cards side by side
    methods = [
        ("TENT Adapter", r"L = L_entropy", "Standard Entropy Min.\nOptimizes candidate\nconfidence sharpness", "#38BDF8", 51.5),
        ("SHOT Adapter", r"L = L_ent - beta * L_div", "Information Maximization\nDiversity term prevents\nrepresentation collapse", "#A855F7", 66.5),
        ("TTT-Aux Adapter", r"L = L_rot (4-way CE)", "Self-Supervised Rotation\nPretext head adapts\nbackbone features", "#EC4899", 81.5),
    ]

    for title, formula, desc, color, bx in methods:
        bbox_m = FancyBboxPatch((bx, 20), 13.5, 17, boxstyle="round,pad=0.3,rounding_size=0.6", facecolor="#181825", edgecolor=color, linewidth=1.3, zorder=3)
        ax.add_patch(bbox_m)
        ax.text(bx + 6.75, 34.5, title, ha="center", fontsize=9.2, fontweight="bold", color=color, zorder=4)
        ax.text(bx + 6.75, 31, formula, ha="center", fontsize=7.8, fontweight="bold", color="#F1F5F9", zorder=4)
        ax.text(bx + 6.75, 25.5, desc, ha="center", fontsize=7.2, color="#94A3B8", zorder=4)

    # Optimization engine bar below
    bbox_opt = FancyBboxPatch((51.5, 6.5), 43, 10.5, boxstyle="round,pad=0.3,rounding_size=0.6", facecolor="#1E293B", edgecolor="#64748B", linewidth=1.2, zorder=3)
    ax.add_patch(bbox_opt)
    ax.text(73, 14.5, "Fast Backward Pass: grad_(gamma, beta) L (Adam, lr=1e-4, 1-step per frame)", ha="center", fontsize=9.2, fontweight="bold", color="#38BDF8", zorder=4)
    ax.text(73, 11.2, "Weight Update: gamma <- gamma - lr * d_gamma  |  beta <- beta - lr * d_beta", ha="center", fontsize=8.4, color="#E2E8F0", zorder=4)
    ax.text(73, 8.2, "Edge Compute Savings: 50% - 80% FLOPs saved on in-distribution clinical feeds", ha="center", fontsize=8.0, color="#10B981", fontweight="bold", zorder=4)

    # -------------------------------------------------------------
    # STAGE 5: Adapted Inference & Edge HUD Telemetry (Right)
    # -------------------------------------------------------------
    draw_card(76, 48, 20, 37, "5. EDGE TELEMETRY", "Adapted Inference Output", bg="#0F172A", border="#38BDF8", title_color="#38BDF8")

    bbox_hud = FancyBboxPatch((78, 51.5), 16, 28, boxstyle="round,pad=0.4,rounding_size=0.8", facecolor="#030712", edgecolor="#38BDF8", linewidth=1.4, zorder=3)
    ax.add_patch(bbox_hud)
    ax.text(86, 76.5, "[ LIVE EDGE HUD ]", ha="center", fontsize=9.5, fontweight="bold", color="#38BDF8", zorder=4)

    hud_lines = [
        ("Status:", "ADAPTING (Grad Step)", "#EF4444"),
        ("Method:", "Entropy-Gated TENT", "#38BDF8"),
        ("Shift:", "Low Contrast Ward", "#F59E0B"),
        ("Entropy H:", "0.462 > 0.38 (tau)", "#FCD34D"),
        ("Inference:", "38.2 ms (Edge CPU)", "#10B981"),
        ("FPS:", "26.2 fps", "#10B981"),
        ("Target mAP:", "+3.8% recovered", "#34D399"),
    ]
    hy = 72.5
    for label, val, color in hud_lines:
        ax.text(79.2, hy, label, fontsize=7.6, color="#94A3B8", zorder=4)
        ax.text(92.8, hy, val, ha="right", fontsize=7.6, fontweight="bold", color=color, zorder=4)
        hy -= 3.1

    # -------------------------------------------------------------
    # CONNECTING ARROWS & FLOW ANNOTATIONS
    # -------------------------------------------------------------
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", style="->", rad=0.0, lw=2.0, ls="-"):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5,head_width=3",
            connectionstyle=f"arc3,rad={rad}",
            color=color, linewidth=lw, linestyle=ls, zorder=10
        )
        ax.add_patch(arrow)

    # Frame stream -> YOLOv8
    draw_arrow(21, 57.5, 25, 57.5, color="#38BDF8", lw=2.2)
    ax.text(23, 59.5, "Frame x_t", fontsize=8, color="#94A3B8", ha="center", zorder=11)

    # YOLOv8 -> Entropy Calculation
    draw_arrow(46, 68, 50, 68, color="#818CF8", lw=2.2)
    ax.text(48, 70, "Proposals p", fontsize=8, color="#A5B4FC", ha="center", zorder=11)

    # Gating YES (H > tau) -> TTA Objectives
    draw_arrow(68.5, 51.5, 73, 42, color="#EF4444", lw=2.0, rad=-0.15)
    ax.text(72.5, 46.5, "Trigger Adaptation", fontsize=8, color="#FCA5A5", fontweight="bold", zorder=11)

    # Gating NO (H <= tau) -> Direct Telemetry Bypass
    draw_arrow(56, 51.5, 76, 57, color="#10B981", lw=2.0, rad=0.25, ls="--")
    ax.text(65, 57, "Skip Backprop (Gated)", fontsize=8, color="#34D399", fontweight="bold", ha="center", zorder=11)

    # TTA Loss Backward -> Normalization Layer update
    draw_arrow(51.5, 11, 35.5, 37, color="#F59E0B", lw=2.2, rad=0.25)
    ax.text(41, 21, "grad_(gamma, beta) L update", fontsize=8.5, color="#FCD34D", fontweight="bold", ha="center", zorder=11)

    # Adapted YOLO -> HUD
    draw_arrow(46, 45, 76, 64, color="#38BDF8", lw=2.0, rad=-0.1)

    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", dpi=300)
    plt.savefig(output_svg, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Generated: {output_png} and {output_svg}")


if __name__ == "__main__":
    png_path = "docs/figures/pipeline_architecture.png"
    svg_path = "docs/figures/pipeline_architecture.svg"
    generate_pipeline_diagram(png_path, svg_path)
