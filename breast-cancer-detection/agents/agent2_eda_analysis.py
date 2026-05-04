"""
==============================================================================
AGENT 2: EDA & Feature Analysis Agent
==============================================================================
Role   : Data Analyst
Purpose: Perform EDA on the breast cancer dataset.

INPUT  : dict from Agent 1 (raw_df, feature_names, label_map)
OUTPUT : dict with keys:
    - class_distribution, top_correlated_features, insights, plot_paths
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TOP_N_FEATURES = 10
FIG_DPI = 180

# ---------------------------------------------------------------------------
# Premium dark theme configuration
# ---------------------------------------------------------------------------
DARK_BG = "#0f0f1a"
CARD_BG = "#1a1a2e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
TEXT_MUTED = "#8888aa"
ACCENT_GREEN = "#00e676"
ACCENT_RED = "#ff5252"
ACCENT_BLUE = "#448aff"
ACCENT_PURPLE = "#7c4dff"
ACCENT_CYAN = "#18ffff"
ACCENT_AMBER = "#ffd740"

PALETTE_LIST = [ACCENT_GREEN, ACCENT_RED]
PALETTE_DICT = {"Benign": ACCENT_GREEN, "Malignant": ACCENT_RED}

# Gradient palette for heatmaps
GRADIENT_COLORS = ["#0f0f1a", "#1a237e", "#283593", "#448aff", "#82b1ff",
                   "#ffffff", "#ff8a80", "#ff5252", "#d50000", "#b71c1c"]


def _apply_dark_theme():
    """Apply a consistent dark theme to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _add_watermark(fig, text="Breast Cancer Detection — EDA"):
    """Add a subtle watermark to plots."""
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=8, color=TEXT_MUTED, alpha=0.4, style="italic")


def run(agent1_output: dict) -> dict:
    print("\n" + "=" * 70)
    print("AGENT 2: EDA & Feature Analysis")
    print("=" * 70)

    _apply_dark_theme()

    df = agent1_output["raw_df"].copy()
    feature_names = agent1_output["feature_names"]
    os.makedirs(PLOT_DIR, exist_ok=True)
    insights = []
    plot_paths = []

    # --- Class Distribution ---
    class_dist = df["diagnosis"].value_counts().to_dict()
    total = len(df)
    print(f"\n📊 Class Distribution:")
    for label, count in class_dist.items():
        print(f"   {label}: {count} ({count/total*100:.1f}%)")

    ratio = class_dist.get("Benign", 0) / max(class_dist.get("Malignant", 1), 1)
    insights.append(
        f"Dataset is moderately imbalanced ({ratio:.1f}:1 Benign:Malignant). "
        f"Track Recall to ensure malignant cases are not missed."
    )

    # ===================================================================
    # PLOT 1: Class Distribution — Donut chart + bar chart combo
    # ===================================================================
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.3)

    # Left: Donut chart
    ax1 = fig.add_subplot(gs[0])
    benign_count = class_dist.get("Benign", 0)
    malignant_count = class_dist.get("Malignant", 0)
    sizes = [benign_count, malignant_count]
    colors = [ACCENT_GREEN, ACCENT_RED]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=["Benign", "Malignant"],
        colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2.5},
        pctdistance=0.78
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")

    # Draw center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc=DARK_BG, ec=GRID_COLOR, linewidth=1.5)
    ax1.add_artist(centre_circle)
    ax1.text(0, 0.05, str(total), ha="center", va="center",
             fontsize=28, fontweight="bold", color=TEXT_COLOR)
    ax1.text(0, -0.15, "Total\nSamples", ha="center", va="center",
             fontsize=10, color=TEXT_MUTED)

    # Right: Horizontal bar chart with value labels
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(
        ["Malignant", "Benign"],
        [malignant_count, benign_count],
        color=[ACCENT_RED, ACCENT_GREEN],
        edgecolor=DARK_BG, linewidth=1.5, height=0.55,
        zorder=3
    )
    ax2.set_xlim(0, max(sizes) * 1.25)
    ax2.grid(axis="x", alpha=0.2, color=GRID_COLOR, zorder=0)

    for bar, count in zip(bars, [malignant_count, benign_count]):
        pct = count / total * 100
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"  {count}  ({pct:.1f}%)",
                 va="center", fontsize=13, fontweight="bold", color=TEXT_COLOR)

    ax2.set_xlabel("Number of Samples", fontsize=12, labelpad=10)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Class Distribution — Benign vs Malignant",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=0.98)
    _add_watermark(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    path1 = os.path.join(PLOT_DIR, "01_class_distribution.png")
    fig.savefig(path1, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    plot_paths.append(path1)
    print(f"   📁 Saved: {path1}")

    # ===================================================================
    # Compute correlations (shared across multiple plots)
    # ===================================================================
    corr_target = df[feature_names].corrwith(df["target"]).abs().sort_values(ascending=False)
    top_features = corr_target.head(TOP_N_FEATURES).index.tolist()
    print(f"\n📊 Top {TOP_N_FEATURES} Features by |Correlation| with Target:")
    for i, feat in enumerate(top_features, 1):
        print(f"   {i:2d}. {feat:30s}  |r| = {corr_target[feat]:.4f}")
    insights.append(
        "Top features are 'worst' and 'mean' measurements of radius, perimeter, "
        "area, and concave points — tumor shape irregularity indicates malignancy."
    )

    # ===================================================================
    # PLOT 2: Correlation Heatmap — triangular with custom colormap
    # ===================================================================
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging",
        ["#d50000", "#ff5252", "#ff8a80", "#1a1a2e", "#82b1ff", "#448aff", "#1a237e"])

    top_15 = corr_target.head(15).index.tolist()
    corr_matrix = df[top_15 + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=custom_cmap,
        center=0, square=True, linewidths=0.8, linecolor=DARK_BG, ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
        annot_kws={"size": 8.5, "fontweight": "bold"},
        vmin=-1, vmax=1
    )
    ax.set_title("Correlation Heatmap — Top 15 Features + Target",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=20)
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    _add_watermark(fig)
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "02_correlation_heatmap.png")
    fig.savefig(path2, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    plot_paths.append(path2)
    print(f"   📁 Saved: {path2}")

    # ===================================================================
    # PLOT 3: Violin plots of top 6 features (richer than boxplots)
    # ===================================================================
    top_6 = top_features[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, feat in enumerate(top_6):
        ax = axes.ravel()[i]
        parts = ax.violinplot(
            [df[df["diagnosis"] == "Benign"][feat].values,
             df[df["diagnosis"] == "Malignant"][feat].values],
            positions=[0, 1], showmeans=True, showmedians=True, widths=0.7
        )
        # Color the violins
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor([ACCENT_GREEN, ACCENT_RED][j])
            body.set_alpha(0.6)
            body.set_edgecolor("white")
            body.set_linewidth(0.5)
        for partname in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if partname in parts:
                parts[partname].set_edgecolor(TEXT_MUTED)
                parts[partname].set_linewidth(1)

        # Overlay strip plot (jittered points)
        for j, label in enumerate(["Benign", "Malignant"]):
            subset = df[df["diagnosis"] == label][feat].values
            jitter = np.random.normal(0, 0.04, len(subset))
            ax.scatter(np.full_like(subset, j) + jitter, subset,
                       s=4, alpha=0.3, color=[ACCENT_GREEN, ACCENT_RED][j],
                       zorder=5, rasterized=True)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Malignant"], fontsize=11, fontweight="bold")
        ax.set_title(feat.replace("_", " ").title(), fontsize=12, fontweight="bold",
                     color=ACCENT_CYAN)
        ax.grid(axis="y", alpha=0.15)

    fig.suptitle("Feature Distributions — Violin Plots (Top 6)",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=1.01)
    _add_watermark(fig)
    plt.tight_layout()
    path3 = os.path.join(PLOT_DIR, "03_feature_violins.png")
    fig.savefig(path3, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    plot_paths.append(path3)
    print(f"   📁 Saved: {path3}")
    insights.append(
        f"Violin plots show clear separation for '{top_6[0]}' and '{top_6[1]}'. "
        "Malignant tumors have larger size and more shape irregularity."
    )

    # ===================================================================
    # PLOT 4: KDE overlays (density plots) — top 6 features
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, feat in enumerate(top_6):
        ax = axes.ravel()[i]
        for label, color in PALETTE_DICT.items():
            subset = df[df["diagnosis"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.25, color=color,
                    edgecolor="none", density=True, label=f"{label} (hist)")
            subset.plot.kde(ax=ax, color=color, linewidth=2.5, label=f"{label} (KDE)")

        ax.set_title(feat.replace("_", " ").title(), fontsize=12,
                     fontweight="bold", color=ACCENT_CYAN)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(axis="y", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Feature Density Plots — Histograms + KDE (Top 6)",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=1.01)
    _add_watermark(fig)
    plt.tight_layout()
    path4 = os.path.join(PLOT_DIR, "04_feature_density.png")
    fig.savefig(path4, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    plot_paths.append(path4)
    print(f"   📁 Saved: {path4}")
    insights.append(
        "KDE plots confirm minimal overlap for 'worst area' and 'worst radius' — "
        "excellent candidates for classification."
    )

    # ===================================================================
    # PLOT 5: Feature Correlation Bar Chart (horizontal, top 15)
    # ===================================================================
    top_15_corr = corr_target.head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_bar = [ACCENT_CYAN if v > 0.7 else ACCENT_BLUE if v > 0.5
                  else ACCENT_PURPLE for v in top_15_corr.values]
    bars = ax.barh(
        range(len(top_15_corr)), top_15_corr.values[::-1],
        color=colors_bar[::-1], edgecolor=DARK_BG, linewidth=0.5, height=0.7
    )
    ax.set_yticks(range(len(top_15_corr)))
    ax.set_yticklabels(top_15_corr.index[::-1], fontsize=10)
    ax.set_xlabel("|Pearson Correlation with Target|", fontsize=12)
    ax.set_title("Top 15 Features — Correlation with Diagnosis",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=15)
    ax.grid(axis="x", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, top_15_corr.values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_COLOR, fontweight="bold")

    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_CYAN, label="|r| > 0.7 (Strong)"),
        Patch(facecolor=ACCENT_BLUE, label="|r| > 0.5 (Moderate)"),
        Patch(facecolor=ACCENT_PURPLE, label="|r| ≤ 0.5 (Weak)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    _add_watermark(fig)
    plt.tight_layout()
    path5 = os.path.join(PLOT_DIR, "05_correlation_ranking.png")
    fig.savefig(path5, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    plot_paths.append(path5)
    print(f"   📁 Saved: {path5}")

    # --- Multicollinearity check ---
    corr_full = df[feature_names].corr().abs()
    upper = corr_full.where(np.triu(np.ones(corr_full.shape), k=1).astype(bool))
    high_pairs = [(idx, col, upper.loc[idx, col])
                  for col in upper.columns for idx in upper.index
                  if not np.isnan(upper.loc[idx, col]) and upper.loc[idx, col] > 0.9]
    high_pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n📊 Multicollinearity: {len(high_pairs)} pairs with |r| > 0.9")
    for f1, f2, v in high_pairs[:5]:
        print(f"   {f1:30s} ↔ {f2:30s}  |r| = {v:.4f}")
    insights.append(
        f"{len(high_pairs)} feature pairs with |r| > 0.9 (e.g. radius <-> perimeter <-> area). "
        "This is expected — they all measure tumor size."
    )

    output = {
        "class_distribution": class_dist,
        "top_correlated_features": top_features,
        "insights": insights,
        "plot_paths": plot_paths,
    }
    print(f"\n✅ Agent 2 complete. {len(plot_paths)} visualizations, {len(insights)} insights.")
    print("=" * 70)
    return output


if __name__ == "__main__":
    from agent1_data_preprocessing import run as run_agent1
    result = run(run_agent1())
