"""
==============================================================================
AGENT 3: Model Training & Evaluation Agent
==============================================================================
Role   : Machine Learning Engineer
Purpose: Train, evaluate, and compare models for Breast Cancer Detection.

INPUT  : dict from Agent 1 with keys:
    - X_train, X_test, y_train, y_test, feature_names

OUTPUT : dict with keys:
    - results          : dict of model_name -> {accuracy, precision, recall, f1, confusion_matrix}
    - best_model_name  : str
    - best_model       : trained model object
    - comparison_table : pd.DataFrame
    - feature_importance : dict (from Logistic Regression coefficients)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FIG_DPI = 180

# Dark theme colors (matching Agent 2)
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

MODEL_COLORS = {
    "Logistic Regression": ACCENT_BLUE,
    "SVM": ACCENT_RED,
    "KNN": ACCENT_GREEN,
}


def _apply_dark_theme():
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


def _add_watermark(fig, text="Breast Cancer Detection — Model Evaluation"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=8, color=TEXT_MUTED, alpha=0.4, style="italic")


def _train_and_evaluate(model, name, X_train, X_test, y_train, y_test):
    """Train a single model and return its metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # ROC curve data
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["roc_auc"] = round(auc(fpr, tpr), 4)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr

    print(f"\n   📌 {name}")
    print(f"      Accuracy : {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall   : {metrics['recall']:.4f}")
    print(f"      F1-Score : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"      ROC AUC  : {metrics['roc_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"      Confusion Matrix:")
    print(f"         [[TN={cm[0][0]:3d}  FP={cm[0][1]:3d}]")
    print(f"          [FN={cm[1][0]:3d}  TP={cm[1][1]:3d}]]")

    return model, metrics


def run(agent1_output: dict) -> dict:
    print("\n" + "=" * 70)
    print("AGENT 3: Model Training & Evaluation")
    print("=" * 70)

    _apply_dark_theme()

    X_train = agent1_output["X_train"]
    X_test = agent1_output["X_test"]
    y_train = agent1_output["y_train"]
    y_test = agent1_output["y_test"]
    feature_names = agent1_output["feature_names"]
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Define models
    # ------------------------------------------------------------------
    # WHY these three:
    #   - Logistic Regression: Fast, interpretable baseline with feature importance
    #   - SVM: Excellent for high-dimensional data, strong decision boundaries
    #   - KNN: Non-parametric, captures local patterns
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    # ------------------------------------------------------------------
    # Train and evaluate each model
    # ------------------------------------------------------------------
    print("\n🔧 Training and evaluating models...")
    results = {}
    trained_models = {}
    for name, model in models.items():
        trained_model, metrics = _train_and_evaluate(
            model, name, X_train, X_test, y_train, y_test
        )
        results[name] = metrics
        trained_models[name] = trained_model

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    comparison_data = []
    for name, m in results.items():
        row = {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1"],
        }
        if "roc_auc" in m:
            row["ROC AUC"] = m["roc_auc"]
        comparison_data.append(row)
    comparison_df = pd.DataFrame(comparison_data)

    print("\n📊 Model Comparison Table:")
    print(comparison_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Select best model — prioritizing RECALL
    # ------------------------------------------------------------------
    # WHY Recall is critical in cancer detection:
    #   - A FALSE NEGATIVE (missing a malignant tumor) means a patient
    #     with cancer is told they're healthy -> delayed treatment -> life-threatening.
    #   - A FALSE POSITIVE (flagging benign as malignant) leads to
    #     additional tests -> inconvenient but not dangerous.
    #   - Therefore, we want to MAXIMIZE RECALL (minimize false negatives).
    #   - Among models with similar recall, we prefer higher F1 as tiebreaker.
    best_name = comparison_df.sort_values(
        by=["Recall", "F1-Score"], ascending=False
    ).iloc[0]["Model"]
    best_model = trained_models[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   Selected because it has the highest Recall ({results[best_name]['recall']:.4f})")
    print(f"   WHY Recall matters: In cancer detection, missing a malignant case")
    print(f"   (false negative) is far more dangerous than a false alarm (false positive).")
    print(f"   High recall = fewer missed cancers = lives saved.")

    # ===================================================================
    # PLOT 6: Radar chart — model comparison
    # ===================================================================
    metrics_radar = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    for name in models:
        vals = [results[name]["accuracy"], results[name]["precision"],
                results[name]["recall"], results[name]["f1"]]
        vals += vals[:1]
        color = MODEL_COLORS[name]
        ax.plot(angles, vals, "o-", linewidth=2.5, label=name, color=color, markersize=7)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=12, fontweight="bold")
    ax.set_ylim(0.90, 1.01)
    ax.set_rticks([0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_yticklabels(["0.92", "0.94", "0.96", "0.98", "1.00"],
                       fontsize=8, color=TEXT_MUTED)
    ax.set_title("Model Performance — Radar Comparison",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=30)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=11)
    ax.grid(color=GRID_COLOR, alpha=0.4)
    ax.spines["polar"].set_color(GRID_COLOR)
    _add_watermark(fig)
    plt.tight_layout()
    path_radar = os.path.join(PLOT_DIR, "06_model_radar.png")
    fig.savefig(path_radar, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"   📁 Saved: {path_radar}")

    # ===================================================================
    # PLOT 7: Grouped bar chart — model comparison
    # ===================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics_to_plot))
    width = 0.22
    model_names = list(models.keys())

    for i, name in enumerate(model_names):
        vals = [results[name]["accuracy"], results[name]["precision"],
                results[name]["recall"], results[name]["f1"]]
        color = MODEL_COLORS[name]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color,
                      edgecolor=DARK_BG, linewidth=1.2, zorder=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, fontweight="bold",
                     padding=3, color=TEXT_COLOR)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_plot, fontsize=13, fontweight="bold")
    ax.set_ylim(0.90, 1.025)
    ax.set_title("Model Performance Comparison",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_watermark(fig)
    plt.tight_layout()
    path_comp = os.path.join(PLOT_DIR, "07_model_comparison.png")
    fig.savefig(path_comp, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"   📁 Saved: {path_comp}")

    # ===================================================================
    # PLOT 8: Confusion matrices — styled
    # ===================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    labels = ["Malignant", "Benign"]

    for i, name in enumerate(model_names):
        cm = results[name]["confusion_matrix"]
        color = MODEL_COLORS[name]
        # Create custom colormap from dark bg to model's accent color
        from matplotlib.colors import LinearSegmentedColormap
        cm_cmap = LinearSegmentedColormap.from_list(
            f"cm_{name}", [CARD_BG, color], N=100
        )
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cm_cmap, ax=axes[i],
            xticklabels=labels, yticklabels=labels, cbar=False,
            annot_kws={"size": 18, "fontweight": "bold"},
            linewidths=2, linecolor=DARK_BG,
            square=True
        )
        # Highlight best model
        title = name
        if name == best_name:
            title = f"🏆  {name}"
        axes[i].set_title(title, fontsize=13, fontweight="bold",
                         color=color, pad=12)
        axes[i].set_xlabel("Predicted", fontsize=11, labelpad=8)
        axes[i].set_ylabel("Actual", fontsize=11, labelpad=8)
        axes[i].tick_params(labelsize=11)

    fig.suptitle("Confusion Matrices — All Models",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=1.02)
    _add_watermark(fig)
    plt.tight_layout()
    path_cm = os.path.join(PLOT_DIR, "08_confusion_matrices.png")
    fig.savefig(path_cm, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"   📁 Saved: {path_cm}")

    # ===================================================================
    # PLOT 9: ROC Curves
    # ===================================================================
    fig, ax = plt.subplots(figsize=(9, 8))
    for name in model_names:
        if "fpr" in results[name]:
            color = MODEL_COLORS[name]
            roc_auc_val = results[name]["roc_auc"]
            ax.plot(results[name]["fpr"], results[name]["tpr"],
                    color=color, linewidth=2.5,
                    label=f"{name} (AUC = {roc_auc_val:.4f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "--", color=TEXT_MUTED, linewidth=1, alpha=0.5,
            label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — All Models",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_watermark(fig)
    plt.tight_layout()
    path_roc = os.path.join(PLOT_DIR, "09_roc_curves.png")
    fig.savefig(path_roc, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"   📁 Saved: {path_roc}")

    # ===================================================================
    # Feature importance (Logistic Regression coefficients)
    # ===================================================================
    lr_model = trained_models["Logistic Regression"]
    coef = np.abs(lr_model.coef_[0])
    feat_imp = dict(zip(feature_names, coef))
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    print(f"\n📊 Feature Importance (Logistic Regression |coefficients|):")
    for i, (feat, imp) in enumerate(list(feat_imp_sorted.items())[:10], 1):
        print(f"   {i:2d}. {feat:30s}  |coef| = {imp:.4f}")

    # ===================================================================
    # PLOT 10: Feature importance — gradient bar chart
    # ===================================================================
    top_feats = list(feat_imp_sorted.keys())[:15]
    top_vals = np.array([feat_imp_sorted[f] for f in top_feats])
    fig, ax = plt.subplots(figsize=(11, 8))

    # Color gradient based on importance
    norm_vals = top_vals / top_vals.max()
    colors_bar = []
    for v in norm_vals:
        r = int(68 + v * (24 - 68))
        g = int(138 + v * (255 - 138))
        b = int(255 + v * (255 - 255))
        colors_bar.append(f"#{r:02x}{g:02x}{b:02x}")

    bars = ax.barh(range(len(top_feats)), top_vals[::-1],
                   color=colors_bar[::-1], edgecolor=DARK_BG, linewidth=0.8,
                   height=0.7, zorder=3)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], fontsize=10)
    ax.set_xlabel("|Coefficient Magnitude|", fontsize=12)
    ax.set_title("Feature Importance — Logistic Regression",
                 fontsize=16, fontweight="bold", color=ACCENT_CYAN, pad=15)
    ax.grid(axis="x", alpha=0.15, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_COLOR,
                fontweight="bold")
    _add_watermark(fig)
    plt.tight_layout()
    path_fi = os.path.join(PLOT_DIR, "10_feature_importance.png")
    fig.savefig(path_fi, dpi=FIG_DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"   📁 Saved: {path_fi}")

    # ------------------------------------------------------------------
    # Package output
    # ------------------------------------------------------------------
    output = {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "comparison_table": comparison_df,
        "feature_importance": feat_imp_sorted,
    }
    print(f"\n✅ Agent 3 complete. Best model: {best_name}")
    print("=" * 70)
    return output


if __name__ == "__main__":
    from agent1_data_preprocessing import run as run_agent1
    result = run(run_agent1())
