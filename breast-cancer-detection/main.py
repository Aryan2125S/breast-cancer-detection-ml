"""
==============================================================================
Breast Cancer Detection — Pipeline Orchestrator
==============================================================================
Runs all 4 agents in sequence, passing outputs from one to the next.

Usage:
    python main.py
==============================================================================
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from agents.agent1_data_preprocessing import run as run_agent1
from agents.agent2_eda_analysis import run as run_agent2
from agents.agent3_model_training import run as run_agent3
from agents.agent4_deployment import run as run_agent4


def main():
    print("\n" + "🔬" * 35)
    print("   BREAST CANCER DETECTION — Multi-Agent ML Pipeline")
    print("🔬" * 35)

    # ------------------------------------------------------------------
    # Agent 1: Data & Preprocessing
    # ------------------------------------------------------------------
    agent1_output = run_agent1()
    # Output: X_train, X_test, y_train, y_test, feature_names, scaler, raw_df, label_map

    # ------------------------------------------------------------------
    # Agent 2: EDA & Feature Analysis
    # ------------------------------------------------------------------
    agent2_output = run_agent2(agent1_output)
    # Output: class_distribution, top_correlated_features, insights, plot_paths

    # ------------------------------------------------------------------
    # Agent 3: Model Training & Evaluation
    # ------------------------------------------------------------------
    agent3_output = run_agent3(agent1_output)
    # Output: results, best_model_name, best_model, comparison_table, feature_importance

    # ------------------------------------------------------------------
    # Agent 4: Deployment & Final Packaging
    # ------------------------------------------------------------------
    agent4_output = run_agent4(agent1_output, agent3_output)
    # Output: model_path, scaler_path, metadata_path, prediction_fn

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "🎉" * 35)
    print("   PIPELINE COMPLETE — Summary")
    print("🎉" * 35)
    print(f"\n   📊 Dataset    : {agent1_output['raw_df'].shape[0]} samples, "
          f"{len(agent1_output['feature_names'])} features")
    print(f"   📈 EDA        : {len(agent2_output['plot_paths'])} visualizations, "
          f"{len(agent2_output['insights'])} insights")
    print(f"   🏆 Best Model : {agent3_output['best_model_name']}")
    print(f"      Accuracy   : {agent3_output['results'][agent3_output['best_model_name']]['accuracy']}")
    print(f"      Recall     : {agent3_output['results'][agent3_output['best_model_name']]['recall']}")
    print(f"   💾 Model saved: {agent4_output['model_path']}")
    print(f"\n   🚀 Run the Streamlit app:")
    print(f"      streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
