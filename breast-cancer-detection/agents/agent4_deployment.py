"""
==============================================================================
AGENT 4: Deployment & Final Packaging Agent
==============================================================================
Role   : ML Deployment Engineer
Purpose: Save the best model and scaler, and provide a prediction function.

INPUT  : dict from Agent 1 (scaler, feature_names, label_map, raw_df)
         dict from Agent 3 (best_model, best_model_name)

OUTPUT : dict with keys:
    - model_path   : str — Path to saved model file
    - scaler_path  : str — Path to saved scaler file
    - prediction_fn: callable — Function that accepts raw features → prediction
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")


def predict(features: list, scaler, model, feature_names: list, label_map: dict) -> dict:
    """
    Make a prediction on raw (unscaled) feature values.

    Parameters
    ----------
    features : list of float
        Raw feature values in the same order as feature_names (30 values).
    scaler : StandardScaler
        Fitted scaler from Agent 1.
    model : trained sklearn model
        Best model from Agent 3.
    feature_names : list[str]
        Feature column names (for validation).
    label_map : dict
        {0: "Malignant", 1: "Benign"}.

    Returns
    -------
    dict with keys: prediction (str), prediction_code (int), features_used (int)
    """
    if len(features) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, got {len(features)}. "
            f"Features must be in order: {feature_names[:3]}..."
        )

    # Reshape to 2D array (1 sample, 30 features) and scale
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    pred_code = model.predict(X_scaled)[0]
    pred_label = label_map.get(pred_code, "Unknown")

    return {
        "prediction": pred_label,
        "prediction_code": int(pred_code),
        "features_used": len(features),
    }


def run(agent1_output: dict, agent3_output: dict) -> dict:
    print("\n" + "=" * 70)
    print("AGENT 4: Deployment & Final Packaging")
    print("=" * 70)

    scaler = agent1_output["scaler"]
    feature_names = agent1_output["feature_names"]
    label_map = agent1_output["label_map"]
    raw_df = agent1_output["raw_df"]
    best_model = agent3_output["best_model"]
    best_model_name = agent3_output["best_model_name"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Save model and scaler
    # ------------------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Saved model  : {model_path}")
    print(f"   Model type   : {best_model_name}")
    print(f"💾 Saved scaler : {scaler_path}")

    # ------------------------------------------------------------------
    # Test prediction with a sample from the dataset
    # ------------------------------------------------------------------
    print(f"\n🧪 Testing prediction system...")

    # Use first malignant and first benign sample as test cases
    for label_code, label_name in label_map.items():
        sample_row = raw_df[raw_df["target"] == label_code].iloc[0]
        sample_features = sample_row[feature_names].values.tolist()

        result = predict(sample_features, scaler, best_model, feature_names, label_map)
        match = "✅" if result["prediction"] == label_name else "❌"
        print(f"   {match} Sample ({label_name}): predicted → {result['prediction']}")

    # ------------------------------------------------------------------
    # Save metadata for the Streamlit app
    # ------------------------------------------------------------------
    # Compute dataset means for default slider values in the app
    means = raw_df[feature_names].mean().to_dict()
    mins = raw_df[feature_names].min().to_dict()
    maxs = raw_df[feature_names].max().to_dict()
    metadata = {
        "feature_names": feature_names,
        "label_map": label_map,
        "feature_means": means,
        "feature_mins": mins,
        "feature_maxs": maxs,
        "best_model_name": best_model_name,
    }
    metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"💾 Saved metadata: {metadata_path}")

    # ------------------------------------------------------------------
    # Package output
    # ------------------------------------------------------------------
    output = {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
        "prediction_fn": predict,
    }
    print(f"\n✅ Agent 4 complete. Model saved and prediction system verified.")
    print("=" * 70)
    return output


if __name__ == "__main__":
    from agent1_data_preprocessing import run as run_agent1
    from agent3_model_training import run as run_agent3
    a1 = run_agent1()
    a3 = run_agent3(a1)
    run(a1, a3)
