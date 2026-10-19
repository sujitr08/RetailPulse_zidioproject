import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from prophet.serialize import model_from_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"


def test_churn_model(model_name, version, model_file, metadata_file):
    version_dir = REGISTRY / model_name / version
    model_path = version_dir / model_file
    metadata_path = version_dir / metadata_file

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = joblib.load(metadata_path)
    features = metadata.get("featureColumns")

    if not isinstance(features, list) or not features:
        raise ValueError(f"{model_name} {version}: invalid featureColumns metadata")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Build a feature-named matrix when the saved booster exposes names.
    # The metadata remains the source of the expected column order.
    if booster.feature_names and booster.feature_names != features:
        raise ValueError(
            f"{model_name} {version}: feature mismatch\n"
            f"Metadata: {features}\nModel: {booster.feature_names}"
        )

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    sample = np.zeros((3, len(features)), dtype=np.float32)
    import pandas as pd
    sample_frame = pd.DataFrame(sample, columns=features)

    probabilities = model.predict_proba(sample_frame)

    if probabilities.shape != (3, 2):
        raise RuntimeError(
            f"{model_name} {version}: unexpected probability shape "
            f"{probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"{model_name} {version}: non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"{model_name} {version}: probabilities do not sum to 1")

    print(
        f"PASS: {model_name} {version} — "
        f"{len(features)} features, prediction shape {probabilities.shape}"
    )


def test_prophet(model_name, version, model_file):
    model_path = REGISTRY / model_name / version / model_file
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = model_from_json(model_path.read_text(encoding="utf-8"))
    history = model.history

    if history is None or history.empty:
        raise RuntimeError(f"{model_name} {version}: model history is missing")

    future = model.make_future_dataframe(periods=7, freq="D")
    forecast = model.predict(future)

    if forecast.empty or not np.isfinite(forecast["yhat"]).all():
        raise RuntimeError(f"{model_name} {version}: invalid forecast")

    print(
        f"PASS: {model_name} {version} — "
        f"{len(history)} history rows, 7-day forecast valid"
    )


def main():
    test_churn_model(
        "churnBaseline", "v1",
        "churnXgboostModel.json", "churnModelMetadata.joblib"
    )
    test_churn_model(
        "churnTemporal", "v1",
        "churnXgboostTemporalModel.json",
        "churnTemporalModelMetadata.joblib"
    )
    test_churn_model(
        "churnOptuna", "v2",
        "churnXgboostOptuna.json", "churnOptunaModelMetadata.joblib"
    )
    test_prophet(
        "prophetBaseline", "v2", "prophetBaselineModel.json"
    )
    print("\nPASS: all registered model smoke tests completed.")


if __name__ == "__main__":
    main()
