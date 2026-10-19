import json
from pathlib import Path

import joblib
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "modelReports"
MODELS = ROOT / "models" / "registry"

def inspect_json(path):
    print(f"\n--- JSON: {path.relative_to(ROOT)} ---")
    data = json.loads(path.read_text(encoding="utf-8"))
    print("Top-level type:", type(data).__name__)
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:40])
    else:
        print("Items:", len(data))

def inspect_joblib(path):
    print(f"\n--- JOBLIB: {path.relative_to(ROOT)} ---")
    data = joblib.load(path)
    print("Object type:", type(data).__name__)
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:50])
    elif isinstance(data, (list, tuple)):
        print("Length:", len(data))
        print("First items:", repr(data[:5]))

def main():
    inspect_json(REPORTS / "prophetBaselineModel.json")

    lstm_path = MODELS / "lstmBaseline" / "v1" / "lstmBaselineModel.pt"
    print(f"\n--- TORCH CHECKPOINT: {lstm_path.relative_to(ROOT)} ---")
    checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=True)
    print("Object type:", type(checkpoint).__name__)
    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys())[:40])
        for key, value in list(checkpoint.items())[:12]:
            if hasattr(value, "shape"):
                print(f"  {key}: shape={tuple(value.shape)}")
            else:
                print(f"  {key}: type={type(value).__name__}")

    for model_name, filename in [
        ("churnBaseline", "churnModelMetadata.joblib"),
        ("churnTemporal", "churnTemporalModelMetadata.joblib"),
    ]:
        inspect_joblib(MODELS / model_name / "v1" / filename)

    print("\n--- XGBOOST MODEL LOAD TEST ---")
    for model_name, filename in [
        ("churnBaseline", "churnXgboostModel.json"),
        ("churnTemporal", "churnXgboostTemporalModel.json"),
        ("churnOptuna", "churnXgboostOptuna.json"),
    ]:
        path = MODELS / model_name / "v1" / filename
        model = xgb.XGBClassifier()
        model.load_model(path)
        print(f"PASS: {model_name} loaded; features={model.n_features_in_}")

if __name__ == "__main__":
    main()
