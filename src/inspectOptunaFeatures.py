from pathlib import Path
import xgboost as xgb

root = Path(__file__).resolve().parents[1]
model = xgb.XGBClassifier()
model.load_model(root / "models" / "churnXgboostOptuna.json")

print("Expected feature count:", model.n_features_in_)
print("Stored feature names:", model.get_booster().feature_names)
