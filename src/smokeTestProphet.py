import numpy as np
from pathlib import Path
from prophet.serialize import model_from_json

root = Path(__file__).resolve().parents[1]
path = root / "reports" / "modelReports" / "prophetBaselineModel.json"

model = model_from_json(path.read_text(encoding="utf-8"))
history = model.history

print("History available:", history is not None)
if history is None or history.empty:
    raise RuntimeError("Prophet model has no usable history")

future = model.make_future_dataframe(periods=7, freq="D")
forecast = model.predict(future)

if forecast.empty or not np.isfinite(forecast["yhat"]).all():
    raise RuntimeError("Prophet forecast is empty or contains non-finite values")

print("PASS: Prophet model deserialized and predicted successfully.")
print("History rows:", len(history))
print("Forecast rows:", len(forecast))
print("Forecast horizon: 7 days")
print("Last 7 predicted values:", np.round(forecast["yhat"].tail(7), 2).tolist())
