from pathlib import Path
import json
import random
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
DAILY_FILE = ROOT / "data" / "processedData" / "dailySales.csv"
OUTPUT_DIR = ROOT / "data" / "processedData"
REPORT_DIR = ROOT / "reports" / "modelReports"
FIGURE_DIR = ROOT / "reports" / "figures"

SEQUENCE_LENGTH = 30
HOLDOUT_DAYS = 30
VALIDATION_DAYS = 60
BATCH_SIZE = 32
MAX_EPOCHS = 100
RANDOM_STATE = 42


class SalesLSTM(pl.LightningModule):
    def __init__(self, hiddenSize=32, learningRate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hiddenSize,
            num_layers=1,
            batch_first=True,
        )
        self.outputLayer = nn.Linear(hiddenSize, 1)
        self.lossFunction = nn.MSELoss()

    def forward(self, inputs):
        sequenceOutput, _ = self.lstm(inputs)
        return self.outputLayer(sequenceOutput[:, -1, :])

    def training_step(self, batch, batchIndex):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.lossFunction(predictions, targets)
        self.log("trainLoss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batchIndex):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.lossFunction(predictions, targets)
        self.log("validationLoss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learningRate,
        )


def makeSequences(values, sequenceLength):
    inputs = []
    targets = []

    for targetIndex in range(sequenceLength, len(values)):
        inputs.append(values[targetIndex - sequenceLength:targetIndex])
        targets.append(values[targetIndex])

    x = np.asarray(inputs, dtype=np.float32).reshape(-1, sequenceLength, 1)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    return x, y


def calculateMetrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = np.abs(actual - predicted)
    nonZeroMask = actual != 0
    totalActual = np.sum(np.abs(actual))

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mapeNonZeroDaysPercent": (
            float(np.mean(errors[nonZeroMask] / np.abs(actual[nonZeroMask])) * 100)
            if nonZeroMask.any() else None
        ),
        "wapePercent": (
            float(np.sum(errors) / totalActual * 100)
            if totalActual > 0 else None
        ),
        "evaluatedDays": int(len(actual)),
        "nonZeroActualDays": int(nonZeroMask.sum()),
    }


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    pl.seed_everything(RANDOM_STATE, workers=True)

    for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not DAILY_FILE.exists():
        raise FileNotFoundError(
            f"{DAILY_FILE} not found. Run prepareTimeSeries.py first."
        )

    dailyData = pd.read_csv(DAILY_FILE, parse_dates=["date"])
    dailyData = dailyData.sort_values("date").dropna(
        subset=["date", "dailySales"]
    )

    if dailyData["date"].duplicated().any():
        raise ValueError("Daily sales data contains duplicate dates.")

    sales = dailyData["dailySales"].to_numpy(dtype=np.float32)

    if np.any(sales < 0):
        raise ValueError("Daily sales must not contain negative values.")

    if len(sales) <= HOLDOUT_DAYS + VALIDATION_DAYS + SEQUENCE_LENGTH:
        raise ValueError("Insufficient observations for train/validation/holdout.")

    # Keep the final 30 days completely outside model fitting and validation.
    trainingSeries = sales[:-HOLDOUT_DAYS]
    actualHoldout = sales[-HOLDOUT_DAYS:]
    holdoutDates = dailyData["date"].iloc[-HOLDOUT_DAYS:].reset_index(drop=True)

    # Reserve the last 60 training days as a chronological validation segment.
    validationStart = len(trainingSeries) - VALIDATION_DAYS

    scaler = MinMaxScaler()
    scaler.fit(trainingSeries[:validationStart].reshape(-1, 1))
    scaledTraining = scaler.transform(trainingSeries.reshape(-1, 1)).flatten()

    xAll, yAll = makeSequences(scaledTraining, SEQUENCE_LENGTH)

    # Each sequence target corresponds to its index in scaledTraining.
    targetIndices = np.arange(SEQUENCE_LENGTH, len(scaledTraining))
    trainMask = targetIndices < validationStart
    validationMask = targetIndices >= validationStart

    xTrain = torch.tensor(xAll[trainMask], dtype=torch.float32)
    yTrain = torch.tensor(yAll[trainMask], dtype=torch.float32)
    xValidation = torch.tensor(xAll[validationMask], dtype=torch.float32)
    yValidation = torch.tensor(yAll[validationMask], dtype=torch.float32)

    trainLoader = DataLoader(
        TensorDataset(xTrain, yTrain),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    validationLoader = DataLoader(
        TensorDataset(xValidation, yValidation),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SalesLSTM(hiddenSize=32, learningRate=0.001)

    earlyStopping = EarlyStopping(
        monitor="validationLoss",
        patience=10,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[earlyStopping],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
    )

    print(f"Training rows: {len(trainingSeries) - VALIDATION_DAYS}")
    print(f"Validation rows: {VALIDATION_DAYS}")
    print(f"Holdout rows: {HOLDOUT_DAYS}")
    print(f"Holdout begins: {holdoutDates.iloc[0].date()}")
    print("Training LSTM...")

    trainer.fit(model, trainLoader, validationLoader)

    # Forecast recursively, feeding each prediction into the next sequence.
    model.eval()
    context = scaledTraining[-SEQUENCE_LENGTH:].astype(np.float32).tolist()
    scaledPredictions = []

    with torch.no_grad():
        for _ in range(HOLDOUT_DAYS):
            inputTensor = torch.tensor(
                context[-SEQUENCE_LENGTH:],
                dtype=torch.float32,
            ).reshape(1, SEQUENCE_LENGTH, 1)

            nextValue = float(model(inputTensor).cpu().numpy().reshape(-1)[0])
            scaledPredictions.append(nextValue)
            context.append(nextValue)

    predictions = scaler.inverse_transform(
        np.asarray(scaledPredictions).reshape(-1, 1)
    ).flatten()
    predictions = np.clip(predictions, 0, None)

    metrics = calculateMetrics(actualHoldout, predictions)
    metrics.update({
        "model": "lstmDailySalesBaseline",
        "sequenceLength": SEQUENCE_LENGTH,
        "hiddenSize": 32,
        "maxEpochs": MAX_EPOCHS,
        "validationDays": VALIDATION_DAYS,
        "trainingRows": int(len(trainingSeries)),
        "holdoutRows": HOLDOUT_DAYS,
        "holdoutStart": str(holdoutDates.iloc[0].date()),
        "holdoutEnd": str(holdoutDates.iloc[-1].date()),
        "scalerFitOnTrainingDataOnly": True,
        "forecastClippedAtZero": True,
        "forecastMethod": "recursive",
    })

    forecastFrame = pd.DataFrame({
        "date": holdoutDates,
        "actualSales": actualHoldout,
        "lstmForecast": predictions,
    })

    forecastFile = OUTPUT_DIR / "lstmHoldoutForecast.csv"
    metricsFile = REPORT_DIR / "lstmBaselineMetrics.json"
    reportFile = REPORT_DIR / "lstmBaselineReport.txt"
    modelFile = REPORT_DIR / "lstmBaselineModel.pt"
    figureFile = FIGURE_DIR / "lstmHoldoutForecast.png"

    forecastFrame.to_csv(forecastFile, index=False)
    torch.save(model.state_dict(), modelFile)
    metricsFile.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        forecastFrame["date"],
        forecastFrame["actualSales"],
        marker="o",
        label="Actual sales",
    )
    axis.plot(
        forecastFrame["date"],
        forecastFrame["lstmForecast"],
        marker="o",
        label="LSTM forecast",
    )
    axis.set_title("LSTM: 30-Day Holdout Evaluation")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales amount (source currency)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figureFile, dpi=150, bbox_inches="tight")
    plt.close(figure)

    reportLines = [
        "RETAILPULSE - LSTM BASELINE EVALUATION",
        "=" * 55,
        "Evaluation: final 30 calendar days, excluded from model fitting",
        f"Holdout period: {metrics['holdoutStart']} to {metrics['holdoutEnd']}",
        f"Training rows: {metrics['trainingRows']}",
        f"Validation days: {VALIDATION_DAYS}",
        f"Sequence length: {SEQUENCE_LENGTH}",
        f"Hidden units: 32",
        f"Maximum epochs: {MAX_EPOCHS}",
        "Optimizer: Adam",
        "Loss: mean squared error",
        "Early stopping: validation loss, patience 10",
        "Forecast: recursive, 30 steps",
        "",
        "METRICS",
        "-" * 55,
        f"MAE: {metrics['mae']:.4f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"MAPE on non-zero actual days: {metrics['mapeNonZeroDaysPercent']:.4f}%",
        f"WAPE: {metrics['wapePercent']:.4f}%",
        "",
        "LIMITATIONS",
        "-" * 55,
        "A single holdout does not establish performance across all seasons.",
        "Recursive forecast errors may accumulate across the 30-day horizon.",
        "The historical dataset ends in December 2011.",
        "The target is eligible positive sales, not net sales after returns.",
        "",
        "OUTPUTS",
        "-" * 55,
        str(forecastFile),
        str(metricsFile),
        str(modelFile),
        str(figureFile),
    ]
    reportFile.write_text("\n".join(reportLines) + "\n", encoding="utf-8")

    print("\nLSTM baseline completed.")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE (non-zero actual days): {metrics['mapeNonZeroDaysPercent']:.4f}%")
    print(f"WAPE: {metrics['wapePercent']:.4f}%")
    print(f"Metrics: {metricsFile}")

if __name__ == "__main__":
    main()
