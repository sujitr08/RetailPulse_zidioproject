import ast
import re
from pathlib import Path

root = Path.cwd()
backupDir = root / "reports" / "modelReports" / "reproducibility" / "mlflowMigrationBackups"

prophetPath = root / "src" / "forecasting" / "trainProphet.py"
ensemblePath = root / "src" / "forecasting" / "evaluateForecastEnsemble.py"

prophetText = prophetPath.read_text(encoding="utf-8-sig")
ensembleText = ensemblePath.read_text(encoding="utf-8-sig")

for path in (prophetPath, ensemblePath):
    backup = backupDir / path.name
    if not backup.exists():
        raise FileNotFoundError(f"Backup missing: {backup}")

# Remove only the exact MLflow imports and directory configuration.
prophetText = re.sub(r"(?m)^import mlflow\s*\n", "", prophetText)
prophetText = re.sub(r'(?m)^MLFLOW_DIR = ROOT / "mlruns"\s*\n', "", prophetText)
prophetText = prophetText.replace(
    "for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR, MLFLOW_DIR):",
    "for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):",
)
prophetText = re.sub(
    r'(?m)^\s*print\(f"MLflow tracking directory: \{MLFLOW_DIR\}"\)\s*\n?',
    "",
    prophetText,
)

# Replace the Prophet logging block using stable start/end markers.
prophetStart = "    # Log metrics and outputs to MLflow for experiment tracking."
prophetEnd = '    print("\\nProphet baseline completed.")'

prophetReplacement = r'''    # Save a local reproducibility record with artifact hashes.
    import hashlib
    from datetime import datetime, timezone

    artifactPaths = [
        forecastFile, metricsFile, reportFile, modelFile,
        figureFile, componentsFile
    ]
    artifactRecords = []
    for artifactPath in artifactPaths:
        if artifactPath.exists():
            artifactRecords.append({
                "path": str(artifactPath.relative_to(ROOT)),
                "sha256": hashlib.sha256(artifactPath.read_bytes()).hexdigest(),
                "sizeBytes": artifactPath.stat().st_size,
            })

    runRecord = {
        "modelName": MODEL_NAME,
        "runType": "prophetBaseline",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "holdoutDays": HOLDOUT_DAYS,
            "yearlySeasonality": True,
            "weeklySeasonality": True,
            "dailySeasonality": False,
            "seasonalityMode": "additive",
            "changepointPriorScale": 0.05,
            "trainingRows": len(trainData),
            "holdoutRows": len(testData),
            "randomState": RANDOM_STATE,
        },
        "metrics": metrics,
        "artifacts": artifactRecords,
    }
    localRunLog = ROOT / "reports" / "modelReports" / "reproducibility" / "localExperimentRuns.jsonl"
    localRunLog.parent.mkdir(parents=True, exist_ok=True)
    with localRunLog.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runRecord, ensure_ascii=False) + "\n")
    print(f"Local experiment record saved: {localRunLog}")

'''

if prophetStart not in prophetText or prophetEnd not in prophetText:
    raise RuntimeError("Prophet logging block markers do not match; no files saved.")

prophetText = (
    prophetText[:prophetText.index(prophetStart)]
    + prophetReplacement
    + prophetText[prophetText.index(prophetEnd):]
)

# Replace the ensemble logging block.
ensembleText = re.sub(r"(?m)^import mlflow\s*\n", "", ensembleText)
ensembleStart = "    # Record the comparison in the project's SQLite-backed MLflow tracking store."
ensembleEnd = '    print("\\nForecast ensemble comparison completed.")'

ensembleReplacement = r'''    # Save a local reproducibility record with artifact hashes.
    import hashlib
    from datetime import datetime, timezone

    artifactPaths = [
        forecastFile, comparisonFile, metricsFile, reportFile, figureFile
    ]
    artifactRecords = []
    for artifactPath in artifactPaths:
        if artifactPath.exists():
            artifactRecords.append({
                "path": str(artifactPath.relative_to(ROOT)),
                "sha256": hashlib.sha256(artifactPath.read_bytes()).hexdigest(),
                "sizeBytes": artifactPath.stat().st_size,
            })

    runRecord = {
        "modelName": "prophetLstmEqualWeightEnsemble",
        "runType": "forecastEnsembleComparison",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "ensembleMethod": "fixedWeightedAverage",
            "prophetWeight": PROPHET_WEIGHT,
            "lstmWeight": LSTM_WEIGHT,
            "holdoutDays": len(combinedData),
        },
        "metrics": comparison,
        "artifacts": artifactRecords,
    }
    localRunLog = ROOT / "reports" / "modelReports" / "reproducibility" / "localExperimentRuns.jsonl"
    localRunLog.parent.mkdir(parents=True, exist_ok=True)
    with localRunLog.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(runRecord, ensure_ascii=False) + "\n")
    print(f"Local experiment record saved: {localRunLog}")

'''

if ensembleStart not in ensembleText or ensembleEnd not in ensembleText:
    raise RuntimeError("Ensemble logging block markers do not match; no files saved.")

ensembleText = (
    ensembleText[:ensembleText.index(ensembleStart)]
    + ensembleReplacement
    + ensembleText[ensembleText.index(ensembleEnd):]
)

# Validate both proposed files before writing either one.
for name, source in (
    ("trainProphet.py", prophetText),
    ("evaluateForecastEnsemble.py", ensembleText),
):
    if re.search(r"(?i)mlflow", source):
        raise RuntimeError(f"MLflow reference remains in proposed {name}; no files saved.")
    ast.parse(source, filename=name)

# Write only after both validations succeed.
prophetPath.write_text(prophetText, encoding="utf-8")
ensemblePath.write_text(ensembleText, encoding="utf-8")

print("UPDATED:", prophetPath.relative_to(root))
print("UPDATED:", ensemblePath.relative_to(root))
print("Both files passed Python syntax validation.")
print("Existing backups preserved.")
