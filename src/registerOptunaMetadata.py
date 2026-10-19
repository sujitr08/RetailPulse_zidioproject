import hashlib
import json
import shutil
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"
REPORTS = ROOT / "reports" / "modelReports"
MODEL_SOURCE = ROOT / "models" / "churnXgboostOptuna.json"
PARAMETERS_SOURCE = REPORTS / "churnOptunaBestParameters.json"
REPORT_SOURCE = REPORTS / "churnOptunaTuningReport.txt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    for path in (MODEL_SOURCE, PARAMETERS_SOURCE, REPORT_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact missing: {path}")

    payload = json.loads(PARAMETERS_SOURCE.read_text(encoding="utf-8"))
    features = payload.get("features")
    cutoff = payload.get("validationCutoff")

    if not isinstance(features, list) or not features:
        raise ValueError("Optuna JSON has no valid non-empty features list.")
    if not cutoff:
        raise ValueError("Optuna JSON has no validationCutoff.")

    # Verify the feature list against the actual saved XGBoost model.
    booster = xgb.Booster()
    booster.load_model(str(MODEL_SOURCE))
    model_features = booster.feature_names

    if model_features and model_features != features:
        raise ValueError(
            "Feature mismatch between metadata and model:\n"
            f"JSON: {features}\nModel: {model_features}"
        )

    metadata = {
        "featureColumns": features,
        "testCutoff": cutoff,
        "modelType": "XGBoost",
        "studyName": payload["studyName"],
        "bestTrial": payload["bestTrial"],
        "validationMetrics": payload["validationMetrics"],
        "trainingSnapshotCutoffs": payload["trainingSnapshotCutoffs"],
        "limitations": payload.get("limitations", []),
    }

    model_dir = REGISTRY / "churnOptuna"
    existing_versions = [
        int(folder.name[1:])
        for folder in model_dir.iterdir()
        if folder.is_dir()
        and folder.name.startswith("v")
        and folder.name[1:].isdigit()
    ]
    version = f"v{max(existing_versions, default=0) + 1}"
    version_dir = model_dir / version

    if version_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {version_dir}")

    registry_path = REGISTRY / "registryManifest.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    staged = [
        (MODEL_SOURCE, MODEL_SOURCE.name),
        (PARAMETERS_SOURCE, PARAMETERS_SOURCE.name),
        (REPORT_SOURCE, REPORT_SOURCE.name),
    ]

    version_dir.mkdir(parents=False, exist_ok=False)

    try:
        artifacts = []
        for source, filename in staged:
            destination = version_dir / filename
            shutil.copy2(source, destination)
            if sha256(source) != sha256(destination):
                raise RuntimeError(f"SHA-256 mismatch: {destination}")

            artifacts.append({
                "file": filename,
                "source": str(source.relative_to(ROOT)),
                "sha256": sha256(destination),
                "sizeBytes": destination.stat().st_size,
                "verifiedCopy": True,
            })

        metadata_path = version_dir / "churnOptunaModelMetadata.joblib"
        joblib.dump(metadata, metadata_path)

        artifacts.append({
            "file": metadata_path.name,
            "source": "generated from churnOptunaBestParameters.json",
            "sha256": sha256(metadata_path),
            "sizeBytes": metadata_path.stat().st_size,
            "verifiedCopy": True,
        })

        version_manifest = {
            "modelName": "churnOptuna",
            "version": version,
            "registeredAtUtc": datetime.now(timezone.utc).isoformat(),
            "pythonVersion": sys.version,
            "platform": platform.platform(),
            "artifacts": artifacts,
            "notes": (
                "Optuna model snapshot with metadata derived from the saved "
                "tuning payload. Validation metrics are offline results, "
                "not a claim of production performance."
            ),
        }

        manifest_path = version_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(version_manifest, indent=2),
            encoding="utf-8",
        )

        registry.setdefault("versions", []).append({
            "modelName": "churnOptuna",
            "version": version,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "artifactCount": len(artifacts),
        })
        registry["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

        print(f"PASS: registered churnOptuna {version}")
        print(f"Feature count: {len(features)}")
        print(f"Validation cutoff: {cutoff}")
        print(f"Artifacts registered: {len(artifacts)}")
        print(f"Registry version manifest: {manifest_path}")

    except Exception:
        shutil.rmtree(version_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
