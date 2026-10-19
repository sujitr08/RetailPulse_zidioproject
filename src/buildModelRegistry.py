import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"
REPORTS = ROOT / "reports" / "modelReports"
REPRO = REPORTS / "reproducibility"

CANDIDATES = {
    "prophetBaseline": [
        REPORTS / "prophetBaselineModel.json",
        REPORTS / "prophetBaselineMetrics.json",
        REPORTS / "prophetBaselineReport.txt",
    ],
    "lstmBaseline": [
        REPORTS / "lstmBaselineModel.pt",
        REPORTS / "lstmBaselineMetrics.json",
        REPORTS / "lstmBaselineReport.txt",
    ],
    "forecastEnsemble": [
        REPORTS / "forecastEnsembleMetrics.json",
        REPORTS / "forecastEnsembleReport.txt",
        REPORTS / "ensembleHoldoutForecast.csv",
    ],
    "churnBaseline": [
        ROOT / "models" / "churnXgboostModel.json",
        ROOT / "models" / "churnModelMetadata.joblib",
        REPORTS / "churnModelMetrics.json",
        REPORTS / "churnModelReport.txt",
    ],
    "churnTemporal": [
        ROOT / "models" / "churnXgboostTemporalModel.json",
        ROOT / "models" / "churnTemporalModelMetadata.joblib",
        REPORTS / "churnTemporalValidationMetrics.json",
    ],
    "churnOptuna": [
        ROOT / "models" / "churnXgboostOptuna.json",
        ROOT / "models" / "churnXgboostOptuna.json",
        REPORTS / "churnOptunaTuningReport.txt",
    ],
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": "RetailPulse",
        "registryType": "local-file-based",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "versions": [],
        "missingArtifacts": [],
    }

    for model_name, candidates in CANDIDATES.items():
        existing = list(dict.fromkeys(p for p in candidates if p.is_file()))
        if not existing:
            manifest["missingArtifacts"].append({
                "model": model_name,
                "expected": [str(p.relative_to(ROOT)) for p in candidates],
            })
            print(f"SKIP: {model_name} — no source artifacts found")
            continue

        version_dir = REGISTRY / model_name / "v1"
        version_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []

        for source in existing:
            destination = version_dir / source.name
            shutil.copy2(source, destination)
            source_hash = sha256(source)
            copied_hash = sha256(destination)

            if source_hash != copied_hash:
                raise RuntimeError(f"Checksum mismatch: {destination}")

            artifacts.append({
                "file": destination.name,
                "source": str(source.relative_to(ROOT)),
                "sha256": copied_hash,
                "sizeBytes": destination.stat().st_size,
                "verifiedCopy": True,
            })

        version_manifest = {
            "modelName": model_name,
            "version": "v1",
            "registeredAtUtc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
            "notes": "Existing artifact snapshot; not a claim of production validation.",
        }

        manifest_path = version_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(version_manifest, indent=2),
            encoding="utf-8",
        )

        manifest["versions"].append({
            "modelName": model_name,
            "version": "v1",
            "manifest": str(manifest_path.relative_to(ROOT)),
            "artifactCount": len(artifacts),
        })
        print(f"PASS: {model_name} v1 — {len(artifacts)} checksum-verified artifacts")

    manifest_path = REGISTRY / "registryManifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    environment = {
        "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "dependencySnapshot": str(
            (REPRO / "environmentSnapshot.txt").relative_to(ROOT)
        ),
        "registryManifest": str(manifest_path.relative_to(ROOT)),
    }
    (REPRO / "reproducibilityManifest.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    print("\nRegistry summary:")
    print(f"Registered model versions: {len(manifest['versions'])}")
    print(f"Missing model groups: {len(manifest['missingArtifacts'])}")
    print(f"Registry manifest: {manifest_path}")
    print(f"Reproducibility manifest: {REPRO / 'reproducibilityManifest.json'}")

    if not manifest["versions"]:
        raise RuntimeError("No model versions registered; inspect source artifact paths.")

if __name__ == "__main__":
    main()

