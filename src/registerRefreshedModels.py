import hashlib
import json
import shutil
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"
REPORTS = ROOT / "reports" / "modelReports"
FIGURES = ROOT / "reports" / "figures"
DATA = ROOT / "data" / "processedData"

MODEL_ARTIFACTS = {
    "prophetBaseline": [
        REPORTS / "prophetBaselineModel.json",
        REPORTS / "prophetBaselineMetrics.json",
        REPORTS / "prophetBaselineReport.txt",
        DATA / "prophetHoldoutForecast.csv",
        FIGURES / "prophetHoldoutForecast.png",
        FIGURES / "prophetForecastComponents.png",
    ],
    "forecastEnsemble": [
        REPORTS / "forecastEnsembleMetrics.json",
        REPORTS / "forecastEnsembleReport.txt",
        REPORTS / "forecastModelComparison.csv",
        DATA / "ensembleHoldoutForecast.csv",
        FIGURES / "forecastModelComparison.png",
    ],
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    registryManifestPath = REGISTRY / "registryManifest.json"
    if not registryManifestPath.exists():
        raise FileNotFoundError(f"Registry manifest missing: {registryManifestPath}")

    # Validate all sources before changing the registry.
    for modelName, sources in MODEL_ARTIFACTS.items():
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{modelName}: required source artifacts missing:\n"
                + "\n".join(missing)
            )

    registryManifest = json.loads(
        registryManifestPath.read_text(encoding="utf-8")
    )

    for modelName, sources in MODEL_ARTIFACTS.items():
        modelDir = REGISTRY / modelName
        modelDir.mkdir(parents=True, exist_ok=True)

        existingNumbers = []
        for folder in modelDir.iterdir():
            if folder.is_dir() and folder.name.startswith("v"):
                suffix = folder.name[1:]
                if suffix.isdigit():
                    existingNumbers.append(int(suffix))

        nextVersion = max(existingNumbers, default=0) + 1
        versionName = f"v{nextVersion}"
        versionDir = modelDir / versionName

        # Never overwrite an existing version.
        if versionDir.exists():
            raise FileExistsError(f"Refusing to overwrite {versionDir}")

        versionDir.mkdir(parents=False, exist_ok=False)
        artifacts = []

        try:
            for source in sources:
                destination = versionDir / source.name
                shutil.copy2(source, destination)

                sourceHash = sha256(source)
                copiedHash = sha256(destination)
                if sourceHash != copiedHash:
                    raise RuntimeError(f"Checksum mismatch: {destination}")

                artifacts.append({
                    "file": destination.name,
                    "source": str(source.relative_to(ROOT)),
                    "sha256": copiedHash,
                    "sizeBytes": destination.stat().st_size,
                    "verifiedCopy": True,
                })

            versionManifest = {
                "modelName": modelName,
                "version": versionName,
                "registeredAtUtc": datetime.now(timezone.utc).isoformat(),
                "pythonVersion": sys.version,
                "platform": platform.platform(),
                "artifacts": artifacts,
                "notes": (
                    "Refreshed local artifact snapshot with verified hashes; "
                    "not a claim of production validation."
                ),
            }

            manifestPath = versionDir / "manifest.json"
            manifestPath.write_text(
                json.dumps(versionManifest, indent=2),
                encoding="utf-8",
            )

            registryManifest.setdefault("versions", []).append({
                "modelName": modelName,
                "version": versionName,
                "manifest": str(manifestPath.relative_to(ROOT)),
                "artifactCount": len(artifacts),
            })

            print(
                f"PASS: {modelName} {versionName} — "
                f"{len(artifacts)} artifacts copied and verified"
            )

        except Exception:
            # Remove only the new incomplete version; never touch old versions.
            shutil.rmtree(versionDir, ignore_errors=True)
            raise

    registryManifest["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
    registryManifestPath.write_text(
        json.dumps(registryManifest, indent=2),
        encoding="utf-8",
    )
    print(f"\nUpdated registry manifest: {registryManifestPath}")
    print("Previous model versions were not overwritten.")

if __name__ == "__main__":
    main()
