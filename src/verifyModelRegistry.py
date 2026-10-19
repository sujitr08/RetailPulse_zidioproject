import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"
registry_path = REGISTRY / "registryManifest.json"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not registry_path.is_file():
    raise FileNotFoundError(f"Missing registry manifest: {registry_path}")

registry = json.loads(registry_path.read_text(encoding="utf-8"))
errors = []
checked_versions = 0
checked_artifacts = 0

for entry in registry.get("versions", []):
    manifest_path = ROOT / entry["manifest"]
    model_name = entry["modelName"]
    version = entry["version"]

    if not manifest_path.is_file():
        errors.append(f"{model_name} {version}: missing manifest {entry['manifest']}")
        continue

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{model_name} {version}: invalid manifest JSON: {exc}")
        continue

    if manifest.get("modelName") != model_name:
        errors.append(f"{model_name} {version}: model name mismatch in manifest")
    if manifest.get("version") != version:
        errors.append(f"{model_name} {version}: version mismatch in manifest")

    artifacts = manifest.get("artifacts", [])
    if len(artifacts) != entry.get("artifactCount"):
        errors.append(
            f"{model_name} {version}: registry count {entry.get('artifactCount')} "
            f"!= manifest count {len(artifacts)}"
        )

    version_ok = True
    for artifact in artifacts:
        artifact_path = manifest_path.parent / artifact["file"]
        checked_artifacts += 1

        if not artifact_path.is_file():
            errors.append(f"{model_name} {version}: missing {artifact['file']}")
            version_ok = False
            continue

        actual_size = artifact_path.stat().st_size
        actual_hash = sha256(artifact_path)

        if actual_size != artifact.get("sizeBytes"):
            errors.append(f"{model_name} {version}: size mismatch for {artifact['file']}")
            version_ok = False
        if actual_hash != artifact.get("sha256"):
            errors.append(f"{model_name} {version}: SHA-256 mismatch for {artifact['file']}")
            version_ok = False

    checked_versions += 1
    print(f"{'PASS' if version_ok else 'FAIL'}: {model_name} {version} "
          f"({len(artifacts)} artifacts)")

print(f"\nVersions checked: {checked_versions}")
print(f"Artifacts checked: {checked_artifacts}")
print(f"Errors: {len(errors)}")

if errors:
    for error in errors:
        print("ERROR:", error)
    raise SystemExit(1)

print("PASS: all registry entries and artifact hashes are consistent.")
