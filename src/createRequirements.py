from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
snapshot = root / "reports" / "modelReports" / "reproducibility" / "environmentSnapshot.txt"
output = root / "requirements.txt"

required = {
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "prophet",
    "torch",
    "pytorch-lightning",
    "xgboost",
    "shap",
    "optuna",
    "evidently",
    "great-expectations",
    "plotly",
    "openpyxl",
    "joblib",
    "statsmodels",
    "matplotlib",
}

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

if not snapshot.is_file():
    raise FileNotFoundError(f"Environment snapshot not found: {snapshot}")

installed = {}
for line in snapshot.read_text(encoding="utf-16").splitlines():
    line = line.strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)==([^=]+)$", line)
    if match:
        installed[normalize(match.group(1))] = (
            match.group(1), match.group(2)
        )

missing = sorted(name for name in required if name not in installed)
if missing:
    raise RuntimeError(
        "These required packages were not found in the snapshot: "
        + ", ".join(missing)
        + ". Do not generate an incomplete requirements.txt."
    )

lines = [
    f"{installed[name][0]}=={installed[name][1]}"
    for name in sorted(required)
]
output.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Created {output}")
print(f"Pinned direct dependencies: {len(lines)}")
print(output.read_text(encoding="utf-8"))

