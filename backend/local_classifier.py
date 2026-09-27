"""Run the offline MobileNetV2 leaf classifier and map it to the Haran diagnosis shape."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "model"
MODEL_PYTHON = MODEL_DIR / ".venv" / "Scripts" / "python.exe"
PREDICT_SCRIPT = MODEL_DIR / "predict_model.py"
WEIGHTS = MODEL_DIR / "olive_classifier.keras"

def available():
    return WEIGHTS.is_file() and MODEL_PYTHON.is_file() and PREDICT_SCRIPT.is_file()


def predict_leaf(image_bytes: bytes, lat: float, lng: float):
    if not available() or not image_bytes:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        image_path = handle.name

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [str(MODEL_PYTHON), str(PREDICT_SCRIPT), "--image", image_path, "--model_dir", str(MODEL_DIR)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            env=env,
        )
    finally:
        Path(image_path).unlink(missing_ok=True)

    if completed.returncode != 0:
        return None

    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    diseased = bool(raw.get("detected"))
    raw_severity = str(raw.get("severity") or "").strip().lower()
    if raw_severity in {"high", "severe"}:
        severity = "High"
    elif raw_severity in {"medium", "moderate"}:
        severity = "Medium"
    else:
        severity = "Low"
    voice = raw.get("voice_script") or ""
    if not voice:
        return None

    return {
        "diagnosis": {
            "disease_name_ar": raw.get("disease_name_en") or ("Diseased leaf" if diseased else "Healthy leaf"),
            "status": "sick" if diseased else "healthy",
            "severity": severity,
            "requires_expert_confirmation": diseased,
        },
        "voice_script": voice,
        "map": {
            "center": {"lat": lat, "lng": lng},
            "contagion_radius_meters": 0 if not diseased else int(raw.get("contagion_radius_meters") or 12),
        },
        "source": "local_classifier",
        "confidence": raw.get("confidence"),
    }
