import base64
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "qwen/qwen3.8-27b"

VALID_STATUSES = {"sick", "healthy"}
VALID_SEVERITIES = {"Low", "Medium", "High"}

MORNAG_LAT = 36.6742
MORNAG_LNG = 10.2854
_MOCK_CURSOR = 0

PEACOCK_VOICE = (
    "Farmer, this tree has peacock spot. Spray copper sulfate on this tree "
    "and the trees around it before the rain, so the disease does not spread."
)
KNOT_VOICE = (
    "Farmer, this tree has olive knot. Cut off the galled branches and sterilize "
    "your tools with alcohol or a flame before you prune the next tree."
)
HEALTHY_VOICE = (
    "Farmer, this tree looks healthy. Keep watering and light pruning. "
    "No spray is needed today."
)

PATHOLOGIST_PROMPT = """You check one olive leaf photo for a family farmer. Return only JSON with these keys:
{
  "diagnosis": {
    "disease_name_ar": "English name of what you see",
    "status": "healthy",
    "severity": "Low",
    "requires_expert_confirmation": false
  },
  "voice_script": "Farmer, ...",
  "map": {
    "center": {"lat": 36.6742, "lng": 10.2854},
    "contagion_radius_meters": 0
  }
}

Keep the severity that matches the leaf. Do not change Low into Medium. Do not change High into Medium.
- No clear disease, or only veins, dust, shadows, insect holes, or yellow tips: status healthy, severity Low, radius 0. Do not name peacock spot. Do not mention spray.
- A few dark round spots: status sick, severity Low. Name peacock spot only when those spots are round and dark. Tell the farmer to watch the leaf. Do not say spray.
- Spots spread across much of the leaf, but not most of it: severity Medium.
- Spots or galls cover most of the leaf: severity High.
- requires_expert_confirmation is true only for Medium or High.
- voice_script is plain English, one or two sentences, and starts with "Farmer,".
"""


def _payload(lat, lng, disease_name, status, severity, voice_script, radius, expert):
    return {
        "diagnosis": {
            "disease_name_ar": disease_name,
            "status": status,
            "severity": severity,
            "requires_expert_confirmation": expert,
        },
        "voice_script": voice_script,
        "map": {
            "center": {"lat": lat, "lng": lng},
            "contagion_radius_meters": radius,
        },
    }


def _peacock_case(lat: float, lng: float) -> dict:
    return _payload(
        lat,
        lng,
        "عين الطاووس (Spilocaea oleagina)",
        "sick",
        "Medium",
        PEACOCK_VOICE,
        12,
        True,
    )


def _knot_case(lat: float, lng: float) -> dict:
    return _payload(
        lat,
        lng,
        "سل الزيتون (Pseudomonas savastanoi)",
        "sick",
        "High",
        KNOT_VOICE,
        18,
        True,
    )


def _healthy_case(lat: float, lng: float) -> dict:
    return _payload(
        lat,
        lng,
        "شجرة لاباس وصحيحة",
        "healthy",
        "Low",
        HEALTHY_VOICE,
        0,
        False,
    )


MOCK_CASES = (_peacock_case, _knot_case, _healthy_case)


def _next_mock(lat: float, lng: float) -> dict:
    global _MOCK_CURSOR
    case = MOCK_CASES[_MOCK_CURSOR % len(MOCK_CASES)]
    _MOCK_CURSOR += 1
    return case(lat, lng)


def _fallback_payload(lat: float, lng: float) -> dict:
    return _next_mock(lat, lng)


def _extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Groq response did not contain JSON")
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _canon_severity(value) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"low", "mild", "slight", "light"}:
        return "Low"
    if text in {"medium", "moderate", "mod"}:
        return "Medium"
    if text in {"high", "severe"}:
        return "High"
    return None


def _canon_status(value, disease_name: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"healthy", "ok", "fine", "clean"}:
        return "healthy"
    if text in {"sick", "diseased", "infected"}:
        return "sick"
    if "health" in disease_name.lower():
        return "healthy"
    return "uncertain"


def _normalize_payload(data: dict, lat: float, lng: float) -> dict:
    diagnosis = data.get("diagnosis") if isinstance(data.get("diagnosis"), dict) else {}
    disease_name = str(diagnosis.get("disease_name_ar") or "Olive leaf")
    status = _canon_status(diagnosis.get("status"), disease_name)
    severity = _canon_severity(diagnosis.get("severity")) or "Low"

    if status == "healthy":
        disease_name = "Healthy leaf"
        severity = "Low"

    voice_script = data.get("voice_script") or data.get("derja_spoken_prescription")
    if not isinstance(voice_script, str) or not voice_script.strip():
        voice_script = HEALTHY_VOICE if status == "healthy" else (
            "Farmer, this leaf has only a few spots. Severity is low. Watch the tree. No spray is needed today."
        )

    map_block = data.get("map") if isinstance(data.get("map"), dict) else {}
    center = map_block.get("center") if isinstance(map_block.get("center"), dict) else {}
    try:
        center_lat = float(center.get("lat", lat))
        center_lng = float(center.get("lng", lng))
    except (TypeError, ValueError):
        center_lat, center_lng = lat, lng

    try:
        radius = int(map_block.get("contagion_radius_meters", 12 if status == "sick" else 0))
    except (TypeError, ValueError):
        radius = 12 if status == "sick" else 0

    if status != "sick":
        radius = 0
        severity = "Low"
    if severity == "Low":
        radius = 0
        if re.search(r"\bspray\b", voice_script, flags=re.IGNORECASE):
            voice_script = (
                "Farmer, this leaf has only a few spots. Severity is low. "
                "Watch the tree. No spray is needed today."
            )

    return {
        "diagnosis": {
            "disease_name_ar": str(disease_name),
            "status": status if status in {"healthy", "sick", "uncertain"} else "uncertain",
            "severity": severity,
            "requires_expert_confirmation": status == "sick" and severity in {"Medium", "High"},
        },
        "voice_script": voice_script.strip(),
        "map": {
            "center": {"lat": center_lat, "lng": center_lng},
            "contagion_radius_meters": radius,
        },
    }


def _call_groq(image_bytes: bytes, lat: float, lng: float) -> dict:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    mime = "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"

    encoded = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        f"{PATHOLOGIST_PROMPT}\n"
        f"إحداثيات الشجرة المعطاة: lat={lat}, lng={lng}. "
        "حط نفس الإحداثيات في map.center."
    )

    response = requests.post(
        GROQ_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0.2,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
        },
        timeout=25,
    )
    response.raise_for_status()
    body = response.json()
    raw_text = body["choices"][0]["message"]["content"]
    return _normalize_payload(_extract_json(raw_text), lat, lng)


def _uncertain_payload(lat: float, lng: float) -> dict:
    return {
        "diagnosis": {
            "disease_name_ar": "Take a closer photo",
            "status": "uncertain",
            "severity": "Low",
            "requires_expert_confirmation": False,
        },
        "voice_script": (
            "Farmer, I cannot tell from this photo. Hold one leaf close to the camera "
            "so the leaf fills the picture."
        ),
        "map": {
            "center": {"lat": lat, "lng": lng},
            "contagion_radius_meters": 0,
        },
        "source": "uncertain",
    }


def _kept_area(mask: np.ndarray, leaf_area: int, max_fraction: float, min_area: int) -> int:
    raw = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    total = 0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_fraction * leaf_area:
            continue
        elongation = max(width, height) / max(min(width, height), 1)
        if elongation > 5 and area < 350:
            continue
        total += area
    return total


def measure_leaf_damage(image_bytes: bytes):
    """Score how much of the leaf is necrotic or spotted. The silver underside of a healthy leaf is ignored."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    height, width = image.shape[:2]
    if width > 960:
        image = cv2.resize(image, (960, max(1, int(height * 960 / width))))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    leaf = (value > 25) & ((green > 18) | (red > 18))
    leaf_area = int(leaf.sum())
    if leaf_area < 800:
        return None

    brown = leaf & (hue <= 26) & (saturation > 50) & (red > green + 18) & (value > 35) & (value < 200)
    core = leaf & (value < 78) & (saturation < 90) & (gray < 90)
    brown_part = _kept_area(brown, leaf_area, 0.70, 25) / leaf_area
    core_part = _kept_area(core, leaf_area, 0.12, 8) / leaf_area
    damage = min(1.0, brown_part + core_part)

    if brown_part >= 0.12 or (brown_part >= 0.08 and core_part >= 0.04):
        band = "High"
    elif brown_part >= 0.04 or core_part >= 0.12:
        band = "Medium"
    elif brown_part >= 0.012 or core_part >= 0.05:
        band = "Low"
    else:
        band = "healthy"

    return {
        "damage": damage,
        "brown": brown_part,
        "core": core_part,
        "band": band,
    }


def _solution_for(band: str, disease_name: str) -> dict:
    if band == "healthy":
        return {
            "status": "healthy",
            "severity": "Low",
            "disease_name_ar": "Healthy leaf",
            "expert": False,
            "radius": 0,
            "check_again_days": 14,
            "solution": "No spray. Keep watering and light pruning. Check this tree again in 14 days.",
            "voice_script": HEALTHY_VOICE,
        }
    if band == "Low":
        return {
            "status": "sick",
            "severity": "Low",
            "disease_name_ar": disease_name,
            "expert": False,
            "radius": 0,
            "check_again_days": 7,
            "solution": "Only a small part of the leaf is marked. Do not spray yet. Check this same tree again in 7 days.",
            "voice_script": (
                "Farmer, only a small part of this leaf is damaged. Severity is low. "
                "Do not spray today. Check this tree again in 7 days."
            ),
        }
    if band == "Medium":
        return {
            "status": "sick",
            "severity": "Medium",
            "disease_name_ar": disease_name,
            "expert": True,
            "radius": 12,
            "check_again_days": 3,
            "solution": (
                "Spots cover a real part of the leaf. If they are peacock spot, spray copper before the next rain "
                "and check the trees beside this one within 3 days."
            ),
            "voice_script": (
                "Farmer, this leaf has a moderate infection. Severity is medium. "
                "Check the trees beside it within 3 days, and ask an expert before you spray."
            ),
        }
    return {
        "status": "sick",
        "severity": "High",
        "disease_name_ar": disease_name,
        "expert": True,
        "radius": 20,
        "check_again_days": 1,
        "solution": (
            "A large part of the leaf is dead or covered in spots. Treat this tree today and check the nearby trees within 1 day."
        ),
        "voice_script": (
            "Farmer, a large part of this leaf is damaged. Severity is high. "
            "Check this tree and the trees around it today."
        ),
    }


def _apply_measured_severity(payload: dict, image_bytes: bytes, lat: float, lng: float) -> dict:
    measured = measure_leaf_damage(image_bytes) if image_bytes else None
    if not measured:
        payload["solution"] = payload.get("voice_script", "")
        payload["check_again_days"] = 7
        return payload

    disease_name = str(payload.get("diagnosis", {}).get("disease_name_ar") or "Olive leaf spots")
    if measured["band"] == "healthy" or "health" in disease_name.lower() or disease_name in {
        "Take a closer photo",
        "Olive leaf",
        "Healthy leaf",
    }:
        if measured["band"] == "healthy":
            disease_name = "Healthy leaf"
        elif measured["band"] != "healthy":
            disease_name = "Olive leaf spots"
    plan = _solution_for(measured["band"], disease_name if measured["band"] != "healthy" else "Healthy leaf")
    diagnosis = payload.setdefault("diagnosis", {})
    diagnosis["status"] = plan["status"]
    diagnosis["severity"] = plan["severity"]
    diagnosis["disease_name_ar"] = plan["disease_name_ar"]
    diagnosis["requires_expert_confirmation"] = plan["expert"]
    payload["voice_script"] = plan["voice_script"]
    payload["solution"] = plan["solution"]
    payload["check_again_days"] = plan["check_again_days"]
    payload["leaf_damage_percent"] = round(measured["damage"] * 100, 1)
    payload["severity_source"] = "leaf-area"
    center = payload.get("map", {}).get("center", {"lat": lat, "lng": lng})
    payload["map"] = {
        "center": center,
        "contagion_radius_meters": plan["radius"],
    }
    print(
        "leaf area",
        f"{payload['leaf_damage_percent']}%",
        plan["severity"] if plan["status"] == "sick" else "healthy",
    )
    return payload


def analyze_leaf(image_bytes: bytes, lat: float = MORNAG_LAT, lng: float = MORNAG_LNG) -> dict:
    local = None
    try:
        from local_classifier import predict_leaf

        local = predict_leaf(image_bytes, lat, lng)
    except Exception:
        local = None

    groq = None
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key and image_bytes:
        try:
            groq = _call_groq(image_bytes, lat, lng)
        except Exception as error:
            print("diagnosis source: groq failed", error)

    local_status = local["diagnosis"]["status"] if local else None
    groq_status = groq["diagnosis"]["status"] if groq else None
    chosen = None

    if local and groq:
        if local_status == "healthy" and groq_status == "healthy":
            print("diagnosis source: agreed healthy")
            groq["source"] = "agreed"
            chosen = groq
        elif local_status == "sick" and groq_status == "sick":
            print("diagnosis source: agreed sick", groq["diagnosis"]["severity"])
            groq["source"] = "agreed"
            chosen = groq
        elif groq_status == "healthy":
            print("diagnosis source: groq healthy, local classifier disagreed")
            groq["source"] = "groq"
            chosen = groq
        else:
            print("diagnosis source: uncertain, local", local_status, "groq", groq_status)
            chosen = _uncertain_payload(lat, lng)
    elif groq:
        print("diagnosis source: groq", groq_status)
        chosen = groq
    elif local and local_status == "healthy":
        print("diagnosis source: local_classifier healthy")
        chosen = local
    else:
        print("diagnosis source: uncertain")
        chosen = _uncertain_payload(lat, lng)

    return _apply_measured_severity(chosen, image_bytes, lat, lng)
