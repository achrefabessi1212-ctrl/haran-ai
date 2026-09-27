import base64
import json
import os
import queue
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

COW_CLASS_ID = 19
# A standing cow seen from the side is already wider than it is tall.
# Only a very flat box means the cow is lying down.
LYING_ASPECT_RATIO = 2.15
STANDING_ASPECT_RATIO = 1.45
STILL_SECONDS = 20.0
HEALTH_CHECK_SECONDS = 25.0
BANNER_TEXT = "CHECK THIS COW: she may be sick. Go and look."
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "qwen/qwen3.8-27b"
COW_HEALTH_PROMPT = """Look at this barn photo. Decide if a cow looks sick.
Return only JSON: {"health":"well","reason":"One short English sentence."}
health must be well, unwell, or unsure.
A cow that is standing or lying calmly is well. Do not call her unwell only because she is lying down.
Say unwell only for a clear problem: a wound, swelling, discharge, a very thin body, collapse, or a cow that cannot get up.
If there is no cow, use unsure.
"""
JPEG_QUALITY = 80
_STATUS = {
    "camera": "starting",
    "cows": 0,
    "lying": 0,
    "posture": "none",
    "health": "none",
    "reason": "",
    "distressed": False,
    "dark": False,
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = str(BASE_DIR / "test_assets" / "cows.mp4")
_YOLO_MODEL = None
_YOLO_TRIED = False


def _load_yolo():
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_TRIED:
        return _YOLO_MODEL
    _YOLO_TRIED = True
    try:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO("yolov8n.pt")
    except Exception:
        _YOLO_MODEL = None
    return _YOLO_MODEL


def cow_status() -> dict:
    return dict(_STATUS)


def _set_status(**values):
    _STATUS.update(values)


def _open_webcam():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
    for index in (0, 1, 2):
        for backend in backends:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = None
            for _ in range(8):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
            if frame is not None:
                print(f"barn camera: opened index {index}")
                return capture
            capture.release()
    print("barn camera: no camera opened")
    return None


def _open_capture(source: str):
    if str(source) in {"0", "webcam"}:
        capture = _open_webcam()
        return capture, capture is None

    if source and Path(source).is_file():
        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            return capture, False
        capture.release()
    return None, True


def _synthetic_frame(tick: float, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (42, 68, 48)
    cv2.rectangle(frame, (0, int(height * 0.62)), (width, height), (36, 48, 58), -1)
    cv2.rectangle(frame, (0, 0), (width, 70), (28, 32, 36), -1)

    sway = int(18 * np.sin(tick * 0.9))
    standing = (210 + sway, 210, 160, 210)
    cv2.ellipse(frame, (standing[0] + 80, standing[1] + 70), (80, 48), 0, 0, 360, (46, 78, 128), -1)
    cv2.circle(frame, (standing[0] + 150, standing[1] + 40), 28, (40, 70, 118), -1)

    lie_progress = min(1.0, max(0.0, (tick - 4.0) / 3.0))
    cow_w = int(150 + 70 * lie_progress)
    cow_h = int(110 - 50 * lie_progress)
    x = 560
    y = 280 + int(40 * lie_progress)
    cv2.ellipse(frame, (x + cow_w // 2, y + cow_h // 2), (cow_w // 2, cow_h // 2), 0, 0, 360, (38, 62, 112), -1)
    cv2.circle(frame, (x + cow_w - 18, y + 22), 22, (34, 56, 104), -1)
    return frame


def _synthetic_cows(tick: float):
    standing = {
        "xyxy": [210, 210, 370, 420],
        "conf": 0.91,
        "lying": False,
    }
    lie_progress = min(1.0, max(0.0, (tick - 4.0) / 3.0))
    cow_w = 150 + 70 * lie_progress
    cow_h = 110 - 50 * lie_progress
    x, y = 560.0, 280.0 + 40.0 * lie_progress
    lying = (cow_w / cow_h) > LYING_ASPECT_RATIO
    downed = {
        "xyxy": [x, y, x + cow_w, y + cow_h],
        "conf": 0.88,
        "lying": lying,
    }
    return [standing, downed]


def _detect_cows(model, frame):
    detections = []
    if model is None:
        return detections

    results = model(frame, classes=[COW_CLASS_ID], verbose=False, imgsz=320)
    if not results:
        return detections

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        if cls_id != COW_CLASS_ID:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        ratio = width / height
        if ratio >= LYING_ASPECT_RATIO:
            posture = "lying"
        elif ratio <= STANDING_ASPECT_RATIO:
            posture = "standing"
        else:
            posture = "unsure"
        detections.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "conf": float(box.conf[0]) if box.conf is not None else 0.0,
                "posture": posture,
                "lying": posture == "lying",
            }
        )
    return detections


def _frame_posture(detections) -> str:
    if not detections:
        return "none"
    votes = [det["posture"] for det in detections if det.get("posture") in {"standing", "lying"}]
    if not votes:
        return "unsure"
    if votes.count("lying") > votes.count("standing"):
        return "lying"
    return "standing"


def _box_motion(previous_gray, gray, box) -> float:
    if previous_gray is None or previous_gray.shape != gray.shape:
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    diff = cv2.absdiff(previous_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2])
    return float((diff > 18).mean())


def _parse_health_json(raw_text: str) -> dict | None:
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    data = json.loads(match.group(0))
    health = str(data.get("health", "")).strip().lower()
    if health not in {"well", "unwell", "unsure"}:
        return None
    reason = str(data.get("reason") or "").strip()
    return {"health": health, "reason": reason}


def _ask_cow_health(frame) -> dict | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, max(1, int(height * 640 / width))))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    response = requests.post(
        GROQ_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "max_tokens": 180,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": COW_HEALTH_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    ],
                }
            ],
        },
        timeout=20,
    )
    response.raise_for_status()
    raw_text = response.json()["choices"][0]["message"]["content"]
    return _parse_health_json(raw_text)


def _pick_arabic_font(size: int = 28):
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _visual_rtl(text: str) -> str:
    return text[::-1]


def _wrap_words(text: str, limit: int = 34) -> list[str]:
    lines = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if len(trial) <= limit:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _draw_banner(frame, text: str):
    banner = frame.copy()
    lines = _wrap_words(text, 34)
    banner_h = 20 + 32 * len(lines)
    cv2.rectangle(banner, (0, 0), (banner.shape[1], banner_h), (0, 0, 220), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            banner,
            line,
            (16, 32 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return banner


def _annotate(frame, detections, health: str):
    annotated = frame.copy()
    if health == "unwell":
        color = (0, 0, 239)
        label = "May be sick"
    elif health == "resting":
        color = (0, 165, 255)
        label = "Resting"
    elif health == "well":
        color = (16, 185, 129)
        label = "Not sick"
    else:
        color = (226, 232, 240)
        label = "Checking"
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if health == "unwell":
        annotated = _draw_banner(annotated, BANNER_TEXT)
    else:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 46), (15, 23, 42), -1)
        cv2.putText(
            annotated,
            "Haran Barn Guardian",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (16, 185, 129),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _notice_frame(text: str, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (30, 41, 59)
    cv2.putText(frame, "Haran Barn Guardian", (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (16, 185, 129), 2, cv2.LINE_AA)
    y = 180
    for line in _wrap_words(text, 42):
        cv2.putText(
            frame,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 42
    return frame


_clients = []
_clients_lock = threading.Lock()
_broadcast_lock = threading.Lock()
_broadcast_stop = threading.Event()
_broadcast_thread = None


def _publish(payload):
    with _clients_lock:
        for client in list(_clients):
            if client.full():
                try:
                    client.get_nowait()
                except queue.Empty:
                    pass
            try:
                client.put_nowait(payload)
            except queue.Full:
                pass


def generate_frames(source: str = "webcam"):
    global _broadcast_thread
    client = queue.Queue(maxsize=1)
    with _clients_lock:
        _clients.append(client)
    with _broadcast_lock:
        if _broadcast_thread is None or not _broadcast_thread.is_alive():
            _broadcast_stop.clear()
            _broadcast_thread = threading.Thread(
                target=_broadcast_loop,
                args=(source,),
                daemon=True,
            )
            _broadcast_thread.start()
    try:
        while True:
            chunk = client.get()
            if chunk is None:
                break
            yield chunk
    finally:
        with _clients_lock:
            if client in _clients:
                _clients.remove(client)
            still_watching = bool(_clients)
        if not still_watching:
            _broadcast_stop.set()


def _broadcast_loop(source: str = "webcam"):
    waiting = _notice_frame("Starting the barn camera.")
    ok_wait, wait_buffer = cv2.imencode(".jpg", waiting, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok_wait:
        _publish(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + wait_buffer.tobytes() + b"\r\n")
    resolved = source
    if str(source) not in {"0", "webcam"}:
        candidate = Path(source)
        if not candidate.is_file():
            alt = BASE_DIR / source
            resolved = str(alt) if alt.is_file() else "webcam"

    model = _load_yolo()
    if model is None:
        print("barn camera: cow detector failed to load")
    else:
        print("barn camera: cow detector ready")
    capture, missing = _open_capture(resolved)
    live = capture is not None and str(resolved) in {"0", "webcam"}
    frames = queue.Queue(maxsize=1)
    stop_reader = threading.Event()

    def _pump():
        while not stop_reader.is_set():
            current = capture
            if current is None:
                time.sleep(0.1)
                continue
            ok_read, pumped = current.read()
            if not ok_read or pumped is None:
                time.sleep(0.05)
                continue
            if frames.full():
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
            frames.put(pumped)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    posture_votes = []
    stable_posture = "checking"
    previous_gray = None
    still_since = None
    health_box = {"health": "", "reason": "", "at": 0.0, "busy": False}
    _set_status(
        camera="missing" if missing else "on",
        cows=0,
        lying=0,
        posture="checking",
        distressed=False,
        dark=False,
    )

    try:
        while not _broadcast_stop.is_set():
            with _clients_lock:
                if _broadcast_stop.is_set() and not _clients:
                    break
            now = time.monotonic()

            notice = False
            if missing or capture is None:
                frame = _notice_frame("No camera found. Plug in a camera and open Barn watch again.")
                detections = []
                dark = False
                notice = True
            else:
                try:
                    frame = frames.get(timeout=1.0)
                except queue.Empty:
                    frame = None
                if frame is None and live:
                    print("barn camera: no frame, reopening")
                    previous = capture
                    capture = _open_webcam()
                    if previous is not None and previous is not capture:
                        previous.release()
                    if capture is None:
                        missing = True
                        frame = _notice_frame("No camera found. Plug in a camera and open Barn watch again.")
                        detections = []
                        dark = False
                        notice = True
                if frame is None and not notice:
                    frame = _notice_frame("Waiting for the camera.")
                    detections = []
                    dark = False
                    notice = True
                elif not notice:
                    detections = _detect_cows(model, frame)
                    dark = float(frame.mean()) < 12.0

            seen = _frame_posture(detections)
            if seen in {"standing", "lying"}:
                posture_votes.append(seen)
                posture_votes = posture_votes[-6:]
                if len(posture_votes) >= 4 and all(vote == posture_votes[-1] for vote in posture_votes[-4:]):
                    stable_posture = posture_votes[-1]
            elif seen == "none":
                posture_votes.clear()
                stable_posture = "none"

            motion = 0.0
            if frame is not None and not notice and detections:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if previous_gray is not None:
                    motion = max(_box_motion(previous_gray, gray, det["xyxy"]) for det in detections)
                previous_gray = gray
            elif not detections:
                previous_gray = None

            if not detections or notice:
                still_since = None
                health = "none"
                reason = ""
            else:
                if motion >= 0.03:
                    still_since = None
                    behavior = "well"
                    behavior_reason = "The cow is moving, so she does not look sick."
                elif stable_posture == "lying":
                    still_since = None
                    behavior = "resting"
                    behavior_reason = "The cow is lying down to rest. That is normal, not sickness."
                else:
                    if still_since is None:
                        still_since = now
                    if now - still_since >= STILL_SECONDS:
                        behavior = "unwell"
                        behavior_reason = "The cow has stayed still. Go and look at her."
                    else:
                        behavior = "well"
                        behavior_reason = "The cow is in view and is not showing a clear problem."

                fresh_vision = health_box["at"] and (now - health_box["at"]) < 45
                if fresh_vision and health_box["health"] == "unwell":
                    health = "unwell"
                    reason = health_box["reason"] or "The photo shows a problem. Go and look at this cow."
                elif fresh_vision and health_box["health"] == "well":
                    health = "well"
                    reason = health_box["reason"] or behavior_reason
                else:
                    health = behavior
                    reason = behavior_reason

                if not dark and not health_box["busy"] and (now - health_box["at"]) >= HEALTH_CHECK_SECONDS:
                    health_box["busy"] = True
                    snapshot = frame.copy()

                    def _run(shot=snapshot):
                        try:
                            result = _ask_cow_health(shot)
                            if result:
                                health_box["health"] = result["health"]
                                health_box["reason"] = result["reason"]
                                print("cow health:", result["health"])
                        except Exception as error:
                            print("cow health failed", error)
                            if "429" in str(error):
                                health_box["at"] = time.monotonic() + 45
                        finally:
                            if health_box["at"] <= time.monotonic():
                                health_box["at"] = time.monotonic()
                            health_box["busy"] = False

                    threading.Thread(target=_run, daemon=True).start()

            _set_status(
                camera="missing" if missing else "on",
                cows=len(detections),
                lying=0,
                posture=stable_posture,
                health=health,
                reason=reason,
                distressed=health == "unwell",
                dark=dark,
            )

            annotated = _annotate(frame, detections, health)
            if dark and health != "unwell" and not notice:
                annotated = _draw_banner(
                    annotated,
                    "Open the camera cover and point it at the cow.",
                )
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            _publish(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.04)
    finally:
        stop_reader.set()
        _publish(None)
        if capture is not None:
            capture.release()
