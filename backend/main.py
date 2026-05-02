import base64
import io
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# MediaPipe Tasks API (mediapipe >= 0.10.x)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MEDIAPIPE_AVAILABLE = False
hand_landmarker = None

try:
    import cv2
    from PIL import Image
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not MODEL_PATH.exists():
        print("Downloading MediaPipe hand landmarker model (~25 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

    _base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    _opts = mp_vision.HandLandmarkerOptions(
        base_options=_base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(_opts)
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe ready (Tasks API).")
except Exception as _err:
    print(f"MediaPipe not available: {_err}")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="TSL System API", description="Tanzania Sign Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class FrameRequest(BaseModel):
    image: str  # base64 data-URL or raw base64 JPEG


class HandLandmark(BaseModel):
    x: float
    y: float
    z: float


class DetectionResult(BaseModel):
    detected: bool
    gesture: Optional[str]
    confidence: float
    landmarks: list[list[HandLandmark]]
    num_hands: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decode_image(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def get_finger_states(lms: list) -> list[bool]:
    """lms is a flat list of NormalizedLandmark with .x .y .z"""
    thumb_up = lms[4].x < lms[3].x
    tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
    fingers = [thumb_up]
    for tip, pip in tips_pips:
        fingers.append(lms[tip].y < lms[pip].y)
    return fingers  # [thumb, index, middle, ring, pinky]


def classify_gesture(lms: list) -> tuple[str, float]:
    t, i, m, r, p = get_finger_states(lms)

    if all([t, i, m, r, p]):
        return "Open Hand — Habari / Hello", 0.88
    if not any([t, i, m, r, p]):
        return "Fist — Simama / Stop", 0.88
    if i and not m and not r and not p:
        return "One / Pointing", 0.82
    if i and m and not r and not p:
        return "Two / Peace / V", 0.82
    if i and m and r and not p:
        return "Three", 0.82
    if i and m and r and p and not t:
        return "Four", 0.82
    if t and not i and not m and not r and not p:
        return "Thumbs Up — Nzuri / Good", 0.84
    if t and not i and not m and not r and p:
        return "I Love You — Nakupenda", 0.85
    if not t and not i and not m and not r and p:
        return "Pinky — Ahadi / Promise", 0.78
    if t and i and not m and not r and not p:
        return "L-Shape / Pistol", 0.75
    if not t and i and not m and not r and p:
        return "Rock On", 0.78
    return "Unknown Gesture", 0.40


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "TSL System API", "status": "running", "mediapipe": MEDIAPIPE_AVAILABLE}


@app.get("/health")
def health_check():
    return {"status": "healthy", "mediapipe": MEDIAPIPE_AVAILABLE}


@app.post("/detect", response_model=DetectionResult)
async def detect_gesture(frame: FrameRequest):
    if not MEDIAPIPE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="MediaPipe not ready. Check server logs.",
        )
    try:
        bgr = decode_image(frame.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return DetectionResult(
                detected=False, gesture=None, confidence=0.0, landmarks=[], num_hands=0
            )

        all_landmarks = [
            [HandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand]
            for hand in result.hand_landmarks
        ]
        gesture, confidence = classify_gesture(result.hand_landmarks[0])

        return DetectionResult(
            detected=True,
            gesture=gesture,
            confidence=confidence,
            landmarks=all_landmarks,
            num_hands=len(result.hand_landmarks),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
