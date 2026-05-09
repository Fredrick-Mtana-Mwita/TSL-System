"""
MediaPipe service: initialises the hand landmarker once at import time
and exposes helpers used by both main.py and routes/predict.py.
"""

import base64
import io
import urllib.request
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "hand_landmarker.task"
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
    print("MediaPipe ready.")
except Exception as _err:
    print(f"MediaPipe not available: {_err}")


def decode_image(b64: str) -> "np.ndarray":
    """Decode a base64 data-URL or raw base64 JPEG → BGR np.ndarray."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def detect_hands(b64_image: str):
    """
    Run hand landmark detection on a base64 image.
    Returns mediapipe HandLandmarkerResult or None if MediaPipe is unavailable.
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    bgr = decode_image(b64_image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return hand_landmarker.detect(mp_image)


def extract_landmark(hand_landmark) -> list[float]:
    """Convert a single MediaPipe hand landmark list to a flat float list."""
    if not hand_landmark:
        return []
    flattened = []
    for lm in hand_landmark:
        flattened.extend([lm.x, lm.y, lm.z])
    return flattened
