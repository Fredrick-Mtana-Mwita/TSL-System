from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MediaPipe initialises here — also consumed by routes/predict.py
from services.mediapipe_service import (
    MEDIAPIPE_AVAILABLE,
    decode_image,
    hand_landmarker,
)
from routes.predict import router as predict_router

app = FastAPI(
    title="TSL System API",
    description="Tanzania Sign Language Detection — AI-powered (Transformer) with bilingual output",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI routes: POST /predict  and  WS /ws/detect
app.include_router(predict_router)


# ---------------------------------------------------------------------------
# Pydantic models (kept for backward-compat /detect endpoint)
# ---------------------------------------------------------------------------
class FrameRequest(BaseModel):
    image: str


class HandLandmark(BaseModel):
    x: float
    y: float
    z: float


class DetectionResult(BaseModel):
    detected:   bool
    gesture:    Optional[str]
    english:    Optional[str]   = None
    swahili:    Optional[str]   = None
    confidence: float
    landmarks:  list[list[HandLandmark]]
    num_hands:  int


# ---------------------------------------------------------------------------
# Rule-based finger-state classifier (legacy fallback for /detect)
# ---------------------------------------------------------------------------
def _finger_states(lms: list) -> list[bool]:
    thumb_up = lms[4].x < lms[3].x
    return [thumb_up] + [lms[tip].y < lms[pip].y for tip, pip in [(8,6),(12,10),(16,14),(20,18)]]


def _classify_gesture(lms: list) -> tuple[str, float]:
    t, i, m, r, p = _finger_states(lms)
    if all([t, i, m, r, p]):                              return "Hello / Habari", 0.88
    if not any([t, i, m, r, p]):                          return "Stop / Simama", 0.88
    if i and not m and not r and not p:                   return "One / Moja", 0.82
    if i and m and not r and not p:                       return "Peace / Amani", 0.82
    if i and m and r and not p:                           return "Three / Tatu", 0.82
    if i and m and r and p and not t:                     return "Four / Nne", 0.82
    if t and not i and not m and not r and not p:         return "Good / Nzuri", 0.84
    if t and not i and not m and not r and p:             return "I Love You / Nakupenda", 0.85
    if not t and not i and not m and not r and p:         return "Promise / Ahadi", 0.78
    if t and i and not m and not r and not p:             return "L-Shape / Umbo-L", 0.75
    if not t and i and not m and not r and p:             return "Rock On / Pumzika", 0.78
    return "Unknown / Haijulikani", 0.40


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "message":   "TSL System API",
        "status":    "running",
        "mediapipe": MEDIAPIPE_AVAILABLE,
        "endpoints": {
            "legacy_http": "POST /detect",
            "ai_http":     "POST /predict",
            "ai_ws":       "WS   /ws/detect",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "mediapipe": MEDIAPIPE_AVAILABLE}


@app.post("/detect", response_model=DetectionResult)
async def detect_gesture(frame: FrameRequest):
    """
    Legacy endpoint — uses AI inference when torch is available,
    falls back to rule-based classification otherwise.
    Kept for backward compatibility with the existing frontend polling.
    """
    if not MEDIAPIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="MediaPipe not ready. Check server logs.")

    try:
        import cv2
        import mediapipe as mp

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

        # Try AI inference first
        try:
            from services.ai_service import predict_from_landmarks
            buf: list = []
            pred = predict_from_landmarks(result.hand_landmarks, buf, session_id="legacy")
            gesture   = f"{pred['english']} / {pred['swahili']}"
            confidence = pred["confidence"]
            english, swahili = pred["english"], pred["swahili"]
        except Exception:
            gesture, confidence = _classify_gesture(result.hand_landmarks[0])
            english = swahili = None

        return DetectionResult(
            detected=True,
            gesture=gesture,
            english=english,
            swahili=swahili,
            confidence=confidence,
            landmarks=all_landmarks,
            num_hands=len(result.hand_landmarks),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
