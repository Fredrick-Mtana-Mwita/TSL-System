"""
AI prediction routes:
  POST /predict        — single-frame HTTP inference
  WS   /ws/detect      — WebSocket streaming inference (one frame per message)
"""

import uuid
from collections import defaultdict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from schemas.prediction_schema import PredictionRequest, PredictionResponse
from services.mediapipe_service import MEDIAPIPE_AVAILABLE, detect_hands
from services.ai_service import predict_from_landmarks, clear_session

router = APIRouter()

# Per-session rolling frame buffers (max 30 frames)
_MAX_BUFFER = 30
_session_bufs: dict[str, list] = defaultdict(list)


def _run_detection(b64_image: str, session_id: str) -> PredictionResponse:
    """Shared logic for HTTP and WebSocket handlers."""
    if not MEDIAPIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="MediaPipe not ready")

    result = detect_hands(b64_image)
    if result is None or not result.hand_landmarks:
        return PredictionResponse(detected=False)

    buf = _session_bufs[session_id]
    prediction = predict_from_landmarks(result.hand_landmarks, buf, session_id=session_id)
    if len(buf) > _MAX_BUFFER:
        del buf[: len(buf) - _MAX_BUFFER]

    landmarks = [
        [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
        for hand in result.hand_landmarks
    ]
    return PredictionResponse(
        detected=True,
        english=prediction["english"],
        swahili=prediction["swahili"],
        gesture=f"{prediction['english']} / {prediction['swahili']}",
        confidence=prediction["confidence"],
        gesture_id=prediction["gesture_id"],
        landmarks=landmarks,
        num_hands=len(result.hand_landmarks),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return _run_detection(request.image, session_id="http-default")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_json()
            b64 = data.get("image", "")
            if not b64:
                continue
            try:
                resp = _run_detection(b64, session_id)
                await websocket.send_json(resp.model_dump())
            except HTTPException as exc:
                await websocket.send_json({"error": exc.detail, "detected": False, "confidence": 0.0})
            except Exception as exc:
                await websocket.send_json({"error": str(exc), "detected": False, "confidence": 0.0})
    except WebSocketDisconnect:
        _session_bufs.pop(session_id, None)
        clear_session(session_id)
