"""
AI inference service:
  - Wraps ModelService with per-session frame buffers
  - Applies majority-vote prediction smoothing
  - Falls back to rule-based classification when torch is unavailable
"""

from collections import deque
import numpy as np

from utils.preprocessing import landmarks_to_feature, pad_sequence, SEQUENCE_LENGTH

SMOOTH_WINDOW = 5

_RULE_LABELS: dict[int, dict[str, str]] = {
    0:  {"english": "Hello",      "swahili": "Habari"},
    1:  {"english": "Stop",       "swahili": "Simama"},
    2:  {"english": "One",        "swahili": "Moja"},
    3:  {"english": "Peace",      "swahili": "Amani"},
    4:  {"english": "Three",      "swahili": "Tatu"},
    5:  {"english": "Four",       "swahili": "Nne"},
    6:  {"english": "Good",       "swahili": "Nzuri"},
    7:  {"english": "I Love You", "swahili": "Nakupenda"},
    8:  {"english": "Promise",    "swahili": "Ahadi"},
    9:  {"english": "L-Shape",    "swahili": "Umbo-L"},
    10: {"english": "Rock On",    "swahili": "Pumzika"},
    11: {"english": "Unknown",    "swahili": "Haijulikani"},
}


def _rule_based_classify(hand_landmarks: list) -> tuple[int, float]:
    lms = hand_landmarks
    thumb_up = lms[4].x < lms[3].x
    fingers = [thumb_up] + [lms[tip].y < lms[pip].y for tip, pip in [(8,6),(12,10),(16,14),(20,18)]]
    t, i, m, r, p = fingers

    if all(fingers):                              return 0, 0.88
    if not any(fingers):                          return 1, 0.88
    if i and not m and not r and not p:           return 2, 0.82
    if i and m and not r and not p:               return 3, 0.82
    if i and m and r and not p:                   return 4, 0.82
    if i and m and r and p and not t:             return 5, 0.82
    if t and not i and not m and not r and not p: return 6, 0.84
    if t and not i and not m and not r and p:     return 7, 0.85
    if not t and not i and not m and not r and p: return 8, 0.78
    if t and i and not m and not r and not p:     return 9, 0.75
    if not t and i and not m and not r and p:     return 10, 0.78
    return 11, 0.40


class _Smoother:
    def __init__(self, window: int = SMOOTH_WINDOW):
        self._buf: deque[int] = deque(maxlen=window)

    def push(self, cls: int) -> int:
        self._buf.append(cls)
        return max(set(self._buf), key=lambda c: list(self._buf).count(c))

    def clear(self) -> None:
        self._buf.clear()


_smoothers: dict[str, _Smoother] = {}


def _get_smoother(session_id: str) -> _Smoother:
    if session_id not in _smoothers:
        _smoothers[session_id] = _Smoother()
    return _smoothers[session_id]


def clear_session(session_id: str) -> None:
    _smoothers.pop(session_id, None)


def predict_from_landmarks(
    hand_landmarks_list: list,
    frame_buffer: list,
    session_id: str = "default",
) -> dict:
    features = landmarks_to_feature(hand_landmarks_list)
    frame_buffer.append(features)
    return predict_sequence(frame_buffer, session_id=session_id)


def predict_sequence(frames: list, session_id: str = "default") -> dict:
    """
    Run transformer inference (with rule-based fallback) on a frame buffer.
    frames: list of (126,) np.ndarray feature vectors.
    """
    if not frames:
        labels = _RULE_LABELS[11]
        return {**labels, "confidence": 0.0, "gesture_id": 11}

    cls_idx, confidence, labels = 11, 0.4, _RULE_LABELS[11]

    try:
        import torch
        from services.model_service import ModelService

        seq = pad_sequence(frames, SEQUENCE_LENGTH)
        tensor = torch.from_numpy(seq)
        ai_cls, ai_conf, ai_labels = ModelService.get().predict(tensor)

        # Only accept AI result when confidence is meaningful
        if ai_conf >= 0.25:
            cls_idx, confidence, labels = ai_cls, ai_conf, ai_labels

    except Exception as exc:
        print(f"[ai_service] AI inference failed, using rule-based fallback: {exc}")

    smoothed = _get_smoother(session_id).push(cls_idx)
    final_labels = labels if smoothed == cls_idx else _RULE_LABELS.get(smoothed, _RULE_LABELS[11])

    return {
        "english":    final_labels["english"],
        "swahili":    final_labels["swahili"],
        "confidence": confidence,
        "gesture_id": smoothed,
    }
