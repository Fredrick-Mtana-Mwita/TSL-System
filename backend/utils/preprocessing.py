import numpy as np

NUM_LANDMARKS       = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND   = NUM_LANDMARKS * COORDS_PER_LANDMARK   # 63
MAX_HANDS           = 2
FEATURE_DIM         = MAX_HANDS * FEATURES_PER_HAND          # 126
SEQUENCE_LENGTH     = 30


def normalize_hand(landmarks: list) -> np.ndarray:
    """Center on wrist and scale to [-1, 1]. Returns (63,) float32."""
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    arr -= arr[0]                              # translate wrist to origin
    scale = np.max(np.abs(arr)) + 1e-8
    return (arr / scale).flatten()


def landmarks_to_feature(hand_landmarks_list: list) -> np.ndarray:
    """
    Convert up to 2 mediapipe hand landmark lists into a (126,) feature vector.
    Missing hands are zero-padded.
    """
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, hand in enumerate(hand_landmarks_list[:MAX_HANDS]):
        start = i * FEATURES_PER_HAND
        feat[start : start + FEATURES_PER_HAND] = normalize_hand(hand)
    return feat


def pad_sequence(frames: list, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Pad (front) or truncate (tail) a list of feature vectors to target_len."""
    arr = np.array(frames, dtype=np.float32)   # (T, 126)
    if len(arr) >= target_len:
        return arr[-target_len:]
    pad = np.zeros((target_len - len(arr), FEATURE_DIM), dtype=np.float32)
    return np.vstack([pad, arr])
