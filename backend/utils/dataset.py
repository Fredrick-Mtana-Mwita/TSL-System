"""
Dataset infrastructure for TSL Transformer training.

Quick start:
    from utils.dataset import generate_synthetic_dataset
    generate_synthetic_dataset()   # writes data/ split files
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURE_DIM     = 126
SEQUENCE_LENGTH = 30
NUM_CLASSES     = 12

TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    pass


if TORCH_AVAILABLE:
    class TSLDataset(Dataset):
        """PyTorch dataset that loads pre-built .npy split files."""

        def __init__(self, split: str = "train"):
            path       = DATA_DIR / f"{split}.npy"
            label_path = DATA_DIR / f"{split}_labels.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {path}.\n"
                    "Run: python -c \"from utils.dataset import generate_synthetic_dataset; generate_synthetic_dataset()\""
                )
            self.X = np.load(path).astype(np.float32)
            self.y = np.load(label_path).astype(np.int64)

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def _build_template(thumb_left: bool, raised: list[bool]) -> np.ndarray:
    """
    Construct a rough (21, 3) hand landmark template from finger states.
    Used only for synthetic data generation.
    """
    lms = np.zeros((21, 3), dtype=np.float32)
    # Finger groups: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    groups = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for fi, (indices, up) in enumerate(zip(groups, raised)):
        for joint_rank, lm_idx in enumerate(indices, start=1):
            lms[lm_idx, 0] = (fi - 2) * 0.2          # spread fingers horizontally
            lms[lm_idx, 1] = joint_rank * 0.25 * (1.0 if up else 0.3)
    lms[4, 0] = -0.6 if thumb_left else -0.2          # thumb tip
    return lms


# One template per class index (matches GESTURE_LABELS order)
_TEMPLATES = [
    _build_template(True,  [True,  True,  True,  True,  True]),   # 0  Hello
    _build_template(False, [False, False, False, False, False]),   # 1  Stop
    _build_template(False, [False, True,  False, False, False]),   # 2  One
    _build_template(False, [False, True,  True,  False, False]),   # 3  Peace
    _build_template(False, [False, True,  True,  True,  False]),   # 4  Three
    _build_template(False, [False, True,  True,  True,  True]),    # 5  Four
    _build_template(True,  [True,  False, False, False, False]),   # 6  Good
    _build_template(True,  [True,  True,  False, False, True]),    # 7  I Love You
    _build_template(False, [False, False, False, False, True]),    # 8  Promise
    _build_template(True,  [True,  True,  False, False, False]),   # 9  L-Shape
    _build_template(False, [False, True,  False, False, True]),    # 10 Rock On
    _build_template(False, [False, False, False, False, False]),   # 11 Unknown
]


def generate_synthetic_dataset(n_per_class: int = 300, noise: float = 0.025) -> None:
    """
    Generate synthetic landmark sequences and write train/val .npy files.
    Each sequence has slight Gaussian noise to simulate real variation.
    """
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    all_X, all_y = [], []

    for cls_idx, template in enumerate(_TEMPLATES):
        for _ in range(n_per_class):
            sequence = []
            for _ in range(SEQUENCE_LENGTH):
                noisy = template + rng.normal(0, noise, template.shape).astype(np.float32)
                centered = noisy - noisy[0]
                scale = np.max(np.abs(centered)) + 1e-8
                frame = np.zeros(FEATURE_DIM, dtype=np.float32)
                frame[:63] = (centered / scale).flatten()   # first hand; second hand zeros
                sequence.append(frame)
            all_X.append(np.array(sequence, dtype=np.float32))
            all_y.append(cls_idx)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "train.npy",        X[:split])
    np.save(DATA_DIR / "train_labels.npy", y[:split])
    np.save(DATA_DIR / "val.npy",          X[split:])
    np.save(DATA_DIR / "val_labels.npy",   y[split:])
    print(f"Synthetic dataset: {split} train / {len(X) - split} val samples saved to {DATA_DIR}")
