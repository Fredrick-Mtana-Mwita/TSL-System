"""
Singleton model service: loads the TSLTransformer checkpoint once and
exposes a predict() method used by ai_service.
"""

from pathlib import Path

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "tsl_transformer.pt"

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


class ModelService:
    _instance: "ModelService | None" = None

    def __init__(self) -> None:
        self.model = None
        self.device = None

    @classmethod
    def get(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if not TORCH_AVAILABLE:
            print("PyTorch not installed — AI inference disabled, using rule-based fallback")
            return

        import torch
        from models.transformer_model import TSLTransformer, NUM_CLASSES

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TSLTransformer(num_classes=NUM_CLASSES).to(self.device)

        if CHECKPOINT_PATH.exists():
            state = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"Model loaded from {CHECKPOINT_PATH} on {self.device}")
        else:
            print(
                f"No checkpoint at {CHECKPOINT_PATH} — model untrained.\n"
                "Run: python train.py   (takes ~1 min on CPU with synthetic data)"
            )
        self.model.eval()

    def predict(self, sequence_tensor) -> tuple[int, float, dict]:
        """
        Returns (class_idx, confidence, label_dict).
        Falls back to (11, 0.0, Unknown) when torch is unavailable or model unloaded.
        """
        from models.transformer_model import GESTURE_LABELS

        if self.model is None or not TORCH_AVAILABLE:
            return 11, 0.0, GESTURE_LABELS[11]

        import torch
        with torch.no_grad():
            x = sequence_tensor.unsqueeze(0).to(self.device)  # (1, 30, 126)
            logits = self.model(x)                             # (1, 12)
            probs = torch.softmax(logits, dim=-1)[0]
            cls_idx = int(probs.argmax())
            confidence = float(probs[cls_idx])

        return cls_idx, confidence, GESTURE_LABELS[cls_idx]
