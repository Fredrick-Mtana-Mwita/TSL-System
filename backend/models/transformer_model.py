import math
import torch
import torch.nn as nn

GESTURE_LABELS: dict[int, dict[str, str]] = {
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

NUM_CLASSES     = len(GESTURE_LABELS)
SEQUENCE_LENGTH = 30
FEATURE_DIM     = 126   # 2 hands × 21 landmarks × 3 coords


class TSLTransformer(nn.Module):
    """
    Transformer encoder for Tanzania Sign Language sequence classification.
    Input:  (batch, seq_len, feature_dim)
    Output: (batch, num_classes)

    Layer names match the Colab-trained checkpoint exactly:
      input_projection, pos_encoder (buffer), encoder, fc
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Pre-computed PE stored as a plain buffer (matches Colab checkpoint key "pos_encoder")
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoder", pe)   # (1, max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # fc matches Colab: Linear → ReLU → Dropout → Linear (indices 0, 1, 2, 3)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)                          # (B, T, d_model)
        t = min(x.size(1), self.pos_encoder.size(1))
        x = x[:, :t] + self.pos_encoder[:, :t]               # add PE, clamp to buffer size
        x = self.encoder(x)                                    # (B, T, d_model)
        x = x.mean(dim=1)                                      # global average pool over time
        return self.fc(x)                                      # (B, num_classes)
