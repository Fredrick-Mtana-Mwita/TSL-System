from typing import Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    image: str   # base64 data-URL or raw base64 JPEG


class PredictionResponse(BaseModel):
    detected:    bool
    english:     Optional[str]  = None
    swahili:     Optional[str]  = None
    gesture:     Optional[str]  = None   # combined "English / Swahili" for backward compat
    confidence:  float          = 0.0
    gesture_id:  Optional[int]  = None
    landmarks:   list           = []
    num_hands:   int            = 0
