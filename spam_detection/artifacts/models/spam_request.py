from __future__ import annotations

from pydantic import BaseModel, Field
from spam_detection.core.config import DEFAULT_TOP_K

class SpamRequest(BaseModel):
    text: str = Field(..., min_length=1)
    explain: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)