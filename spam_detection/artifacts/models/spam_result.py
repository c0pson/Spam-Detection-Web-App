from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

class SpamResult(BaseModel):
    result: str = Field(..., pattern="^(Spam|Ham)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    spam_probability: float = Field(..., ge=0.0, le=1.0)
    keywords: Optional[List[Tuple[str, float]]] = None