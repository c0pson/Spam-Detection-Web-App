from __future__ import annotations

from pydantic import BaseModel, Field

class SpamRequest(BaseModel):
    text: str = Field(..., min_length=1)
    explain: bool = False
    top_k: int = Field(default=12, ge=1, le=50)