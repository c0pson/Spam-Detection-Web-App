from dataclasses import dataclass

@dataclass
class SpamRequest:
    text: str
    source: str | None = None
