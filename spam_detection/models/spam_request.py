from dataclasses import dataclass

@dataclass
class SpamRequest:
    """Request object for spam detection analysis.
    
    Attributes:
        text: The text content to analyze for spam.
        source: Optional source identifier for the text.
    """
    text: str
    source: str | None = None
