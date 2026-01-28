"""API routes module.

Provides Flask blueprints and route handlers for the spam detection web service.
"""

from spam_detection.api.routes_spam import spam_bp

__all__ = [
    "spam_bp",
]
