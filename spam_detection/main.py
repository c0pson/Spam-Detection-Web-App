"""Main application factory and initialization.

This module creates and configures the Flask application, registers blueprints,
and sets up the home page route. It serves as the entry point for the spam
detection web service.
"""

from flask import Flask, render_template
from spam_detection.api.routes_spam import spam_bp
from spam_detection.core.logger import Logger

logger = Logger.get_logger(__name__)

def create_app():
    """Create and configure the Flask application.

    Initializes the Flask application, registers the spam detection blueprint,
    and sets up the home route. Application logging is configured through
    the logger module.

    Returns:
        Flask: Configured Flask application instance.

    Raises:
        Exception: If blueprint registration or configuration fails.
    """
    logger.info("Creating Flask application")
    app = Flask(__name__)

    @app.route("/")
    def home():
        """Render the home page.

        Returns:
            str: Rendered HTML for the home page.
        """
        logger.debug("Home page accessed")
        return render_template("home.html")

    app.register_blueprint(spam_bp)
    logger.info("Spam detection blueprint registered")
    return app

if __name__ == "__main__":
    try:
        logger.info("Starting spam detection application")
        app = create_app()
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        logger.error("Failed to start application", exc_info=True)
        raise
