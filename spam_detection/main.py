from flask import Flask, render_template
from spam_detection.api.routes_spam import spam_bp
from spam_detection.core.logger import Logger

logger = Logger.get_logger(__name__)

def create_app():
    logger.info("Creating Flask application")
    app = Flask(__name__)

    @app.route("/")
    def home():
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
