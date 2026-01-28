from spam_detection.main import create_app
from spam_detection.core.logger import Logger

logger = Logger.get_logger(__name__)

app = create_app()

if __name__ == "__main__":
    try:
        logger.info("Launching application from __main__.py")
        app.run(debug=True)
    except Exception as e:
        logger.error("Application error", exc_info=True)
        raise
