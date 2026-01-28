"""Spam detection API routes.

This module provides Flask blueprint routes for the spam detection service,
including endpoints for checking spam messages and displaying the spam
detection form. It handles:

- GET/POST requests to the form page
- Text validation and retry logic for service failures
- Response rendering with spam predictions
- HTTP header optimization with resource preloading

Routes:
    /form: GET/POST endpoint for spam detection form and processing
"""

import time
from flask import Blueprint, render_template, request
from spam_detection.services.spam_checker import SpamService
from spam_detection.core.config import DEFAULT_TOP_K
from spam_detection.core.logger import Logger
from requests.exceptions import ConnectionError, Timeout

logger = Logger.get_logger(__name__)

spam_bp = Blueprint(
    "spam",
    __name__,
    static_folder="static",
    url_prefix="/",
)

@spam_bp.after_request
def push_resources(response):
    """Add preload headers for static resources.

    Optimizes page load performance by preloading CSS stylesheets and
    Material Icons in the HTTP headers.

    Args:
        response: Flask response object.

    Returns:
        Flask.Response: Modified response with preload headers.
    """
    if response.status_code == 200 and 'text/html' in response.content_type:
        response.headers.add(
            'Link', 
            '</static/css/styles.css>; rel=preload; as=style'
        )
        response.headers.add(
            'Link',
            '<https://fonts.googleapis.com/icon?family=Material+Icons>; rel=preload; as=style'
        )
    return response

@spam_bp.route("/form", methods=["GET", "POST"])
def show_form():
    """Handle spam detection form display and text classification.

    Processes form submissions by classifying text using the SpamService.
    Implements exponential backoff retry logic for transient service failures.
    Renders results or error pages based on classification outcome.

    For POST requests:
    - Extracts email body text from form
    - Retries classification up to 3 times on connection errors
    - Returns spam prediction with confidence and keywords if available
    - Falls back to 404 error page on persistent failures

    For GET requests:
    - Returns blank form for user input

    Returns:
        str: Rendered HTML template (check_spam.html or 404.html)

    Raises:
        None: All errors are caught and returned as 404 error pages.
    """
    if request.method == "POST":
        text = request.form.get("email_body", "")
        logger.info(f"Processing spam check request with text length: {len(text)}")
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                prediction = SpamService.classify_text(
                    text,
                    explain=True,
                    top_k=DEFAULT_TOP_K,
                )
                logger.info(f"Spam classification result: is_spam={prediction.is_spam}, confidence={prediction.confidence:.2f}")
                return render_template(
                    "check_spam.html",
                    show_result=True,
                    is_spam=prediction.is_spam,
                    user_text=text,
                    keywords=prediction.keywords if prediction.is_spam else [],
                )
            except (ConnectionError, ConnectionResetError, Timeout, RuntimeError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}")
                if attempt == max_retries - 1:
                    logger.error(f"Max retries exceeded after {max_retries} attempts", exc_info=True)
                    return render_template("404.html")
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unexpected error during spam classification", exc_info=True)
                return render_template("404.html")
    logger.debug("Form page accessed (GET request)")
    return render_template("check_spam.html", show_result=False)
