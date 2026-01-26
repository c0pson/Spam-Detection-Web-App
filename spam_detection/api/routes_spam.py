import time
from flask import Blueprint, render_template, request, make_response
from spam_detection.services.spam_checker import SpamService
from spam_detection.core.config import DEFAULT_TOP_K
from requests.exceptions import ConnectionError, Timeout, RequestException

spam_bp = Blueprint(
    "spam",
    __name__,
    static_folder="static",
    url_prefix="/",
)

@spam_bp.after_request
def push_resources(response):
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
    if request.method == "POST":
        text = request.form.get("email_body", "")
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                prediction = SpamService.classify_text(
                    text,
                    explain=True,
                    top_k=DEFAULT_TOP_K,
                )
                return render_template(
                    "check_spam.html",
                    show_result=True,
                    is_spam=prediction.is_spam,
                    user_text=text,
                    keywords=prediction.keywords if prediction.is_spam else [],
                )
            except (ConnectionError, ConnectionResetError, Timeout, RuntimeError) as e:
                if attempt == max_retries - 1:
                    return render_template("404.html")
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            except Exception as e:
                return render_template("404.html")
    return render_template("check_spam.html", show_result=False)
