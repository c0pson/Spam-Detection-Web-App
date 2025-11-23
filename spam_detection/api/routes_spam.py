from flask import Blueprint, render_template, request, jsonify

from spam_detection.services.spam_checker import SpamService

spam_bp = Blueprint(
    "spam",
    __name__,
    url_prefix="/api/spam",
)

@spam_bp.route("/form")
def show_form():
    return render_template("check_spam.html")


@spam_bp.route("/check", methods=["POST"])
def check():
    data = request.get_json()
    text = data.get("text", "")
    is_spam = SpamService.classify_text(text)
    return jsonify({"result": is_spam})
