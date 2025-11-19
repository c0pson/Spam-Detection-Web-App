"""Flask web app 
"""
from flask import Blueprint, render_template
from flask import Flask

print(__file__)

spam_bp = Blueprint(
    "spam",
    __name__,
    url_prefix="/api/spam",
    
)

@spam_bp.route("/form")
def show_form():
    return render_template("index.html")
