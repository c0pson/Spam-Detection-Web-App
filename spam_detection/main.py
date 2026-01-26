from flask import Flask, render_template
from spam_detection.api.routes_spam import spam_bp

from spam_detection.utils.http import ServerCommunication

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        if ServerCommunication.check_availability():
            return render_template("home.html")
        else:
            return render_template("404.html")
    app.register_blueprint(spam_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
