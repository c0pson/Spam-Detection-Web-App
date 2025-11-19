from flask import Flask, render_template
from spam_detection.api.routes_spam import spam_bp

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("home.html")

    app.register_blueprint(spam_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
