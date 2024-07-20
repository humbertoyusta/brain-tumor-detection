import os
from flask import Flask, render_template
from app.api import api_blueprint

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(api_blueprint, url_prefix="/api/v1")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
