from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from app.inference import ModelExecutor

api_blueprint = Blueprint("api", __name__)

model_executor = ModelExecutor()


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 422

    image_file = request.files["image"]
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        result = model_executor.predict(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
