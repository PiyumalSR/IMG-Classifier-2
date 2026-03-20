import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import keras

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model.weights.h5")

# Build model from architecture (config.json), then load weights
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

model = keras.models.model_from_json(json.dumps(config))
model.load_weights(WEIGHTS_PATH)

# ── Class labels (CIFAR-10) ──────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Image classifier API is running."})

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send an image with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((32, 32))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]

        probabilities = {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
