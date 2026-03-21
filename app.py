import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import keras

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load model once at startup ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model.weights.h5")

log.info("Loading model config from: %s", CONFIG_PATH)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
log.info("Config loaded successfully.")

def remove_unsupported_keys(obj):
    unsupported = {"quantization_config", "dtype_policy", "shared_object_id"}
    if isinstance(obj, dict):
        return {k: remove_unsupported_keys(v) for k, v in obj.items() if k not in unsupported}
    elif isinstance(obj, list):
        return [remove_unsupported_keys(i) for i in obj]
    return obj

log.info("Cleaning config...")
clean_config = remove_unsupported_keys(config)

log.info("Building model from config...")
model = keras.models.model_from_json(json.dumps(clean_config))

log.info("Loading weights from: %s", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH)
log.info("✅ Model ready! Input shape: %s", model.input_shape)

# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── CORS headers ──────────────────────────────────────────────────────────────
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    log.info("Health check hit.")
    return jsonify({"status": "ok", "message": "Image classifier API is running."})

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        log.info("OPTIONS preflight request received.")
        return jsonify({}), 200

    log.info("POST /predict received.")

    if "file" not in request.files:
        log.warning("No file in request.")
        return jsonify({"error": "No file provided. Send an image with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        log.warning("Empty filename.")
        return jsonify({"error": "Empty filename."}), 400

    log.info("File received: %s", file.filename)

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        log.info("Image opened. Original size: %s", img.size)

        img = img.resize((32, 32))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        log.info("Image preprocessed. Array shape: %s", img_array.shape)

        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        log.info("Prediction: %s (confidence: %.2f%%)", predicted_class, confidence * 100)

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
        log.error("Error during prediction: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Flask on port %d", port)
    app.run(host="0.0.0.0", port=port)
