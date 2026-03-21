import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Load model ────────────────────────────────────────────────────────────────
import tensorflow as tf

BASE_DIR     = os.path.dirname(__file__)
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model.weights.h5")

# Limit TensorFlow memory usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limit CPU threads to save RAM
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

log.info("Loading config...")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def remove_unsupported_keys(obj):
    unsupported = {"quantization_config", "dtype_policy", "shared_object_id", "optional"}
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in unsupported:
                continue
            # rename batch_shape -> shape for older Keras InputLayer
            if k == "batch_shape":
                # convert [None, 32, 32, 3] -> [32, 32, 3] (remove batch dim)
                cleaned["shape"] = v[1:] if isinstance(v, list) and len(v) > 1 else v
            else:
                cleaned[k] = remove_unsupported_keys(v)
        return cleaned
    elif isinstance(obj, list):
        return [remove_unsupported_keys(i) for i in obj]
    return obj

log.info("Building model...")
clean_config = remove_unsupported_keys(config)
model = tf.keras.models.model_from_json(json.dumps(clean_config))

log.info("Loading weights...")
model.load_weights(WEIGHTS_PATH)
log.info("✅ Model ready! Input shape: %s", model.input_shape)

# Warm up model with a dummy prediction to pre-allocate memory
dummy = np.zeros((1, 32, 32, 3), dtype="float32")
model.predict(dummy, verbose=0)
log.info("✅ Model warmed up.")

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
        return jsonify({}), 200

    log.info("POST /predict received.")

    if "file" not in request.files:
        log.warning("No file in request.")
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    log.info("File received: %s", file.filename)

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        log.info("Image opened. Size: %s", img.size)

        img = img.resize((32, 32))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        log.info("Preprocessed shape: %s", img_array.shape)

        predictions = model(img_array, training=False).numpy()  # lighter than model.predict()
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_index]
        log.info("Prediction: %s (%.2f%%)", predicted_class, confidence * 100)

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
        log.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Flask on port %d", port)
    app.run(host="0.0.0.0", port=port)
