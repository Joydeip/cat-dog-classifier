import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# ── Flask Backend ──
flask_app = Flask(__name__)
CORS(flask_app)

model = load_model("my_cnn_model(cat-dog).h5")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@flask_app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_array = preprocess_image(file.read())
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    return jsonify({"label": label, "confidence": round(confidence * 100, 2)})

@flask_app.route("/predict-url", methods=["POST"])
def predict_url():
    data = request.get_json()
    url = data.get("url", "")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        img_array = preprocess_image(response.content)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    return jsonify({"label": label, "confidence": round(confidence * 100, 2)})

@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "awake"}), 200

# ── Run Flask in background thread ──
def run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

# ── Gradio just serves the HTML file ──
with gr.Blocks() as demo:
    gr.HTML(open("index.html").read())

demo.launch(server_name="0.0.0.0", server_port=7860)