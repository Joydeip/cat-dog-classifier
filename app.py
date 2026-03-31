import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import io

model = load_model("my_cnn_model(cat-dog).h5")

def predict(image, url):
    try:
        if url and url.strip() != "":
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url.strip(), timeout=10, headers=headers)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        elif image is not None:
            image = image.convert("RGB")
        else:
            return "⚠️ Please upload an image or enter a URL!"

        img = image.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        return {label: round(confidence, 2)}

    except Exception as e:
        return f"❌ Error: {str(e)}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="📁 Upload Image"),
        gr.Textbox(label="🔗 Or Paste Image URL", placeholder="https://example.com/cat.jpg")
    ],
    outputs=gr.Label(label="Prediction"),
    title="🐾 Cat or Dog Classifier",
    description="Upload an image or paste a URL to predict whether it's a cat or a dog!",
    examples=[
        ["sample_cat.jpg", ""],
        ["sample_dog.jpg", ""],
        ["sample_cat2.jpg", ""],
        ["sample_dog2.jpg", ""],
    ],
    theme=gr.themes.Soft()
)

demo.launch(server_name="0.0.0.0", server_port=7860)
