import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("my_cnn_model(cat-dog).h5")

def predict(image):
    img = image.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return {label: round(confidence, 2)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Cat or Dog Image"),
    outputs=gr.Label(label="Prediction"),
    title="🐾 Cat or Dog Classifier",
    description="Upload an image to predict whether it's a cat or a dog!",
    examples=[   # sample images shown on page
        ["sample_cat.jpg"],
        ["sample_dog.jpg"],
    ],
    theme=gr.themes.Soft()
)

demo.launch()