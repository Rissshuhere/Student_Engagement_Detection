import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Student Engagement Detection")

st.title("Student Engagement Detection")
st.write("Upload a student image to predict engagement state.")

model = tf.keras.models.load_model("models/student_engagement_cnn.keras")

class_names = [
    "Engaged",
    "Confused",
    "Bored",
    "Distracted",
    "Focused",
    "Happy",
    "Neutral",
    "Sleepy",
    "Surprised"
]

uploaded_file = st.file_uploader("Choose a student image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Student Image")

    image = image.resize((128, 128))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100

    st.subheader("Prediction")
    st.success("Engagement State: " + class_names[predicted_class])
    st.info("Confidence: " + str(round(float(confidence), 2)) + "%")
