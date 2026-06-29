import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍",
    layout="centered"
)

st.title("✍ Handwritten Digit Recognition")

st.write(
    "Draw a digit (0-9) below and click Predict."
)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.keras")

model = load_model()

# -----------------------------
# DRAW CANVAS
# -----------------------------

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------

if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data[:, :, 0]

        img = Image.fromarray(img.astype("uint8"))

        img = img.resize((28, 28))

        img = np.array(img)

        img = img.astype("float32") / 255.0

        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)

        digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit : {digit}")

        st.info(f"Confidence : {confidence:.2f}%")

        st.bar_chart(prediction[0])

        st.image(
            img.reshape(28,28),
            width=180,
            caption="Processed Image"
        )

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Project")

st.sidebar.write("CNN Model")

st.sidebar.write("MNIST Dataset")

st.sidebar.write("Accuracy : 98-99%")

st.sidebar.write("Draw digits using mouse")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.caption(
    "Developed using TensorFlow, Streamlit and CNN"
)