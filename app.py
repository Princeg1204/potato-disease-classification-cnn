# ==========================================================
# Block 1 : Import Required Libraries
# ==========================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ==========================================================
# Configure Streamlit Page
# ==========================================================

st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="centered"
)

# ==========================================================
# Block 2 : Load Model and Class Names
# ==========================================================

@st.cache_resource
def load_resources():

    # Load trained model
    model = tf.keras.models.load_model("models/best_model.keras")

    # Load class names
    with open("models/class_names.json", "r") as file:
        class_names = json.load(file)

    return model, class_names


# Load resources
model, class_names = load_resources()

# ==========================================================
# Block 3 : Image Preprocessing
# ==========================================================

IMAGE_SIZE = (256, 256)

def preprocess_image(image):

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# ==========================================================
# Block 4 : Prediction Function (Debug Version)
# ==========================================================

def predict(image):

    # Preprocess image
    processed_image = preprocess_image(image)

    # Predict
    predictions = model.predict(processed_image, verbose=0)

    # ===============================

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(predictions[0]) * 100)

    return predicted_class, confidence

# ==========================================================
# Block 5 : User Interface + Session State
# ==========================================================

st.title("🥔 Potato Disease Classification")

st.write(
    """
Welcome to the **Potato Disease Classification** application.

Upload a potato leaf image, and the trained Deep Learning model
will predict whether the leaf is:

• 🍃 Early Blight

• 🍂 Late Blight

• ✅ Healthy
"""
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.header("🥔 Potato Disease Classifier")

    st.write(
        """
This application uses a **Convolutional Neural Network (CNN)**
trained on the PlantVillage dataset.
"""
    )

    st.divider()

    st.subheader("Supported Classes")

    st.write("🍃 Early Blight")
    st.write("🍂 Late Blight")
    st.write("✅ Healthy")

    st.divider()

    st.subheader("Model Details")

    st.write("Framework : TensorFlow / Keras")
    st.write("Architecture : CNN")
    st.write("Input Size : 256 × 256")
    st.write("Classes : 3")
    st.write("Test Accuracy : 96.09%")

# ==========================================================
# Session State Initialization
# ==========================================================

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ==========================================================
# Block 10 : Upload Instructions
# ==========================================================

st.info(
    """
### 📤 Upload Instructions

For the best prediction results:

- 🌿 Upload **only a potato leaf image**
- 📸 Use a clear, well-lit image
- 🍃 Ensure the leaf is clearly visible
- 🚫 Do not upload images of people, animals, vehicles, or other objects

**Note:** This AI model is trained only on potato leaf images. Uploading unrelated images may produce incorrect predictions.
"""
)


# ==========================================================
# Block 6 : Upload Image
# ==========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

# ==========================================================
# Block 7 : Prediction + Result + Disease Information
# ==========================================================

    # Predict Button
    if st.button("🔍 Predict Disease", use_container_width=True):

        with st.spinner("Analyzing the leaf image..."):

            predicted_class, confidence = predict(image)

        # ==================================================
        # Prediction Result
        # ==================================================

        st.success("✅ Prediction Completed Successfully!")

        display_names = {
            "Potato___Early_blight": "🍃 Early Blight",
            "Potato___Late_blight": "🍂 Late Blight",
            "Potato___healthy": "✅ Healthy"
        }

        st.subheader("🌿 Prediction Result")

        st.markdown(
            f"### Disease: **{display_names[predicted_class]}**"
        )

        st.markdown(
            f"### Confidence: **{confidence:.2f}%**"
        )

        st.progress(confidence / 100)

        # ==================================================
        # Disease Information
        # ==================================================

        disease_info = {

            "Potato___Early_blight": {

                "about":
                "Early Blight is a fungal disease caused by Alternaria solani. "
                "It usually appears as dark brown circular spots with concentric rings, mainly on older leaves.",

                "recommendation": [

                    "Remove infected leaves immediately.",

                    "Apply a recommended fungicide.",

                    "Avoid overhead irrigation.",

                    "Maintain proper spacing between plants."

                ]
            },

            "Potato___Late_blight": {

                "about":
                "Late Blight is caused by Phytophthora infestans. "
                "It spreads rapidly in cool and humid weather and may destroy the entire potato crop if not controlled.",

                "recommendation": [

                    "Remove infected plants immediately.",

                    "Apply an effective fungicide.",

                    "Reduce excess moisture around plants.",

                    "Inspect nearby plants regularly."

                ]
            },

            "Potato___healthy": {

                "about":
                "The uploaded potato leaf appears healthy with no visible symptoms of Early Blight or Late Blight.",

                "recommendation": [

                    "Continue regular irrigation.",

                    "Maintain balanced fertilization.",

                    "Inspect leaves periodically.",

                    "Follow good agricultural practices."

                ]
            }
        }

        st.divider()

        st.subheader("📖 Disease Information")

        st.write(disease_info[predicted_class]["about"])

        st.subheader("💡 Recommendations")

        for tip in disease_info[predicted_class]["recommendation"]:
            st.write(f"✅ {tip}")

        # ==================================================
        # Model Information
        # ==================================================

        st.divider()

        st.subheader("🤖 Model Information")

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                """
**Model**

• CNN (TensorFlow / Keras)

• Input Size: 256 × 256

• Number of Classes: 3

• Framework: TensorFlow
"""
            )

        with col2:

            st.success(
                """
**Performance**

• Test Accuracy: 96.09%

• Real-Time Image Classification

• Optimized for Deployment

• Streamlit Powered Application
"""
            )

# ==========================================================
# Block 9 : About Dataset
# ==========================================================

st.divider()

with st.expander("📚 About This Project"):

    st.markdown("""
### 🌱 Dataset

This application is trained using the **PlantVillage Dataset**, a widely used benchmark dataset for plant disease classification.

### 🧠 Model

- Convolutional Neural Network (CNN)
- TensorFlow / Keras
- Input Size: **256 × 256**
- Number of Classes: **3**

### 📂 Supported Diseases

- 🍃 Early Blight
- 🍂 Late Blight
- ✅ Healthy

### 📌 How to Use

1. Upload a potato leaf image.
2. Click **Predict Disease**.
3. View the prediction, confidence score, disease information, and recommendations.

### ⚠️ Note

This application is intended for educational and demonstration purposes. It should not replace professional agricultural diagnosis.
""")


# ==========================================================
# Block 8 : Footer
# ==========================================================

st.divider()

st.markdown(
    """
<div style="text-align:center">

### 👨‍💻 Developed by Prince Gajera

**B.Tech Information Technology**

Deep Learning • Computer Vision • TensorFlow • Streamlit

---

📌 **Potato Disease Classification using Convolutional Neural Network (CNN)**

⚠️ *This application is intended for educational and demonstration purposes. Always consult an agricultural expert before making crop management decisions.*

</div>
""",
    unsafe_allow_html=True
)