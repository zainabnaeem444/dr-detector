import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Download model from Google Drive
@st.cache_resource
def load_dr_model():
    url = 'https://drive.google.com/uc?id=1gwCm1YTUHxLtGl6Gj5Ck4RXvRj3M37am'
    response = requests.get(url)
    with open("final_epoch13_855acc_fullmodel.h5", "wb") as f:
        f.write(response.content)
    return load_model("final_epoch13_855acc_fullmodel.h5")

model = load_dr_model()
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

st.title("🩺 Diabetic Retinopathy Detector")
st.markdown("Upload a retina image to detect DR stage using AI")

uploaded_file = st.file_uploader("Choose a retina image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Preview', use_column_width=True)

    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        preds = model.predict(img_array)
        prediction = class_names[np.argmax(preds)]
        st.success(f"🧠 Prediction: {prediction}")
        st.bar_chart(preds[0])