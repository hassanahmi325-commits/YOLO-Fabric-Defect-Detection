
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fabric Defect Detection",
    page_icon="🧵",
    layout="centered"
)

# App title
st.title("🧵 Fabric Defect Detection")
st.write(
    "Upload a fabric image to detect defects "
    "using YOLO AI Model."
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Fabric Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Fabric Image")
    st.image(image, use_container_width=True)

    # Detection button
    if st.button("🔍 Detect Fabric Defects"):

        with st.spinner("Detecting Fabric Defects..."):

            results = model.predict(image)

            # Get detected image
            result_image = results[0].plot()

            # Convert BGR to RGB
            result_image = result_image[:, :, ::-1]

            # Display result
            st.subheader("Fabric Defect Detection Result")

            st.image(
                result_image,
                channels="RGB",
                use_container_width=True
            )

            # Detection count
            count = len(results[0].boxes)

            st.success(
                f"Detection Completed! Found {count} Fabric Defect(s)."
            )
