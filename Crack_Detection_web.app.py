import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

st.set_page_config(page_title="AI Crack Detection", page_icon="🏗️", layout="centered")

st.title("🏗️ AI-Based Building Crack Detection")
st.write("Preliminary Visual Assessment Tool")

uploaded_file = st.file_uploader("Upload Concrete Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Start AI Inspection"):
        with st.spinner("Analyzing image..."):
            
            results = model(image)
            result = results[0]
            
            if len(result.boxes) > 0:
                has_crack = True
                confidence = float(result.boxes.conf[0]) * 100 
                
                res_image = result.plot() 
                st.image(res_image, caption="AI Detection Result", use_container_width=True)
            else:
                has_crack = False
                confidence = 0.0

            if has_crack and confidence >= 85:
                severity = "High Risk"
            elif has_crack and confidence < 85:
                severity = "Moderate"
            else:
                severity = "Low / No Risk"

            st.success("Analysis Complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Crack Detected", value="Yes" if has_crack else "No")
                st.metric(label="Confidence", value=f"{confidence:.1f}%")
            with col2:
                st.metric(label="Severity Level", value=severity)

            st.subheader("📋 Engineering Recommendation")
            if severity == "High Risk":
                st.error("🚨 **Immediate professional structural inspection recommended. Do not rely on visual analysis alone.**")
            elif severity == "Moderate":
                st.warning("⚠️ **Further visual inspection and documentation of crack dimensions required.**")
            else:
                st.info("ℹ️ **No severe cracks detected. Monitor periodically.**")

            st.caption("Disclaimer: This is an AI-assisted preliminary inspection tool, not a replacement for a structural engineer.")