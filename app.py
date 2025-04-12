import streamlit as st
from PIL import Image
import os
from ai_model import TileDefectDetector

# Page config
st.set_page_config(page_title="TileDefect AI", page_icon="🛠️", layout="wide")

# Initialize model
@st.cache_resource
def load_detector():
    checkpoint_path = os.path.join(os.path.dirname(__file__), "models", "yolov8.pt")  # Adjust path
    print("Loading detector...")
    detector = TileDefectDetector(checkpoint_path)
    print("Detector loaded successfully")
    return detector

detector = load_detector()

# Header
st.title("TileDefect AI")
st.subheader("Automated Tile Quality Inspection with YOLOv8")
st.write("Upload a tile image to detect defects using YOLOv8-powered computer vision.")

# File uploader
uploaded_file = st.file_uploader("Choose a tile image...", type=["jpg", "png", "webp"])

if uploaded_file is not None:
    # Display original image
    original_image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)

    # Analyze button
    if st.button("Analyze Tile"):
        with st.spinner("Analyzing tile image with YOLOv8..."):
            try:
                print("Starting defect detection...")
                defects = detector.detect_defects(original_image)
                print("Defects detected:", defects)
                
                print("Annotating image...")
                processed_image = detector.annotate_image(original_image, defects)
                print("Image annotated")
                
                # Display results
                st.subheader("Detection Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Original Image")
                    st.image(original_image, use_container_width=True)
                
                with col2:
                    st.write("Processed Image with Defects")
                    st.image(processed_image, use_container_width=True)
                
                # Defect summary
                if defects:
                    st.subheader("Defect Analysis Report")
                    critical = sum(1 for d in defects if d['severity'] == 'critical')
                    moderate = sum(1 for d in defects if d['severity'] == 'moderate')
                    minor = sum(1 for d in defects if d['severity'] == 'minor')
                    
                    st.write(f"Critical Defects: **{critical}**")
                    st.write(f"Moderate Defects: **{moderate}**")
                    st.write(f"Minor Defects: **{minor}**")
                    
                    for defect in defects:
                        st.write(f"- {defect['defect_type']} ({defect['severity']}): "
                                f"Position ({defect['x_position']:.1f}%, {defect['y_position']:.1f}%), "
                                f"Size ({defect['width']:.1f}%, {defect['height']:.1f}%), "
                                f"Confidence: {defect['confidence']:.2f}")
                else:
                    st.success("No defects detected. This tile meets quality standards.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                print(f"Error: {str(e)}")

# Footer
st.markdown("---")
