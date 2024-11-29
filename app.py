import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO('/media/husnain/nain/Programming/Python/UNI/YOLO_Acne_Detection/Deployment/best_yolov8_model.pt')

# Streamlit app
def main():
    st.title("YOLOv8 Image Prediction App")
    st.write("Upload an image to get predictions!")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing the image...")

        # Save the uploaded file temporarily for YOLO input
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Perform prediction
        results = model.predict(
            source=temp_path,
            conf=0.25
        )

        # Display the predictions
        for result in results:
            # Convert result.plot() (numpy array) to an image
            predicted_img = result.plot()
            predicted_img = Image.fromarray(predicted_img)
            st.image(predicted_img, caption="Predicted Image", use_column_width=True)

if __name__ == "__main__":
    main()
