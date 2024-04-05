import streamlit as st
import easyocr
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

# Function to perform glasses detection using EasyOCR
def detect_glasses(image):
    # Load the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform text detection and recognition
    result = reader.readtext(image)

    # Filter out detections related to glasses
    glasses_detections = [detection for detection in result if "glasses" in detection[1].lower()]

    # Draw bounding boxes and text on the image
    annotated_image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    draw = ImageDraw.Draw(annotated_image)
    for detection in glasses_detections:
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]
        if confidence > 0.2:
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0][0], bbox[0][1]), text, fill="red")

    return np.array(annotated_image)  # Convert PIL Image back to NumPy array

# Function to read text from an image and draw bounding boxes
def read_text_and_draw(image_path, reader, output_folder):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = reader.readtext(img)

    # Process OCR results
    for detection in results:
        # Extract text and bounding box
        text = detection[1]
        bbox = detection[0]

        # Draw bounding box and text on image
        cv2.rectangle(img_rgb, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (255, 0, 0), 2)
        cv2.putText(img_rgb, text, (int(bbox[0][0]), int(bbox[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the processed image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

# Main function to run the Streamlit app
def main():
    st.title("Glasses Detection and OCR App")
    st.write("Upload an image to detect glasses and perform OCR.")

    # File uploader to upload images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file).convert('RGB')

        # Perform glasses detection and annotation
        glasses_annotated_image = detect_glasses(np.array(image))

        # Display annotated image with glasses detection
        st.image(glasses_annotated_image, caption="Glasses Detection", use_column_width=True)

        # OCR processing
        # Initialize OCR reader
        reader = easyocr.Reader(['en', 'ar'], gpu=False)

        # Create output folder for processed images
        output_folder = 'ocr_results'
        os.makedirs(output_folder, exist_ok=True)

        # Perform OCR and draw bounding boxes
        read_text_and_draw(uploaded_file.name, reader, output_folder)

        # Display processed image with OCR results
        processed_image_path = os.path.join(output_folder, uploaded_file.name)
        processed_image = Image.open(processed_image_path)
        st.image(processed_image, caption="OCR Results", use_column_width=True)

if __name__ == "__main__":
    main()
