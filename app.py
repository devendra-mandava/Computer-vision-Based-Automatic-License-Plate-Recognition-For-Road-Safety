from flask import Flask, request, render_template
import cv2
import os
from src.models.vehicle_detector import VehicleDetector
from src.models.license_plate_detector import LicensePlateDetector
from src.models.ocr_reader import OCRReader

# Initialize Flask app and directories
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize models
vehicle_detector = VehicleDetector('./models/yolov8n.pt')
license_plate_detector = LicensePlateDetector('./models/license_plate_detector.pt')
ocr_reader = OCRReader()

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for handling file uploads and rendering results.
    """
    if request.method == "POST":
        uploaded_file = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(file_path)

        ext = os.path.splitext(uploaded_file.filename)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            output_path = os.path.join(OUTPUT_FOLDER, "output_image_with_boxes.jpg")
            license_plate_texts = process_image(file_path, output_path)

            # Pass the relative path of the processed image to the template
            processed_image_path = f"/{output_path.replace(os.sep, '/')}"
            return render_template("index.html",
                                   license_plate_texts=license_plate_texts,
                                   processed_image=processed_image_path)

        else:
            return "Unsupported file format!", 400

    return render_template("index.html")

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}.")
        return []

    license_plate_bboxes = license_plate_detector.detect(image)
    license_plate_texts = []  # Store detected plates and their confidence

    for idx, license_plate in enumerate(license_plate_bboxes):
        cropped_plate = ocr_reader.preprocess_plate(image, license_plate)

        # Skip invalid cropped plates
        if cropped_plate is None or cropped_plate.size == 0:
            continue

        text, confidence = ocr_reader.read_text(cropped_plate)

        # Append text and confidence, even if confidence is low
        if text:
            license_plate_texts.append((text, confidence))
            print(f"Detected License Plate: {text} (Confidence: {confidence:.2f})")
        else:
            license_plate_texts.append(("No text detected", 0.0))

        # Draw bounding box around license plate
        x1, y1, x2, y2, _ = map(int, license_plate)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text overlay (even for low confidence)
        if text:
            cv2.putText(image, f"{text} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the processed image
    cv2.imwrite(output_path, image)
    print(f"Processed image saved at {output_path}.")
    return license_plate_texts




if __name__ == "__main__":
    app.run(debug=True)
