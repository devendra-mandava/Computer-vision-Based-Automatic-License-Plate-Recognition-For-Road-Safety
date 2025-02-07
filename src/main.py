import cv2
import os
from models.vehicle_detector import VehicleDetector
from models.license_plate_detector import LicensePlateDetector
from models.ocr_reader import OCRReader
from utils.data_writer import write_results_to_csv


def process_image(image_path, output_path):
    """
    Processes the uploaded image to detect vehicles and license plates,
    draw bounding boxes, and save the final image with annotations.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image with annotations.

    Returns:
        list[tuple]: Detected license plates and their confidence scores.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}.")
        return []

    # Detect license plates
    license_plate_bboxes = license_plate_detector.detect(image)
    license_plate_texts = []

    for idx, license_plate in enumerate(license_plate_bboxes):
        # Preprocess the license plate
        cropped_plate = ocr_reader.preprocess_plate(image, license_plate)

        # Read the text from the cropped plate
        text, confidence = ocr_reader.read_text(cropped_plate)
        if text:
            license_plate_texts.append((text, confidence))

        # Draw bounding box around the detected license plate
        x1, y1, x2, y2, _ = map(int, license_plate)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if text:
            cv2.putText(image, f"{text} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save the processed image with annotations
    cv2.imwrite(output_path, image)
    print(f"Processed image with bounding boxes saved at: {output_path}")

    return license_plate_texts


def process_image_with_box(image_path, vehicle_detector, license_plate_detector, ocr_reader):
    """
    Process an image to detect vehicles and license plates,
    draw bounding boxes around detected plates, and save the result.

    Args:
        image_path (str): Path to the input image.
        vehicle_detector (object): Vehicle detection model.
        license_plate_detector (object): License plate detection model.
        ocr_reader (object): OCR reader for text extraction.

    Returns:
        str: Path to the processed image with bounding boxes.
    """
    # Read the input image
    image = cv2.imread(image_path)

    # Detect license plates
    license_plate_bboxes = license_plate_detector.detect(image)

    for bbox in license_plate_bboxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2, _ = map(int, bbox)

        # Draw a red rectangle around the detected license plate
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Crop and preprocess the license plate for OCR
        cropped_plate = ocr_reader.preprocess_plate(image, bbox)

        # Read the text from the license plate
        text, confidence = ocr_reader.read_text(cropped_plate)
        if text:
            print(f"Detected License Plate: {text} (Confidence: {confidence:.2f})")
            # Draw the license plate text above the bounding box
            cv2.putText(
                image, f"{text} ({confidence:.2f})",
                (x1, y1 - 10),  # Position the text above the rectangle
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    # Save the processed image to the output folder
    output_image_path = "./static/output/processed_image.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved to {output_image_path}")

    return output_image_path


def process_video(video_path, vehicle_detector, license_plate_detector, ocr_reader):
    """
    Process a video to detect vehicles, license plates, and extract text.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video path
    output_video_path = "./static/output/processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    results = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results[frame_idx] = {}

        # Detect vehicles
        vehicle_bboxes = vehicle_detector.detect(frame)

        # Detect license plates
        license_plate_bboxes = license_plate_detector.detect(frame)

        for idx, bbox in enumerate(license_plate_bboxes):
            x1, y1, x2, y2, _ = map(int, bbox)

            # Draw bounding box around license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Crop and preprocess the license plate
            cropped_plate = ocr_reader.preprocess_plate(frame, [x1, y1, x2, y2])
            if cropped_plate is not None:
                text, confidence = ocr_reader.read_text(cropped_plate)
                if text:
                    print(f"Frame {frame_idx}, License Plate {idx}: {text} (Confidence: {confidence:.2f})")
                    results[frame_idx][idx] = {"text": text, "confidence": confidence}

        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

    # Save results to CSV
    csv_path = "./static/output/video_results.csv"
    write_results_to_csv(results, csv_path)
    print(f"Results saved to {csv_path}")

    return output_video_path


if __name__ == "__main__":
    # Initialize models
    vehicle_detector = VehicleDetector('./models/yolov8n.pt')
    license_plate_detector = LicensePlateDetector('./models/license_plate_detector.pt')
    ocr_reader = OCRReader()

    # File path for testing
    file_path = "./data/sample.mp4"

    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file_path, vehicle_detector, license_plate_detector, ocr_reader)
    elif file_path.lower().endswith(('.mp4', '.avi')):
        output_video = process_video(file_path, vehicle_detector, license_plate_detector, ocr_reader)
        print(f"Processed video available at: {output_video}")
    else:
        print("Unsupported file format! Please provide an image or video.")
