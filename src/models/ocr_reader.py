import easyocr  # EasyOCR library for optical character recognition
import cv2  # OpenCV for image processing

class OCRReader:
    """
    OCRReader class for detecting and extracting text from license plates.
    Utilizes EasyOCR for text detection and OpenCV for image preprocessing.
    """

    def __init__(self):
        """
        Initialize the OCRReader instance.
        This includes initializing the EasyOCR reader for English text detection.
        """
        # Initialize EasyOCR reader for English text
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR Reader initialized successfully (GPU: Disabled).")

    def preprocess_plate(self, frame, bbox):
        """
        Preprocess the license plate region for OCR.

        Args:
            frame (numpy.ndarray): Original image frame from which the plate is extracted.
            bbox (list): Bounding box coordinates (x1, y1, x2, y2) of the license plate.

        Returns:
            numpy.ndarray: Preprocessed binary image of the license plate.
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        height, width, _ = frame.shape

        # Ensure the coordinates are within the image boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Crop the region of interest (ROI) containing the license plate
        cropped_plate = frame[y1:y2, x1:x2]
        print(f"Cropped plate dimensions: {cropped_plate.shape}")

        # Convert the cropped image to grayscale
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        print("Converted cropped plate to grayscale.")

        # Resize the image to a standard size for OCR consistency
        resized_plate = cv2.resize(gray_plate, (200, 64))
        print("Resized cropped plate to 200x64 dimensions.")

        # Apply thresholding to binarize the image for better OCR detection
        _, thresh_plate = cv2.threshold(resized_plate, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Applied thresholding to preprocess the license plate.")

        return thresh_plate

    def read_text(self, cropped_plate):
        """
        Perform OCR on the preprocessed license plate image to extract text.

        Args:
            cropped_plate (numpy.ndarray): Preprocessed binary image of the license plate.

        Returns:
            tuple: Detected text (str) and its confidence score (float).
                Returns (None, None) if no valid text is detected.
        """
        detections = self.reader.readtext(cropped_plate)
        print(f"OCR detections: {detections}")

        # Return the first detection, even if confidence is low
        for bbox, text, confidence in detections:
            print(f"Detected text: '{text}' with confidence: {confidence:.2f}")
            if len(text.strip()) > 0:
                return text.strip().upper(), confidence

        # If no text is detected, return None
        print("No text detected in the license plate.")
        return None, None



