from ultralytics import YOLO

class VehicleDetector:
    """
    VehicleDetector class for detecting vehicles in an image using YOLO.
    """

    def __init__(self, model_path):
        """
        Initialize the VehicleDetector with a YOLO model.

        Args:
            model_path (str): Path to the pretrained YOLO model.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect vehicles in a video frame.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            list: A list of detections in the format [x1, y1, x2, y2, confidence].
        """
        detections = self.model(frame)[0]
        vehicle_classes = [2, 3, 5, 7]  # Class IDs for vehicles (car, bus, truck, etc.)
        return [
            [x1, y1, x2, y2, confidence]
            for x1, y1, x2, y2, confidence, class_id in detections.boxes.data.tolist()
            if int(class_id) in vehicle_classes
        ]
