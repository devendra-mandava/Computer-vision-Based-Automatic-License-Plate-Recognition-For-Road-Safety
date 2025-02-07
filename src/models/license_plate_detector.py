from ultralytics import YOLO 

class LicensePlateDetector:
    """
    LicensePlateDetector class for detecting license plates in images.
    """

    def __init__(self, model_path):
        """
        Initialize the LicensePlateDetector with a YOLO model.

        Args:
            model_path (str): Path to the pretrained YOLO model.
        """
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Detect license plates in a video frame.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            list: A list of detections in the format [x1, y1, x2, y2, confidence].
        """
        detections = self.model(frame)[0]
        return [
            [x1, y1, x2, y2, confidence] 
            for x1, y1, x2, y2, confidence, class_id in detections.boxes.data.tolist()
        ]
