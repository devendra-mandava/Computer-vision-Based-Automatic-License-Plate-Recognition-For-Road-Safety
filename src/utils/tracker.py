from utils.sort import Sort

class SORTTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        return self.tracker.update(detections)

    def get_matching_vehicle(self, license_plate_bbox):
        # Custom logic to match a license plate to a tracked vehicle
        pass
