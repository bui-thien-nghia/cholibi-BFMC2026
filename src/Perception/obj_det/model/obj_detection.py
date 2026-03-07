from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, path_to_model):
        self.model = YOLO(path_to_model)

    def detect(self, img):
        self.model.predict(
            
        )