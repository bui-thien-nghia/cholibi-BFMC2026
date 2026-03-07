from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # Configurations:
        self.imgsz = 416
        self.half = True
        self.conf = 0.75


    def detect(self, img):
        '''Returns a list of objects' classes and normalized coords on image'''
        result = self.model.predict(
            source=img,
            half=self.half,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )
        
        return result.boxes.cls.tolist(), result.boxes.xywhn.tolist()