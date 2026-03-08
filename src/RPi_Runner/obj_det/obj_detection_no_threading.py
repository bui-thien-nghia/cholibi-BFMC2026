from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, verbose=False, debug=False):
        self.model = YOLO(model_path)

        # Configurations:
        self.imgsz = 320
        self.half = True
        self.conf = 0.75
        self.verbose = verbose
        self.debug = debug


    def detect(self, img):
        '''Returns a list of objects' classes and normalized coords on image'''
        results = self.model.predict(
            source=img,
            half=self.half,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=self.verbose,
            show=self.debug
        )
        
        return results[0].boxes.cls.tolist(), results[0].boxes.xywhn.tolist()
