import threading
import time
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.imgsz = 320
        self.half = True
        self.conf = 0.75 # Slightly lowered for speed, tune as needed

        # Threading variables
        self.current_frame = None
        self.detections = ([], []) # (classes, coordinates)
        self.running = True
        self.lock = threading.Lock()
        
        # Start the background thread
        self.thread = threading.Thread(target=self._run_inference, daemon=True)
        self.thread.start()

    def update_frame(self, img):
        """Called in the main loop to give the thread the newest frame."""
        with self.lock:
            self.current_frame = img.copy()

    def get_latest_detections(self):
        """Returns the most recent detections without blocking."""
        with self.lock:
            return self.detections

    def _run_inference(self):
        """Runs in the background, continuously processing the latest frame."""
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame
                    self.current_frame = None # Process it once

            if frame_to_process is not None:
                results = self.model.predict(
                    source=frame_to_process,
                    half=self.half,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    verbose=False # Turn off verbose to save console I/O time
                )
                
                cls = results[0].boxes.cls.tolist()
                coords = results[0].boxes.xywhn.tolist()
                
                with self.lock:
                    self.detections = (cls, coords)
            else:
                time.sleep(0.01) # Wait briefly if no new frame is ready

    def stop(self):
        self.running = False
        self.thread.join()