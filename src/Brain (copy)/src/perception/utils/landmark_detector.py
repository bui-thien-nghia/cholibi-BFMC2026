import os
import numpy as np
import cv2

# Attempt to import ultralytics; gracefully degrade if unavailable
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

class LandmarkDetector:
    """
    Landmark Detection using Ultralytics YOLO (v8/v11).
    Adapted from ROS2 Sign Detector node.
    """
    def __init__(self, 
                 model_path='src/perception/model/obj.pt', 
                 conf_threshold=0.5,
                 imgsz=416,
                 device='', # '' = auto (cpu / cuda)
                 half=False):
        
        self.conf = conf_threshold
        self.imgsz = imgsz
        self.device = device or None
        self.half = half
        
        # Mapping from Class ID to Name (derived from comprehensive_sim.py)
        self.classes = {
            0: 'pedestrian', 1: 'cyclist', 2: 'car', 3: 'bus', 4: 'truck',
            5: 'red_light', 6: 'yellow_light', 7: 'green_light',
            8: 'pedestrian_sign', 9: 'enter_highway_sign', 10: 'leave_highway_sign',
            11: 'oneway_sign', 12: 'parking_sign', 13: 'priority_sign',
            14: 'notallowed_sign', 15: 'roundabout_sign', 16: 'stop_sign'
        }

        self.model = None
        if not YOLO_AVAILABLE:
            print('[ERROR] LandmarkDetector: ultralytics is not installed. Run: pip install ultralytics')
        elif not model_path or not os.path.isfile(model_path):
            print(f'[WARNING] LandmarkDetector: No valid YOLO model at "{model_path}". Detection disabled.')
        else:
            try:
                self.model = YOLO(model_path)
                print(f'[INFO] LandmarkDetector: YOLO model loaded successfully from {model_path}')
            except Exception as e:
                print(f'[ERROR] LandmarkDetector: Failed to load YOLO model: {e}')

    def detect(self, frame):
        """
        Runs YOLO inference on a frame and returns detected landmarks.
        Returns list of dicts: [{'class_id': id, 'name': name, 'box': [x,y,w,h], 'conf': conf, 'xywhn': [cx,cy,w,h]}]
        """
        if self.model is None:
            return []

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            half=self.half,
            device=self.device,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                xywh = r.boxes.xywh.cpu().numpy()   # pixel coordinates [cx, cy, w, h]
                xywhn = r.boxes.xywhn.cpu().numpy() # normalized [cx, cy, w, h]

                for cls_id, conf, box, box_n in zip(cls_ids, confs, xywh, xywhn):
                    cx, cy, w, h = box
                    # Convert cx,cy to top-left x,y for 'box' consistency
                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    
                    detections.append({
                        'class_id': int(cls_id),
                        'name': self.model.names[int(cls_id)] if hasattr(self.model, 'names') else self.classes.get(int(cls_id), 'unknown'),
                        'box': [x, y, int(w), int(h)],
                        'conf': float(conf),
                        'xywhn': box_n.tolist() # [cx_norm, cy_norm, w_norm, h_norm]
                    })
        
        return detections

    def estimate_distance(self, detection):
        """
        Estimates physical distance to landmark based on bounding box height.
        """
        x, y, w, h = detection['box']
        if h == 0: return 0
        # Heuristic calibration (example value)
        distance = 1.0 / h * 1000 
        return distance
