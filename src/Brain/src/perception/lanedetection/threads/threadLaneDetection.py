import base64
import cv2
import numpy as np
import time
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.allMessages import mainCamera, laneDetectionResult
from Perception_UNDER_DEVELOPMENT.lane_detect import (
    run_lane_detect
)

class threadLaneDetection(ThreadWithStop):
    def __init__(self, queuesList, logger, debugger):
        super(threadLaneDetection, self).__init__(pause=0.05)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger

        self.cameraSub = messageHandlerSubscriber(queuesList, mainCamera, "lastOnly", True)
        self.laneSender = messageHandlerSender(queuesList, laneDetectionResult)
    
        # Taken from real-life measures and PyCam 3 wide-angle's specs
        self.camera_height = 0.225
        self.camera_angle = 15
        self.focal_length = 2.75e-3
        self.m_per_px = 1.3e-6


    def thread_work(self):
        cameraRecv = self.cameraSub.receive()
        if cameraRecv is None:
            return
        
        try:
            img_data = base64.b64decode(cameraRecv)
            img_buffer = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

            path_poly, first_lane_poly, second_lane_poly = run_lane_detect(
                img,
                h=self.h, # ủa cái này hình như là self.camera_height ý
                theta=self.camera_angle,
                f=self.focal_length,
                k=self.m_per_px,
            )

            if path_poly is not None:
                self.laneSender.send({
                    'path': path_poly,
                    'first_lane': first_lane_poly,
                    'second_lane': second_lane_poly,
                    'timestamp': time.time()
                })
                    
        except Exception as e:
            self.logger.error(f'LANE DETECTION ERROR: {e}')