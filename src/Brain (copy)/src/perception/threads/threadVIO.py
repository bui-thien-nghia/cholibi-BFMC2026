# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import cv2
import numpy as np
import socket
import time
import ast
import math
from picamera2 import Picamera2

from src.templates.threadwithstop import ThreadWithStop
from src.perception.lane_detection_short import LaneDetector
from src.perception.lane_keeping_PID import LaneController
from src.perception.utils.visual_odometry import VisualOdometry2D
from src.perception.utils.landmark_detector import LandmarkDetector
from src.perception.utils.ekf_fusion import EKF_Fusion
from src.perception.utils.map_handler import MapHandler
from src.utils.messages.allMessages import ImuData, VCDCalib, Odometry, NavGoal
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.utils.messages.messageHandlerSender import messageHandlerSender

class threadVIO(ThreadWithStop):
    """This thread handles the Visual Inertial Odometry and Map Navigation logic.\n
    Args:
        queuesList (dictionary of multiprocessing.queues.Queue): Dictionary of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queuesList, logger, debugger=False):
        super(threadVIO, self).__init__(pause=0.01)
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        
        # 1. Initialize Vision and Control
        self.detector = LaneDetector(img_w=640, img_h=480)
        self.controller = LaneController()
        
        # 2. Initialize VIO Components
        self.vo = VisualOdometry2D(
            xm_per_pix=self.detector.xm_per_pix / 100.0,
            ym_per_pix=self.detector.ym_per_pix / 100.0
        )
        self.landmark_detector = LandmarkDetector()
        self.ekf = EKF_Fusion()
        
        # 3. Initialize Map Handler
        self.map_handler = MapHandler()
        self.current_path = [] # List of [x,y] waypoints
        self.current_node = None
        self.target_node = None
        
        # 4. Initialize IPC
        self.imuSubscriber = messageHandlerSubscriber(self.queuesList, ImuData, "lastOnly", True)
        self.navGoalSubscriber = messageHandlerSubscriber(self.queuesList, NavGoal, "lastOnly", True)
        self.vcdSender = messageHandlerSender(self.queuesList, VCDCalib)
        self.odoSender = messageHandlerSender(self.queuesList, Odometry)
        
        # 5. Initialize Camera
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 6. Initialize External Command Socket (UDP)
        # Allows sending NavGoal via terminal: echo "342,98" | nc -u -w1 127.0.0.1 5555
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(('127.0.0.1', 5555))
        self.cmd_sock.setblocking(False)
        
        self.current_speed = 0
        self.prev_control_time = time.time()
        self.last_imu_data = None

    def thread_work(self):
        # --- A. Inputs and Perception ---
        frame = self.picam2.capture_array()
        if frame is None: return

        # 1. Get Nav Goal from Dashboard
        nav_msg = self.navGoalSubscriber.receive()
        
        # 2. ALSO Check External UDP Socket for Nav Goal
        try:
            data, addr = self.cmd_sock.recvfrom(1024)
            nav_msg = data.decode('utf-8').strip()
        except Exception:
            pass

        if nav_msg is not None:
            try:
                start_n, end_n = nav_msg.split(',')
                
                # Snap EKF to the start node position (Global Initialization)
                start_pose = self.map_handler.get_node_pose(start_n)
                if start_pose is not None:
                    self.ekf.set_state(start_pose[0], start_pose[1], 0.0)
                
                # Plan the path
                self.current_path = self.map_handler.get_path(start_n, end_n)
                if self.debugger:
                    self.logger.info(f"VIO: EKF initialized to node {start_n}. Path planned.")
            except Exception as e:
                if self.debugger:
                    self.logger.error(f"VIO: Error parsing NavGoal: {e}")

        # EKF Prediction (IMU)
        imu_msg = self.imuSubscriber.receive()
        if imu_msg is not None:
            try:
                self.last_imu_data = ast.literal_eval(imu_msg)
                # self.ekf.predict(accel_x, accel_y, gyro_z)
            except Exception: pass

        # Vision processing
        binary_warped, warped_color = self.detector.preprocess(frame)
        
        # VO Update
        dx, dy, dtheta = self.vo.process(warped_color)
        # self.ekf.update_vo(dx, dy, dtheta)

        # Landmark Snap
        landmarks = self.landmark_detector.detect(frame)
        # TODO: Lookup landmark global coords and update EKF

        # --- B. Map Localization ---
        state = self.ekf.get_state()
        curr_x, curr_y, curr_theta = state[0], state[1], state[2]
        
        # Find position on graph
        self.current_node = self.map_handler.find_nearest_node(curr_x, curr_y)
        
        # Publish Pose for Dashboard
        self.odoSender.send({
            "x": curr_x, "y": curr_y, "theta": curr_theta,
            "node": self.current_node
        })

        # --- C. Navigation Control ---
        steer = 0.0
        speed = 0.0

        if self.current_path:
            # 1. Map-Based Waypoint Following
            target_wp = self.current_path[0]
            dist_to_wp = math.sqrt((target_wp[0]-curr_x)**2 + (target_wp[1]-curr_y)**2)
            
            if dist_to_wp < 0.2: # Threshold to switch waypoint
                self.current_path.pop(0)
                if not self.current_path:
                    if self.debugger: self.logger.info("VIO: Destination reached.")
            
            # REUSING the existing LaneController logic for waypoints
            steer, speed, state = self.controller.get_waypoint_control(curr_x, curr_y, curr_theta, target_wp)
        else:
            # 2. Fallback to Vision-Based Lane Keeping
            left_poly, right_poly = self.detector.find_lanes(binary_warped)
            steer, speed, state, target_point = self.controller.get_control(left_poly, right_poly, self.current_speed)

        # --- D. Hardware Command ---
        self.current_speed = speed
        steer_scaled = round(steer * 10)
        speed_scaled = round(speed * 10)

        if time.time() - self.prev_control_time > 0.25:
            self.vcdSender.send({"Time": 3, "Speed": speed_scaled, "Steer": steer_scaled})
            self.prev_control_time = time.time()

    def stop(self):
        self.cmd_sock.close()
        self.picam2.stop()
        self.picam2.close()
        super(threadVIO, self).stop()
