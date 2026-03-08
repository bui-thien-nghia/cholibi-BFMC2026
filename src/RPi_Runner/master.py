import cv2
import socket
import os
import sys
import argparse
import serial
import traceback
import time
import numpy as np

# !CHANGE THIS AFTER FINISHING
sys.path.insert(0, os.path.abspath('./obj_det/'))
sys.path.insert(0, os.path.abspath('./lane_keep/'))
sys.path.insert(0, os.path.abspath('./mode_changer/'))

from camera import Pi5Camera
from live_debugger import LaneVisualizer
from lane_keeping_PID import LaneController
from lane_detection_short import LaneDetector
from Perception.obj_det.obj_detection_no_threading import ObjectDetector
from carMode import CarModeChanger

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE   = 115200
CAMERA_ID   = -1
LAPTOP_IP = '192.168.50.1'
# LAPTOP_IP = '0.0.0.0'
LAPTOP_PORT = 9999
MODEL_PATH = './obj_det/model/obj.onnx'
UPDATE_RATE = 0.25

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help="Toggle on/off the debug mode for detection visualization")
    args = parser.parse_args()

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
        ser.flush()
        print(f"Connected to Microcontroller on {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
    
    ser.write("#kl:30;;\r\n".encode('utf-8'))
    ser.flush()
    ser.write("#imu:0;;\r\n".encode('utf-8'))
    ser.flush()
    ser.write("#instant:0;;\r\n".encode('utf-8'))
    ser.flush()
    ser.write("#battery:0;;\r\n".encode('utf-8'))
    ser.flush()
    ser.write("#resourceMonitor:0;;\r\n".encode('utf-8'))
    ser.flush()

    time.sleep(0.2)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000) # Set max sending bytes to 1000000 bytes
    current_speed = 0
    prev_time = time.time()
    loop_prev_time = time.time()

    cap = Pi5Camera(width=854, height=480)
    visualizer = LaneVisualizer(img_w=854, img_h=480)
    lane_detector = LaneDetector(img_w=854, img_h=480)
    controller = LaneController()
    obj_detector = ObjectDetector(model_path=MODEL_PATH, debug=args.debug)
    mode_changer = CarModeChanger() 
    
    if args.debug:
        print(f'Streaming debug data on port {LAPTOP_PORT}')

    try:
        while True:
            now = time.time()
            loop_dt = now - loop_prev_time
            loop_prev_time = now
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Camera frame lost!")
                time.sleep(0.1)
                continue
            
            # Lane detection
            binary_warped, warped_color = lane_detector.preprocess(frame)
            left_poly, right_poly = lane_detector.find_lanes(binary_warped)

            
            try:
                obj_detector.update_frame(frame)
                obj_cls, obj_coordn = obj_detector.get_latest_detections()
                
                mode_changer.record_detection(obj_cls, obj_coordn)
                mode_changer.update_timer(loop_dt)
                mode_changer.change_state()
                
                # Get the max allowed speed (0, 20, 30, or 50)
                yolo_speed_limit = mode_changer._get_speed().value 
                
            except Exception as e:
                print(f"[ERROR]: {e}")
                yolo_speed_limit = 30 # Default safe speed if YOLO fails

            target_point = None

            # 3. If YOLO says STOP (0), bypass Pure Pursuit entirely
            if yolo_speed_limit == 0:
                steer = np.rad2deg(controller.prev_steer)
                speed = 0.0
                target_point = (0, 0)
            else:
                controller.max_speed = float(yolo_speed_limit)

                steer, speed, _, target_point = controller.get_control(left_poly, right_poly, current_speed)

                if yolo_speed_limit == 50:
                    speed = 50.0

            current_speed = speed
            serial_steer = round(steer * 10)
            serial_speed = round(speed * 10)


            if 'ser' in locals() and ser.is_open:
                if time.time() - prev_time > UPDATE_RATE:
                    print(f"#vcdCalib:{serial_speed};{serial_steer};2.8;;\r\n")
                    print(f"total process time per iter: {time.time() - prev_time}s")
                    ser.write(f"#vcdCalib:{serial_speed};{serial_steer};2.8;;\r\n".encode('utf-8'))
                    prev_time = time.time()
            
            # Debug
            if args.debug:
                debug_frame = visualizer.draw_debug_frame(binary_warped, left_poly, right_poly, target_point, steer, speed)

                _, encoded_img = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

                try:
                    data = encoded_img.tobytes()
                    client_socket.sendto(b'IMG' + data, (LAPTOP_IP, LAPTOP_PORT))
                except Exception as e:
                    print(f'error streaming: {e}, data length is {len(data)}')

    except KeyboardInterrupt:
        print("\nStopping...")
        if 'ser' in locals() and ser.is_open:
            # Send stop command
            ser.write("#vcdCalib:0.0;0.0;0;;\r\n".encode('utf-8'))
            ser.close()
        
    except Exception:
        traceback.print_exc()
        
    finally:
        cap.release()

if __name__ == "__main__":
    main()

