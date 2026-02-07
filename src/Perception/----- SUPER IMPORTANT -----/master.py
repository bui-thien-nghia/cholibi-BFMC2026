import cv2
import numpy as np
import os
import glob
from lane_detection import LaneDetector
from lane_keeping import LaneController
import serial
import traceback
import time
from picamera2 import Picamera2

SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE   = 115200          
CAMERA_ID   = -1

class Pi5Camera:
    def __init__(self, width=640, height=480):
        # Initialize the official Pi5 camera library
        self.picam2 = Picamera2()
        
        # Configure it for BGR video (OpenCV standard)
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"}
        )
        self.picam2.configure(config)
        
        # Start the camera continuously
        self.picam2.start()

    def read(self):
        # Grab the latest frame directly as a numpy array
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                return False, None
            return True, frame
        except Exception as e:
            print(f"Picamera2 Error: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
        self.picam2.close()

# --- MAIN LOOP ---
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        ser.reset_input_buffer()
        ser.reset_output_buffer()
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

    cap = Pi5Camera(width=640, height=480)

    detector = LaneDetector(img_w=640, img_h=480)
    controller = LaneController()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame lost!")
                time.sleep(0.1)
                continue

            binary_warped, warped_color = detector.preprocess(frame)
            left_poly, right_poly = detector.find_lanes(binary_warped)

            steer, speed, state = controller.get_control(left_poly, right_poly)
            steer = round(steer * 8)
            speed = round(speed * 8)

            if 'ser' in locals() and ser.is_open:
                print(f"#vcdCalib:{speed};{steer};50;;\r\n")
                ser.write(f"#vcdCalib:{speed};{steer};10;;\r\n".encode('utf-8'))
                ser.flush()
                time.sleep(0.25)

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
