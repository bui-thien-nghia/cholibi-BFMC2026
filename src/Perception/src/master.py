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
# TEST_IMAGES_PATH = "test/test/*.jpg" 
# SHOW_GUI = False       

# def draw_steering_dashboard(img, steer_angle, speed, sign_class):
#     h, w = img.shape[:2]
#     overlay = img.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
#     color = (0, 255, 0) # Green
#     if abs(steer_angle) > 15: color = (0, 165, 255) # Orange
#     if abs(steer_angle) > 23: color = (0, 0, 255)   # Red

#     cv2.putText(img, f"STEER: {steer_angle:.1f} deg", (20, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     cv2.putText(img, f"SPEED: {speed:.1f} cm/s", (20, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
#     # Static sign text for testing
#     cv2.putText(img, f"SIGN: DISABLED", (300, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

#     # Visual Steering Arrow
#     center_x, center_y = w // 2, h
#     arrow_len = 100
#     rad = np.deg2rad(steer_angle - 90) 
#     end_x = int(center_x + arrow_len * np.cos(rad))
#     end_y = int(center_y + arrow_len * np.sin(rad))
#     cv2.arrowedLine(img, (center_x, center_y-20), (end_x, end_y), color, 4, tipLength=0.3)
    
#     return img

# def project_lane_overlay(img, lane_detector, left_poly, right_poly):
#     if left_poly is None or right_poly is None: return img
    
#     # 1. Generate Y values (0 to height)
#     ploty = np.linspace(0, lane_detector.img_h - 1, lane_detector.img_h)
    
#     # 2. Convert Screen Y to Real Y (Bottom-up) for the poly calc
#     real_y = (lane_detector.img_h - ploty) * lane_detector.ym_per_pix
    
#     try:
#         # 3. Get Real X from poly
#         left_real_x = left_poly(real_y)
#         right_real_x = right_poly(real_y)
        
#         # 4. Convert Real X back to Screen X (Add offset)
#         center_offset = lane_detector.img_w / 2.0
#         left_fitx = (left_real_x / lane_detector.xm_per_pix) + center_offset
#         right_fitx = (right_real_x / lane_detector.xm_per_pix) + center_offset
        
#         # Cast to int
#         left_fitx = left_fitx.astype(np.int32)
#         right_fitx = right_fitx.astype(np.int32)
#     except Exception as e:
#         print(f"Vis Error: {e}")
#         return img 

    # # Drawing (Standard OpenCV stuff)
    # warp_zero = np.zeros((lane_detector.img_h, lane_detector.img_w), dtype=np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # pts = np.hstack((pts_left, pts_right))

    # cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # newwarp = cv2.warpPerspective(color_warp, lane_detector.Minv, (img.shape[1], img.shape[0])) 
    # return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

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
    # --- 1. Multiprocessing Disabled for Image Test ---
    # manager = Manager()
    # shared_dict = manager.dict()
    # shared_sign = Value('i', 0)
    # p_sign = Process(target=sign_process_func, args=(shared_dict, shared_sign))
    # p_sign.start()

    # --- 2. Load Images ---
    # images = glob.glob(TEST_IMAGES_PATH)
    # if not images:
    #     print(f"No images found in {TEST_IMAGES_PATH}. Please check the path.")
    #     return
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
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

    # cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)

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

            # controller.prev_steer = 0.0
            steer, speed, state = controller.get_control(left_poly, right_poly)
            steer = round(steer * 10)
            speed = round(speed * 10)

            if 'ser' in locals() and ser.is_open:
                print(f"#vcdCalib:{speed};{steer};50;;\r\n")
                ser.write(f"#vcdCalib:{speed};{steer};10;;\r\n".encode('utf-8'))
                ser.flush()
                time.sleep(0.5)

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

    # for fname in images:
    #     frame = cv2.imread(fname)
    #     if frame is None: continue

    #     # 1. Preprocess
    #     binary_warped, warped_color = detector.preprocess(frame)
        
    #     # 2. Find Lanes
    #     left_poly, right_poly = detector.find_lanes(binary_warped)
        
    #     controller.prev_steer = 0.0 
        
    #     steer, speed, state = controller.get_control(left_poly, right_poly)
        
    #     print(f"Image: {os.path.basename(fname)} | Steer: {steer:.2f} | Speed: {speed:.2f} | State: {state}")

    #     if SHOW_GUI:
    #         # 1. Overlay Lane
    #         result_img = project_lane_overlay(frame, detector, left_poly, right_poly)
            
    #         # 2. Overlay Dashboard
    #         result_img = draw_steering_dashboard(result_img, steer, speed, 0)
            
    #         # 3. Show Result
    #         cv2.imshow("Test Mode: Result", result_img)
            
    #         # Show Debug Views (Optional)
    #         # cv2.imshow("Debug: Binary Bird's Eye", binary_warped * 255)

    #         # Wait for user input to move to next image
    #         key = cv2.waitKey(0) 
    #         if key == ord('q'): # Press 'q' to quit early
    #             break

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
