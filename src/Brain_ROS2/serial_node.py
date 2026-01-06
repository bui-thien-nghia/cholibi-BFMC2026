import cv2
import time
import serial
import traceback
from lane_detect import run_lane_detect
from lane_keeping import LaneController

# CONFIGURATION
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE   = 115200          
CAMERA_ID   = 0               

# MEASURED CONSTANTS
CAMERA_HEIGHT = 20.5      
CAMERA_PITCH  = 25.0      
FOCAL_LENGTH  = 273.0     
LANE_WIDTH    = 37.0      
WHEELBASE     = 26.5      

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        print(f"Connected to Microcontroller on {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    controller = LaneController(
        wheelbase=WHEELBASE, 
        max_steering_angle=25, 
        max_speed=30.0, 
        min_speed=10.0,
        lane_width=LANE_WIDTH
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame lost!")
                time.sleep(0.1)
                continue

            try:
                _, left_poly, right_poly = run_lane_detect(
                    frame, 
                    h=CAMERA_HEIGHT, 
                    theta=CAMERA_PITCH, 
                    f=FOCAL_LENGTH, 
                    k=1.0,
                    use_deprojected=True
                )
            except Exception:
                left_poly, right_poly = None, None

            steer, speed, state = controller.get_control(left_poly, right_poly, current_speed=15.0)

            if 'ser' in locals() and ser.is_open:
                ser.write(f"#vcdCalib:{speed:.2f};{steer:.2f};100;;\r\n".encode('utf-8'))

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