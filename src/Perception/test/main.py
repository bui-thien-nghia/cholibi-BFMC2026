import cv2
import numpy as np
import traceback
from src.lane_detect import run_lane_detect
from lane_keeping import LaneController

CAMERA_HEIGHT = 20.5      # cm
CAMERA_PITCH  = 25.0      # degrees
FOCAL_LENGTH  = 273.0     # calibrated pixels (for 640 width)
LANE_WIDTH    = 37.0      # cm
WHEELBASE     = 26.5      # cm

# Image to test
TEST_IMAGE = "road4.jpg"

def draw_hud(img, steer_deg, speed, state, left_poly, right_poly, goal_point=None):
    h, w = img.shape[:2]
    display_img = img.copy()

    # 1. Draw Info Panel
    cv2.rectangle(display_img, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(display_img, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_img, f"Speed: {speed:.1f}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_img, f"Steer: {steer_deg:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

    # 2. Draw Lane Availability
    l_status = "YES" if left_poly else "NO"
    r_status = "YES" if right_poly else "NO"
    cv2.putText(display_img, f"L:{l_status} R:{r_status}", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # 3. Visualize Steering (Blue Stick)
    center_x, center_y = w // 2, h
    line_len = 150
    angle_rad = np.deg2rad(steer_deg - 90) 
    
    end_x = int(center_x + line_len * np.cos(angle_rad))
    end_y = int(center_y + line_len * np.sin(angle_rad))
    
    cv2.line(display_img, (center_x, center_y), (end_x, end_y), (255, 0, 0), 5)
    cv2.circle(display_img, (center_x, center_y), 10, (0, 0, 255), -1)

    if goal_point:
        gx, gy = goal_point
        scale = 3.0 
        px = int(center_x + (gx * scale))
        py = int(h - (gy * scale))
        py = max(0, min(h-1, py)) # Clip to screen
        px = max(0, min(w-1, px))
        
        cv2.circle(display_img, (px, py), 10, (0, 255, 0), -1)
        cv2.putText(display_img, "TARGET", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return display_img

def main():
    
    # 1. Initialize Controller
    controller = LaneController(
        wheelbase=WHEELBASE, 
        max_steering_angle=25, 
        max_speed=25.0, 
        min_speed=10.0,
        lane_width=LANE_WIDTH
    )

    # 2. Load Image
    frame = cv2.imread(TEST_IMAGE)
    if frame is None:
        print(f"Error: Could not load {TEST_IMAGE}. Check filename.")
        return

    frame = cv2.resize(frame, (640, 480))

    try:
        print("Running Lane Detection...")
        path, left_poly, right_poly = run_lane_detect(
            frame, 
            h=CAMERA_HEIGHT, 
            theta=CAMERA_PITCH, 
            f=FOCAL_LENGTH, 
            k=1.0,
            use_deprojected=True
        )

        steer, speed, state = controller.get_control(left_poly, right_poly, current_speed=15.0)

        goal_point = None
        target_poly, offset_mode = controller._select_target_path(left_poly, right_poly)
        if target_poly:
            lookahead = controller.min_lookahead + (controller.k_lookahead * (15.0 - 10))
            lookahead = max(controller.min_lookahead, min(lookahead, 120))
            tx = target_poly(lookahead)
            if offset_mode == "RIGHT_OFFSET": tx += (LANE_WIDTH / 2.0)
            elif offset_mode == "LEFT_OFFSET": tx -= (LANE_WIDTH / 2.0)
            goal_point = (tx, lookahead)

        # 6. Visualization
        print("\n--- RESULTS ---")
        print(f"Steering Angle: {steer:.2f} degrees")
        print(f"Target Speed:   {speed:.2f}")
        print(f"Robot State:    {state}")
        print(f"Left Lane:      {'FOUND' if left_poly else 'MISSING'}")
        print(f"Right Lane:     {'FOUND' if right_poly else 'MISSING'}")

        result_img = draw_hud(frame, steer, speed, state, left_poly, right_poly, goal_point)
        
        cv2.imshow("Debug View (Press Key to Exit)", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
