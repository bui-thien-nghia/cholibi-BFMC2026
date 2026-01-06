import numpy as np
import math

class LaneController:
    def __init__(self, 
                 wheelbase=26.5,
                 max_steering_angle=25,
                 max_speed=25.0, 
                 min_speed=10.0,
                 lane_width=37.0):
        
        self.L = wheelbase
        self.max_steer = np.deg2rad(max_steering_angle)
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.lane_width = lane_width

        # tuning parameters
        self.k_lookahead = 1.0     
        self.min_lookahead = 40.0
        self.lookahead_radius = 0  
        self.k_speed = 30.0       
        self.alpha = 0.6          
        
        self.prev_steer = 0.0
        self.no_lane_counter = 0 
        self.PATIENCE_LIMIT = 15   

    def get_control(self, left_poly, right_poly, current_speed=15.0):
        target_poly, offset_mode = self._select_target_path(left_poly, right_poly)

        if target_poly is None:
            self.no_lane_counter += 1
            if self.no_lane_counter < self.PATIENCE_LIMIT:
                return self.prev_steer, self.min_speed, "FREE"
            else:
                return 0.0, 0.0, "NO_LANE"
        
        self.no_lane_counter = 0 

        self.lookahead_radius = self.min_lookahead + (self.k_lookahead * (current_speed))
        self.lookahead_radius = max(self.min_lookahead, min(self.lookahead_radius, 120)) 

        try:
            gx, gy = self._find_circle_intersection(target_poly, self.lookahead_radius, offset_mode)
            
        except Exception as e:
            return self.prev_steer, 0.0, "MATH_ERR"

        alpha_angle = math.atan2(gx, gy) 
        
        # Steering Angle = atan(2 * L * sin(alpha) / Ld)
        steer_rad = math.atan((2 * self.L * math.sin(alpha_angle)) / self.lookahead_radius)
        steer_rad = np.clip(steer_rad, -self.max_steer, self.max_steer)
        
        # 4. Smoothing
        steer_rad = (self.alpha * steer_rad) + ((1 - self.alpha) * self.prev_steer)
        self.prev_steer = steer_rad
        
        # 5. Speed Control
        curvature = abs(steer_rad)
        target_speed = self.max_speed / (1 + self.k_speed * curvature)
        target_speed = max(self.min_speed, min(self.max_speed, target_speed))

        # 6. State
        steer_deg = np.rad2deg(steer_rad)
        state = "STRAIGHT"
        if steer_deg > 3: state = "RIGHT_TURN"
        elif steer_deg < -3: state = "LEFT_TURN"

        return steer_deg, target_speed, state

    def _select_target_path(self, left_poly, right_poly):
        if left_poly is not None and right_poly is not None:
            # Average the coefficients for center path
            avg_coeffs = (left_poly.coeffs + right_poly.coeffs) / 2
            return np.poly1d(avg_coeffs), "CENTER"
        elif left_poly is not None:
            return left_poly, "RIGHT_OFFSET" 
        elif right_poly is not None:
            return right_poly, "LEFT_OFFSET"
        return None, "NONE"

    def _find_circle_intersection(self, poly, radius, mode):
        # search_points = np.linspace(10, radius + 20, 1000)
        
        # best_point = (0, radius) 
        # min_dist_diff = float('inf')

        # for y in search_points:
        #     # Calculate X at this Y
        #     raw_x = poly(y)
            
        #     # Apply Offset 
        #     if mode == "RIGHT_OFFSET":
        #         raw_x += (self.lane_width * 0.5) 
        #     elif mode == "LEFT_OFFSET":
        #         raw_x -= (self.lane_width * 0.5)

        #     # Check Distance from Car
        #     dist_from_car = math.sqrt(raw_x**2 + y**2)

        #     diff = abs(dist_from_car - radius)
            
        #     if diff < min_dist_diff:
        #         min_dist_diff = diff
        #         best_point = (raw_x, y)
            
        #     if dist_from_car > radius:
        #         break
                
        # return best_point
    
        new_coeffs = poly.coeffs.copy()

        if mode == "RIGHT_OFFSET":
            new_coeffs[-1] += (self.lane_width * 0.5)
        elif mode == "LEFT_OFFSET":
            new_coeffs[-1] -= (self.lane_width * 0.5)
        
        poly1 = new_coeffs
        poly2 = [1,0,0]
        combined_poly = np.polyadd(np.polymul(poly1,poly1), poly2)
        combined_poly[-1] -= radius**2
        search_points = np.roots(combined_poly)
        search_points = search_points[np.isreal(search_points)].real
        forward_points = search_points[search_points > 0]
        
        if len(forward_points) == 0:
            return (0, radius)
        
        y = np.min(forward_points)
        x = np.polyval(new_coeffs, y)
        best_point = (x,y)
        return best_point
