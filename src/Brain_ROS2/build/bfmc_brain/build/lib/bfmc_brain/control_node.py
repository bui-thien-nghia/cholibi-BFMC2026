import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from bfmc_interfaces.msg import LaneInfo
from std_msgs.msg import Header
import numpy as np

from .lane_keeping import LaneController

class LaneKeepingNode(Node):
    def __init__(self):
        super().__init__('lane_keeping_node')
        
        self.sub = self.create_subscription(
            LaneInfo, 
            '/perception/lane',
            self.lane_callback,
            10
        )
        
        self.pub_cmd = self.create_publisher(
            Twist, 
            '/cmd_vel',
            10
        )
        
        self.pub_debug_point = self.create_publisher(
            PointStamped,
            '/debug/lookahead_point',
            10
        )
        self.controller = LaneController(
            wheelbase=26.5,
            max_steering_angle=25.0,
            max_speed=25.0, 
            min_speed=10.0,
            lane_width=37.0
        )
        
        self.current_speed_sim = 15.0
        
        self.get_logger().info("Lane Keeping Controller Started...")
        
    def lane_callback(self, msg):
        
        left_poly = None
        right_poly = None
        
        if msg.left_found and len(msg.left_coeffs) > 0:
            left_poly = np.poly1d(msg.left_coeffs)
            self.get_logger().info(f"DEBUG: Left Coeff:{msg.left_coeffs}")
            
        if msg.right_found and len(msg.right_coeffs) > 0:
            right_poly = np.poly1d(msg.right_coeffs)
            
        # --- SUPER IMPORTANT --- #
        self.controller.prev_steer = 0.0
        # --- SUPER IMPORTANT --- #
        
        steer_deg, speed, state = self.controller.get_control(
            left_poly=left_poly,
            right_poly=right_poly,
            current_speed=self.current_speed_sim
        )
        
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(steer_deg)
        
        self.pub_cmd.publish(cmd)
        
        self.publish_debug_point(left_poly, right_poly)
        
        # Debug:
        # self.get_logger().info(f"State: {state} | Speed: {speed:.1f} | Steer: {steer_deg:.1f}")
        
    def publish_debug_point(self, left_poly, right_poly):
        target_poly, offset_mode = self.controller._select_target_path(left_poly, right_poly)        
        if target_poly:
            lookahead = self.controller.min_lookahead + (self.controller.k_lookahead * (self.current_speed_sim - 10))
            lookahead = max(self.controller.min_lookahead, min(lookahead, 120))
            
            LANE_WIDTH = self.controller.lane_width
            tx = target_poly(lookahead)
            
            if offset_mode == "RIGHT_OFFSET": 
                tx += (LANE_WIDTH / 2.0)
            elif offset_mode == "LEFT_OFFSET": 
                tx -= (LANE_WIDTH / 2.0)

            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "camera_frame"
            
            point_msg.point.x = float(tx)
            point_msg.point.y = float(lookahead)
            point_msg.point.z = float(0)
            
            self.pub_debug_point.publish(point_msg)        
def main(args=None):
    rclpy.init(args=args)
    node = LaneKeepingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()