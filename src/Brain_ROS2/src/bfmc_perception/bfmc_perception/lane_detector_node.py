import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from bfmc_interfaces.msg import LaneInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .lane_detect import run_lane_detect

class LaneDetector(Node):
    def __init__(self):
        super().__init__('lane_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(
            LaneInfo,
            '/perception/lane',
            10
        )
        
        self.bridge = CvBridge()
        self.get_logger().info("Lane Detector Node started...")
        
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.resize(frame, (640, 480))
            h = 20.5
            theta = 25.0
            f = 273.0
            
            _, left_poly, right_poly = run_lane_detect(frame, h=h, theta=theta, f=f, k=1.0, use_deprojected=True)
            
            lane_msg = LaneInfo()
            lane_msg.detected = False
            
            if left_poly is not None:
                lane_msg.left_found = True
                lane_msg.left_coeffs = [float(c) for c in left_poly.coeffs]
                lane_msg.detected = True
            else:
                lane_msg.left_found = False
                
            if right_poly is not None:
                lane_msg.right_found = True
                lane_msg.right_coeffs = [float(c) for c in right_poly.coeffs]
                lane_msg.detected = True
            else:
                lane_msg.right_found = False
                
            self.publisher_.publish(lane_msg)
            
        except Exception as e:
            self.get_logger().error(f"Lane Detection Error: {e}")

def main(args = None):
    rclpy.init(args=args)
    node = LaneDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()