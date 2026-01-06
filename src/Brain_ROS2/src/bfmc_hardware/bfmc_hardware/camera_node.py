import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import io

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.publisher_ = self.create_publisher(Image, "/camera/image_raw", 10)
        timer_period = 0.033
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.get_logger().info("Err: Cannot open camera")
        else:
            self.get_logger().info("Camera Node started sucessfully")
            
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Err: Cannot converting image: {e}")
        else:
            self.get_logger().warn("Failed to capture image")
            
    def destroy_node(self):
        self.cap.release()
        return super().destroy_node()
    
def main(args = None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == "__main__":
    main()