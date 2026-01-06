import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class ImageFeeder(Node):
    def __init__(self, image_path):
        super().__init__('image_feeder')
        self.publisher_ = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )
        
        self.bridge = CvBridge()
        
        if not os.path.exists(image_path):
            self.get_logger().info(f"File not found: {image_path}")
            exit(1)
            
        self.cv_image = cv2.imread(image_path)
        self.cv_image = cv2.resize(self.cv_image, (640, 480))
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(f"Publishing {image_path} to /camera/image_raw...")
        
    def timer_callback(self):
        msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)
        
def main(args = None):
    rclpy.init(args=args)
    node = ImageFeeder('road7.jpeg')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    main()